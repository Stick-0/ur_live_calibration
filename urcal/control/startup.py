"""Receive-only alignment readiness and the explicitly armed native preflight.

No controller is constructed here. The caller owns its MotionSession, checks
fresh paired vision/robot data, and closes that session on every failure.
"""

import time
from copy import deepcopy

import numpy as np

from urcal.control.geometry import flange_from_tcp, inverse, pose_delta, rotation_distance
from urcal.control.policy import CalibrationUnavailable, ControlFault, PlanningUnavailable, stationary_velocity_limits


ROBOTS = ("ur5e", "ur10e")
CALIBRATION_WAIT_SECONDS = 5.
CONDITIONING_SEARCH_SCALES = (2., 4., 6.)


def endpoint_conditioning(args, offsets, states, targets, seeds, goals, native_tcp=None):
    """Evaluate returned IK joints with the alignment and live tracking models.

    A search seed is only a proposal. Small kinematic model differences near a
    singularity can move the returned solution substantially, without a change
    of IK branch. Check it before spending time on the native path samples.
    """
    from urcal.control.alignment import CapturedKinematics
    from urcal.cooperative.kinematics import NominalKinematics

    minimum = float(args.min_singular_value)
    result = {"minimum_singular_value": minimum, "robots": {}, "accepted": True}
    for name, q in goals.items():
        captured = CapturedKinematics(name, states[name].joints, states[name].tcp, offsets[name])
        tracking = NominalKinematics(name, offsets[name])
        _, seed_jacobian = captured.pose_and_jacobian(seeds[name])
        _, captured_jacobian = captured.pose_and_jacobian(q)
        nominal, tracking_jacobian = tracking.pose_and_jacobian(q)
        actual = native_tcp[name] if isinstance(native_tcp, dict) and name in native_tcp else targets[name]
        inverse(actual)
        # Match CooperativeController.prepare(): its short-horizon Jacobian is
        # shifted from the nominal TCP origin to the measured controller TCP.
        tracking_jacobian[:3] += np.cross(tracking_jacobian[3:].T, actual[:3, 3] - nominal[:3, 3]).T
        seed_sigma = float(np.linalg.svd(seed_jacobian, compute_uv=False)[-1])
        captured_sigma = float(np.linalg.svd(captured_jacobian, compute_uv=False)[-1])
        tracking_sigma = float(np.linalg.svd(tracking_jacobian, compute_uv=False)[-1])
        accepted = min(captured_sigma, tracking_sigma) >= minimum
        result["robots"][name] = {
            "seed_q_rad": np.asarray(seeds[name]).copy(), "native_q_rad": np.asarray(q).copy(),
            "native_tcp": np.asarray(actual).copy(),
            "seed_singular_value": seed_sigma, "captured_singular_value": captured_sigma,
            "tracking_singular_value": tracking_sigma,
            "seed_to_native_joint_distance_rad": float(np.linalg.norm(np.asarray(q) - seeds[name])),
            "accepted": accepted,
        }
        result["accepted"] = result["accepted"] and accepted
    return result


class AlignmentRequestFault(PlanningUnavailable):
    """A rejected preview request with numerical context for the experiment log."""

    category = "initial_alignment"

    def __init__(self, message, details):
        self.details = deepcopy(details)
        super().__init__(message)


class AlignmentCalibrationPending(ControlFault):
    """Fresh encoder states, but vision is rejected, stale or still warming up.

    Only an idle preflight may wait on this signal. The attached bases are
    always the last accepted poses; rejected observations never replace them.
    Encoder failures and operator stops propagate as ordinary faults.
    """

    def __init__(self, reason, states, bases=None):
        super().__init__(reason)
        self.states, self.bases = states, bases


class AlignmentCalibrationTimeout(ControlFault):
    """Bounded pre-motion wait expired; a new explicit Arm is required."""

    category = "arm_calibration"


def preflight_calibration_bases(tracker, mappings, states, now):
    """Require stable accepted vision without admitting a rejected pose.

    The caller must obtain fresh encoders and handle Stop before calling this.
    Pending vision carries the accepted bases for original-Arm drift checks.
    It is only waitable before any watchdog is enabled or velocity is sent.
    """
    bases = {name: tracker.poses[name] @ mappings[name] for name in ROBOTS}
    if tracker.last_rejection:
        raise AlignmentCalibrationPending(
            "rejected camera pair: " + tracker.last_rejection, states, bases)
    try:
        tracker.check_fresh(now)
    except ControlFault as exc:
        raise AlignmentCalibrationPending(str(exc), states, bases) from exc
    return bases


def check_stationary(states, args=None):
    for name in ROBOTS:
        state = states[name]
        qd, speed = np.asarray(state.joint_velocity), np.asarray(state.speed)
        if (qd.shape != (6,) or speed.shape != (6,)
                or not np.isfinite([qd, speed]).all()):
            raise ControlFault(name + ": invalid initial alignment velocity")
        fixed_leader = getattr(args, "mode", None) == "follow" and name == getattr(args, "leader", None)
        joint, linear, angular = stationary_velocity_limits(args, leader=fixed_leader)
        if (np.max(np.abs(qd)) > joint or np.linalg.norm(speed[:3]) > linear
                or np.linalg.norm(speed[3:]) > angular):
            raise ControlFault("initial alignment requires both robots and bases to remain stationary")


def check_start_snapshot(args, original_bases, original_states, bases, states,
                         *, phase="initial alignment preflight"):
    """Preflight must never silently recapture a moving robot or moving base."""
    check_stationary(states, args)
    for name in ROBOTS:
        if (np.linalg.norm(bases[name][:3, 3] - original_bases[name][:3, 3])
                > min(.01, args.max_calibration_jump_m)
                or rotation_distance(bases[name], original_bases[name])
                > np.deg2rad(min(1., args.max_calibration_jump_deg))):
            change = "base moved" if phase == "initial alignment preflight" else "base calibration changed"
            fault = CalibrationUnavailable if phase == "post-alignment calibration handoff" else ControlFault
            raise fault(name + ": " + change + " during " + phase)
    _check_encoder_snapshot(original_states, states, phase)


def _check_encoder_snapshot(original_states, states, phase):
    for name in ROBOTS:
        dp, dr = pose_delta(original_states[name].tcp, states[name].tcp)
        if (np.max(np.abs(states[name].joints - original_states[name].joints)) > .01
                or np.linalg.norm(dp) > .003 or np.linalg.norm(dr) > np.deg2rad(.5)):
            raise ControlFault(name + ": robot moved during " + phase)


def check_handoff_snapshot(args, original_bases, original_states, bases, states,
                           *, offsets, nominal_targets, live_targets):
    """Admit a fresh camera baseline by its task effect, never by moving a robot.

    The caller has already accumulated a new stable, fresh calibration cluster.
    Across that interval camera estimation error need not meet the preflight
    10 mm / 1 degree freeze check. Instead its change to each nominal flange
    command must fit the separately configured handoff allowance. This does not
    change the task tolerance or recapture the nominal goal. The next control
    proposal still checks actual task error, geometry and every motion bound.
    """
    phase = "post-alignment calibration handoff"
    # Older saved settings retain their original task-based handoff limits.
    position = float(getattr(args, "handoff_max_shift_m", getattr(args, "target_tolerance_m", 0.)))
    angle = np.deg2rad(float(getattr(args, "handoff_max_shift_deg", getattr(args, "target_tolerance_deg", 0.))))
    if not np.isfinite([position, angle]).all() or min(position, angle) < 0:
        raise ControlFault("invalid handoff allowance during " + phase)
    if position == 0 and angle == 0:
        return check_start_snapshot(args, original_bases, original_states, bases, states, phase=phase)
    check_stationary(states, args)
    _check_encoder_snapshot(original_states, states, phase)
    if not nominal_targets or set(nominal_targets) != set(live_targets):
        raise ControlFault("missing nominal target during " + phase)
    for name in ROBOTS:
        inverse(bases[name])
    for name in nominal_targets:
        original = flange_from_tcp(nominal_targets[name], offsets[name])
        current = flange_from_tcp(live_targets[name], offsets[name])
        dp, dr = pose_delta(original, current)
        distance, rotation = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        # A zero component requests precise control, not bit-identical camera
        # matrices. Use the same numerical floor as native FK validation.
        position_limit = position if position else .0005 + .002 * np.linalg.norm(offsets[name][:3, 3])
        angle_limit = angle if angle else .002
        if distance > position_limit + 1e-9 or rotation > angle_limit + 1e-9:
            raise CalibrationUnavailable(
                f"{name}: calibration handoff shifts nominal flange target by "
                f"{distance * 1000:.1f}mm / {np.rad2deg(rotation):.2f}deg, beyond "
                f"{position_limit * 1000:g}mm / {np.rad2deg(angle_limit):g}deg handoff allowance")


def check_alignment_request(args, offsets, bases, states, targets, models, guard, origins=None):
    """Allow Arm for native IK/path validation, without allowing a velocity."""
    check_stationary(states, args)
    flanges = {}
    for name in ROBOTS:
        inverse(bases[name])
        inverse(states[name].tcp)
        expected, _ = models[name].pose_and_jacobian(states[name].joints)
        dp, dr = pose_delta(expected, states[name].tcp)
        if np.linalg.norm(dp) > args.frame_check_m or np.linalg.norm(dr) > np.deg2rad(args.frame_check_deg):
            raise ControlFault(name + ": nominal FK / controller TCP mismatch")
        flanges[name] = bases[name] @ flange_from_tcp(states[name].tcp, offsets[name])
    if np.linalg.norm(flanges[ROBOTS[0]][:3, 3] - flanges[ROBOTS[1]][:3, 3]) < args.min_gap:
        raise PlanningUnavailable("initial alignment flange separation below min_gap")
    for name, target in targets.items():
        inverse(target)
        origin = states[name].tcp if origins is None else origins[name]
        for checked_pose, pose in (("current", states[name].tcp), ("target", target)):
            displacement = pose[:3, 3] - origin[:3, 3]
            distance = float(np.linalg.norm(displacement))
            if distance > args.max_travel:
                raise AlignmentRequestFault(
                    f"{name}: initial alignment {checked_pose} outside start-position envelope "
                    f"(distance={distance * 1000:.1f}mm, limit={args.max_travel * 1000:.1f}mm)",
                    {"robot": name, "checked_pose": checked_pose, "distance_m": distance,
                     "limit_m": args.max_travel, "displacement_m": displacement,
                     "origin_tcp": origin, "measured_tcp": states[name].tcp, "target_tcp": target})
        if not guard.pose_allowed(name, target):
            raise PlanningUnavailable(name + ": initial alignment target outside TCP workspace")
    allowed, reason, _ = guard.check(bases, {name: states[name].joints for name in ROBOTS})
    if not allowed:
        raise PlanningUnavailable("initial alignment: " + reason)


def prepare_alignment(session, args, offsets, bases, states, targets, read_current,
                      report=None, *, wait=None, clock=None, diagnostic=None):
    """Native IK, sampled joint safety/FK, then watchdog; never send a velocity.

    Live callers return their frozen Arm calibration and fresh encoder states
    during the potentially slow controller preflight. Encoder/stop/process
    failures and motion away from the Arm snapshot propagate immediately.
    Other callers may signal AlignmentCalibrationPending before watchdog setup
    only; that legacy path has a total five-second camera recovery bound.
    The original Arm snapshot and nominal target are never recaptured. A failed
    exact IK may select a separate command within configured flange tolerances.
    The next runtime
    tick must revalidate before speedJ.
    """
    from urcal.control.alignment import LiveAlignment
    from urcal.control.runtime import ArmConditioningUnavailable, ArmIKUnavailable

    wait = time.sleep if wait is None else wait
    clock = time.monotonic if clock is None else clock
    remaining_wait = CALIBRATION_WAIT_SECONDS

    def checked(allow_wait):
        nonlocal remaining_wait
        waiting_since, last_report = None, None
        reason = ""

        def timed_out():
            raise AlignmentCalibrationTimeout(
                f"initial alignment calibration wait exceeded {CALIBRATION_WAIT_SECONDS:g}s: "
                + reason + ". No motion started; press Arm again when calibration is ready.")

        while True:
            try:
                current_bases, current_states = read_current()
            except AlignmentCalibrationPending as exc:
                # Even without usable vision, fresh encoders must show both
                # robots still at the original Arm snapshot, not just at rest.
                check_start_snapshot(args, bases, states,
                                     bases if exc.bases is None else exc.bases, exc.states)
                if not allow_wait:
                    raise
                now = clock()
                if waiting_since is None:
                    waiting_since = now
                reason = str(exc)
                left = remaining_wait - (now - waiting_since)
                if left <= 0:
                    timed_out()
                if report is not None and (last_report is None or now - last_report >= 1.):
                    report("PREFLIGHT WAIT: " + reason + "; no motion yet")
                    last_report = now
                wait(min(.02, left))
                continue
            check_start_snapshot(args, bases, states, current_bases, current_states)
            if waiting_since is not None:
                remaining_wait -= clock() - waiting_since
                if remaining_wait < 0:
                    timed_out()
                if report is not None:
                    report("PREFLIGHT: calibration stable; continuing controller checks; no motion yet")
            return

    def fresh():
        checked(allow_wait=True)

    fresh()
    if report is not None:
        report("PREFLIGHT: controller IK and joint path checks; no motion yet")
    nominal_targets = {name: target.copy() for name, target in targets.items()}
    selected_targets = {name: target.copy() for name, target in targets.items()}
    adjustments = {"adjustments": {}}
    approximate = False
    position = float(getattr(args, "target_tolerance_m", 0.))
    angle = np.deg2rad(float(getattr(args, "target_tolerance_deg", 0.)))
    tolerant = position > 0 or angle > 0
    attempts, last_ik_error = [], None
    conditioning_failed = False

    def record_attempt(record):
        attempts.append(deepcopy(record))
        if diagnostic is not None:
            diagnostic(deepcopy(record))

    def native_candidate(candidate, seeds, attempt, scale):
        nonlocal last_ik_error, conditioning_failed
        record = {"attempt": attempt, "conditioning_scale": scale,
                  "nominal_tcp_goals": nominal_targets, "command_tcp_goals": candidate,
                  "seed_joints": seeds}
        try:
            solved = session.solve_joint_targets(candidate, seeds, fresh, report=report)
        except ArmIKUnavailable as exc:
            last_ik_error = exc
            record.update(outcome="native_ik_unavailable", reason=str(exc))
            record_attempt(record)
            return None
        if tolerant:
            fresh()
            condition = endpoint_conditioning(args, offsets, states, candidate, seeds, solved,
                                               getattr(session, "last_ik_tcp", None))
            record["conditioning"] = condition
            if report is not None:
                for name, values in condition["robots"].items():
                    report(f"PREFLIGHT: {name.upper()} returned IK conditioning: "
                           f"alignment={values['captured_singular_value']:.6g}, "
                           f"tracking={values['tracking_singular_value']:.6g}, "
                           f"minimum={args.min_singular_value:g}; no motion yet")
            if not condition["accepted"]:
                conditioning_failed = True
                record.update(outcome="conditioning_rejected", reason="returned IK is too near a tracking singularity")
                record_attempt(record)
                if report is not None:
                    report("PREFLIGHT: returned IK rejected before path checks; trying another bounded candidate; no motion yet")
                return None
        record.update(outcome="accepted", native_joints=solved)
        record_attempt(record)
        return solved

    goals = native_candidate(selected_targets, {name: states[name].joints for name in targets}, 0, None)
    if goals is None:
        if not tolerant:
            raise last_ik_error
        from urcal.control.approximation import (
            TargetApproximationUnavailable, flange_adjustment, select_alignment_candidate,
        )
        if report is not None:
            report(f"PREFLIGHT: exact pose IK unavailable or unsuitable for tracking; searching within flange allowance "
                   f"{position * 1000:g}mm / {np.rad2deg(angle):g}deg; no motion yet")
        search_failure = ""
        for attempt, scale in enumerate(CONDITIONING_SEARCH_SCALES, 1):
            if report is not None:
                report(f"PREFLIGHT: bounded candidate {attempt}/{len(CONDITIONING_SEARCH_SCALES)}, "
                       f"search conditioning preference={scale * args.min_singular_value:g}; no motion yet")
            try:
                selected_targets, seeds, adjustments = select_alignment_candidate(
                    args, offsets, bases, states, nominal_targets, check_fresh=fresh, conditioning_scale=scale)
            except TargetApproximationUnavailable as exc:
                search_failure = str(exc)
                record_attempt({"attempt": attempt, "conditioning_scale": scale,
                                "outcome": "no_bounded_candidate", "reason": search_failure})
                continue
            for name, target in selected_targets.items():
                distance, rotation = flange_adjustment(nominal_targets[name], target, offsets[name])
                if distance > position + 1e-9 or rotation > angle + 1e-9:
                    raise ControlFault(name + ": approximate command exceeds nominal flange allowance")
                if report is not None:
                    report(f"PREFLIGHT: {name.upper()} command adjusted {distance * 1000:.1f}mm / "
                           f"{np.rad2deg(rotation):.2f}deg from nominal; validating native IK/FK; no motion yet")
            goals = native_candidate(selected_targets, seeds, attempt, scale)
            if goals is not None:
                approximate = True
                break
        if goals is None:
            if conditioning_failed:
                raise ArmConditioningUnavailable({
                    "position_tolerance_m": position, "rotation_tolerance_deg": float(np.rad2deg(angle)),
                    "max_attempts": len(CONDITIONING_SEARCH_SCALES), "attempts": attempts,
                    "reason": "No returned IK passed the unchanged tracking conditioning limit.",
                })
            last_ik_error.details["alignment_attempts"] = deepcopy(attempts)
            if search_failure:
                last_ik_error.details["approximate_target_failure"] = search_failure
                last_ik_error.args = (str(last_ik_error) + " Bounded target search: " + search_failure,)
            raise last_ik_error
    fresh()
    if report is not None:
        report("PREFLIGHT: building geometric joint path; no motion yet")
    alignment = LiveAlignment(args, offsets, bases, states, selected_targets, goals)
    alignment.nominal_targets = nominal_targets
    alignment.target_adjustments = adjustments
    alignment.target_adjustments["native_attempts"] = deepcopy(attempts)
    if report is not None:
        speed_scale = getattr(args, "align_speed_scale", 1.)
        fraction = alignment.command_speed_fraction
        report(f"PREFLIGHT: alignment speed x{speed_scale:g}; command caps "
               f"{fraction * args.max_speed * 1000:g}mm/s, "
               f"{np.rad2deg(fraction * args.max_angular_speed):.2f}deg/s, "
               f"joint {fraction * args.max_joint_speed:g}rad/s "
               f"({fraction:.0%} of hard limits; measured limits unchanged); no motion yet")
    fresh()
    poses = session.validate_joint_path(alignment.samples, fresh, report=report)
    if len(poses) != len(alignment.samples):
        raise ControlFault("controller returned incomplete initial alignment path")
    # Controller-calibrated FK independently checks the proxy path, tool origin
    # travel and flange gap; geometric link capsules remain an approximation.
    for sample, calibrated in zip(alignment.samples, poses):
        flanges = {}
        for name in ROBOTS:
            tcp = calibrated[name] if name in targets else states[name].tcp
            inverse(tcp)
            proxy, _ = alignment.models[name].pose_and_jacobian(sample[name])
            dp, dr = pose_delta(proxy, tcp)
            if (np.linalg.norm(dp) > args.frame_check_m
                    or np.linalg.norm(dr) > np.deg2rad(args.frame_check_deg)):
                raise ControlFault(name + ": calibrated FK differs from alignment path model")
            flange = flange_from_tcp(tcp, offsets[name])
            for pose, initial in ((tcp, states[name].tcp),
                                  (flange, flange_from_tcp(states[name].tcp, offsets[name]))):
                if np.linalg.norm(pose[:3, 3] - initial[:3, 3]) > args.max_travel:
                    raise PlanningUnavailable(name + ": controller FK path exceeds travel envelope")
            if not alignment.guard.pose_allowed(name, tcp):
                raise PlanningUnavailable(name + ": controller FK path outside TCP workspace")
            flanges[name] = bases[name] @ flange
        if np.linalg.norm(flanges[ROBOTS[0]][:3, 3] - flanges[ROBOTS[1]][:3, 3]) < args.min_gap:
            raise PlanningUnavailable("controller FK path flange separation below min_gap")
    if approximate:
        # Independently check the native endpoint. In particular a nonzero TCP
        # offset couples native angular residual into flange position error.
        for name in selected_targets:
            distance, rotation = flange_adjustment(nominal_targets[name], poses[-1][name], offsets[name])
            numerical_position = .0005 + .002 * np.linalg.norm(offsets[name][:3, 3])
            if (distance > (position if position else numerical_position) + 1e-9
                    or rotation > (angle if angle else .002) + 1e-9):
                raise ControlFault(name + ": native endpoint exceeds nominal flange allowance")
    fresh()
    if report is not None:
        report("PREFLIGHT: joint path validated; enabling watchdog; no motion yet")
    # A hold session enables two watchdogs in sequence. Never wait after even
    # the first is set, regardless of the session's aggregate enabled flag.
    session.enable_watchdog(check_fresh=lambda: checked(allow_wait=False))
    return alignment
