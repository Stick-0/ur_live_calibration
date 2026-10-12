import numpy as np
import cv2
from scipy.optimize import least_squares
from pyrr import Quaternion

def convert_rvec_to_quaternion(rvec):
    """Convert rvec (which is log quaternion) to quaternion"""
    theta = np.sqrt(
        rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]
    )  # in radians
    raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

    # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
    quaternion = Quaternion.from_axis_rotation(raxis, theta)
    quaternion.normalize()
    return quaternion

def solve_pnp(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    refinement=True,
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec = cv2.solvePnP(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            flags=method,
        )

        if refinement:
            pnp_retval, rvec, tvec = cv2.solvePnP(
                canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
                projections_proc.reshape(projections_proc.shape[0], 1, 2),
                camera_K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

    except:
        pnp_retval = False
        translation = None
        quaternion = None

    return pnp_retval, translation, quaternion


def solve_pnp_ransac(
    canonical_points,
    projections,
    camera_K,
    method=cv2.SOLVEPNP_EPNP,
    inlier_thresh_px=5.0,  # this is the threshold for each point to be considered an inlier
    dist_coeffs=np.array([]),
):

    n_canonial_points = len(canonical_points)
    n_projections = len(projections)
    assert (
        n_canonial_points == n_projections
    ), "Expected canonical_points and projections to have the same length, but they are length {} and {}.".format(
        n_canonial_points, n_projections
    )

    # Process points to remove any NaNs
    canonical_points_proc = []
    projections_proc = []
    for canon_pt, proj in zip(canonical_points, projections):

        if (
            canon_pt is None
            or len(canon_pt) == 0
            or canon_pt[0] is None
            or canon_pt[1] is None
            or proj is None
            or len(proj) == 0
            or proj[0] is None
            or proj[1] is None
        ):
            continue

        canonical_points_proc.append(canon_pt)
        projections_proc.append(proj)

    # Return if no valid points
    if len(canonical_points_proc) == 0:
        return False, None, None, None

    canonical_points_proc = np.array(canonical_points_proc)
    projections_proc = np.array(projections_proc)

    # Use cv2's PNP solver
    try:
        pnp_retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
            projections_proc.reshape(projections_proc.shape[0], 1, 2),
            camera_K,
            dist_coeffs,
            reprojectionError=inlier_thresh_px,
            flags=method,
        )

        translation = tvec[:, 0]
        quaternion = convert_rvec_to_quaternion(rvec[:, 0])

    except:
        pnp_retval = False
        translation = None
        quaternion = None
        inliers = None

    return pnp_retval, translation, quaternion, inliers

class EMPnP:
    def __init__(self, K, distCoeffs=None, sigma=3.0, image_size=(640, 480)):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.zeros(5) if distCoeffs is None else np.asarray(distCoeffs, dtype=np.float64)
        self.sigma2 = sigma**2

        self.image_area = image_size[0] * image_size[1]

        self.rvec = None
        self.tvec = None
        self.w = None
        self.reproj_errors = None

    def project(self, rvec, tvec, X):
        pts, _ = cv2.projectPoints(X, rvec, tvec, self.K, self.dist)
        return pts.reshape(-1, 2)

    def e_step(self, rvec, tvec, X, u, u_pre, lamda=0.9):
        proj = self.project(rvec, tvec, X)

        sq_pre = np.sum((u_pre - proj) ** 2, axis=1)
        cor_prob = np.exp(-sq_pre / (2 * self.sigma2))
        correction = np.multiply(cor_prob[:, np.newaxis], (lamda * (u_pre - u) + (1 - lamda) * (proj - u)))
        u = u + correction
        sq = np.sum((u - proj)**2, axis=1)
        L_in = np.exp(-sq / (2*self.sigma2)) / (2*np.pi*self.sigma2)
        L_out = (1.0 / self.image_area) * np.ones_like(L_in)

        n_points = self.pi_prior.shape[0]
        pi_post = np.divide(
            (np.diag(self.pi_prior) @ L_in),
            (np.diag(self.pi_prior) @ L_in + (np.identity(n_points) - np.diag(self.pi_prior)) @ L_out + 1e-12),
        )
        if self.pi_prior.ndim == 2:
            self.pi_prior = np.diag(self.pi_prior)
        gamma = 0.5
        self.pi_prior = gamma * self.pi_prior + (1 - gamma) * pi_post
        denom = np.max(self.pi_prior) - np.min(self.pi_prior)
        if denom > 1e-12:
            self.pi_prior = (self.pi_prior - np.min(self.pi_prior)) / denom
        w = self.pi_prior
        w[w < 0] = 0

        return w, proj, u

    def m_step(self, rvec, tvec, X, u, w):
        def residuals(params):
            r, t = params[:3], params[3:6]
            proj = self.project(r, t, X)
            res = (u - proj) * np.sqrt(w)[:, None]
            return res.ravel()

        p0 = np.hstack([rvec, tvec])
        res = least_squares(residuals, p0, method='lm')
        return res.x[:3], res.x[3:6]

    def fit(self, X3d, u2d, u2d_pre, pi_prior=0.9, max_iter=10, tol=1e-6, use_ransac_init=True, lamda=0.9):
        X = np.asarray(X3d, dtype=np.float64)
        u = np.asarray(u2d, dtype=np.float64)
        u_pre = np.asarray(u2d_pre, dtype=np.float64)

        n_points = X.shape[0]
        if np.isscalar(pi_prior):
            self.pi_prior = np.full(n_points, float(pi_prior), dtype=np.float64)
        else:
            self.pi_prior = np.asarray(pi_prior, dtype=np.float64).reshape(n_points)

        if use_ransac_init:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(X, u, self.K, self.dist, reprojectionError=3.0)
            if not ok:
                raise RuntimeError("PnPRansac failed to initialize pose")
            rvec, tvec = rvec.ravel(), tvec.ravel()
        else:
            rvec, tvec = np.zeros(3), np.zeros(3)

        for _ in range(max_iter):
            w, _, u = self.e_step(rvec, tvec, X, u, u_pre, lamda=lamda)
            rvec_new, tvec_new = self.m_step(rvec, tvec, X, u, w)
            change = np.linalg.norm(np.hstack([rvec_new-rvec, tvec_new-tvec]))
            rvec, tvec = rvec_new, tvec_new
            if change < tol:
                break

        proj_final = self.project(rvec, tvec, X)
        reproj_errs = np.linalg.norm(u - proj_final, axis=1)

        self.rvec, self.tvec = rvec, tvec
        self.w = w
        self.reproj_errors = reproj_errs

        return rvec, tvec, w, reproj_errs
