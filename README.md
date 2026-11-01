# UR5e / UR10e 실시간 캘리브레이션과 팔로워 제어

RealSense D435 RGB 영상 한 장과 양팔의 RTDE 관절값으로 카메라 기준 베이스 포즈를 추정하고, **UR5e가 UR10e를 따라가며 플랜지를 마주보도록** 제어합니다. 영상 왼쪽은 UR5e, 오른쪽은 UR10e입니다. 로봇 종류를 자동 인식하는 방식은 아닙니다.

현재 제어 모드는 **follow 하나**입니다. `--mode follow`는 생략할 수 있습니다. 영상 추론만 실행하거나, 실제 상태를 한 번 취득해 이후 제어를 가상 관절로 실행할 수도 있습니다.

양팔 추론·제어는 기본적으로 **두 베이스가 같은 바닥면에 수직으로 설치되어 있다**는 조건을 사용합니다. 두 베이스의 수직축과 바닥면 높이를 함께 맞추며, 베이스 사이 거리와 수평 회전각은 영상에서 추정합니다. 카메라의 수평면에 맞추는 기능은 아닙니다. 다른 높이·기울기로 설치한 경우 `--no_common_base_plane`으로 해제합니다.

전체 동작·좌표계·설정·제약·코드 위치는 [현재 구현 문서](docs/CURRENT_IMPLEMENTATION.md)에 정리했습니다.

## 실행할 수 있는 것

| 작업 | 진입점 | 실제 로봇 구동 |
| --- | --- | --- |
| UR5e / UR10e 단독 포즈 추정 | `run_ur5e.sh` / `run_ur10e.sh` | 없음 |
| 같은 영상에서 양팔 포즈 추정 | `run_dual.sh` | 없음 |
| 리더–팔로워 제어와 상대 작업 경로 | `run_dual_control.sh` / `run_cooperative.sh` | `--execute`와 Arm 후 팔로워만 |
| 초기 상태 취득 후 가상 제어 | `run_simulation.sh` | 없음 |
| 연결 없는 GUI 데모 | `python3 -m urcal.tools.preview_gui` | 없음 |
| 합성 제어 방식 비교 | `python3 -m urcal.tools.simulate` | 없음 |
| 실행 로그 분석 | `python3 -m urcal.tools.analyze_log` | 없음 |

## 환경 준비

Python 3.10 환경에서 다음 의존성을 설치합니다. GUI에는 시스템 Tkinter, 실제 영상에는 D435 드라이버와 USB 접근 권한이 필요합니다. GPU 사용 시 PyTorch/CUDA 환경이 설치된 드라이버와 맞아야 합니다.

```bash
cd ~/ur_live_calibration
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
sudo apt install python3-tk
```

이미 준비한 Python 환경이 있으면 그 환경을 사용해도 됩니다. 실행 스크립트는 저장소 기준 `PYTHONPATH`와 `TORCH_HOME`을 설정합니다. 별도 인터프리터를 지정할 때는 `ROBOTPOSE_PYTHON=/경로/python`을 사용합니다.

모델 파일은 Git LFS로 관리합니다. GitHub에서 새로 내려받는 경우 저장소 접근 권한이 있는 계정으로 다음과 같이 받습니다.

```bash
git lfs install
git clone https://github.com/Stick-0/ur_live_calibration.git
cd ur_live_calibration
git lfs pull
```

| 용도 | 기본 가중치 |
| --- | --- |
| UR5e | `trained_models/v3m2a3/best.pt` |
| UR10e | `trained_models/ur10e_2/best.pt` |
| 모델 초기화용 ResNet | `torch_cache/hub/checkpoints/resnet18-f37072fd.pth` |

## 영상 추론

```bash
./run_ur5e.sh --robot_ip 169.254.186.10 --show_seg
./run_ur10e.sh --robot_ip 169.254.186.6 --show_seg
```

같은 프레임에서 양팔을 처리하려면 다음을 실행합니다. 두 모델은 같은 프레임에 대해 순차적으로 실행됩니다.

```bash
./run_dual.sh \
  --ur5e_ip 169.254.186.10 \
  --ur10e_ip 169.254.186.6 \
  --show_seg
```

카메라 설정은 기본 640×480, 30 FPS이고 실제 추론 처리 속도는 장비에 따라 다릅니다. `--split_ratio 0.5`를 기준으로 좌우 영역을 나누어 추론하며 영상에 경계선은 그리지 않습니다. 영상에는 키포인트 점·좌표축·segmentation을 표시하고, 상태 문구·키포인트 이름·FPS 문구는 표시하지 않습니다. 추론 전용 실행은 FPS·PnP 방식·성공/실패·유효 키포인트 수를 약 1초마다 터미널에 요약합니다. 제어 포함 실행은 원본 검출점의 평균·최대 재투영 오차, 촬영 후 경과 시간과 측정 거절 이유도 표시합니다. `s`는 segmentation 표시, `k`는 키포인트 점 표시를 전환하며 `q` 또는 Esc로 종료합니다. 단축키 안내는 시작할 때 터미널에 한 번 출력합니다.

추론 전용에서 이전 포즈의 영향이 강하면 `--ur5e_empnp_lambda 0.01 --ur10e_empnp_lambda 0.01`처럼 값을 낮출 수 있습니다. `run_dual.sh`에는 제어용 `--filter_alpha` 옵션이 없습니다.

## 실제 팔로워 제어

먼저 구동 없는 미리보기로 영상과 목표를 확인할 수 있습니다.

```bash
./run_dual_control.sh \
  --ur5e_ip 169.254.186.10 \
  --ur10e_ip 169.254.186.6 \
  --gap 0.30 --show_seg --initial_align
```

실제 구동은 다음과 같습니다. 아래의 0 TCP 오프셋은 **양팔의 활성 TCP가 실제 플랜지와 일치하는 설치**에 해당합니다. 공구나 별도 TCP가 있으면 실제 설치값 `x y z rx ry rz`로 바꿉니다. 제어 대상 로봇은 Remote Control과 RTDE 연결이 가능해야 합니다.

```bash
./run_dual_control.sh --mode follow \
  --ur5e_ip 169.254.186.10 \
  --ur10e_ip 169.254.186.6 \
  --ur5e_tcp_offset 0 0 0 0 0 0 \
  --ur10e_tcp_offset 0 0 0 0 0 0 \
  --gap 0.30 --show_seg --initial_align --execute
```

1. GUI에서 현재 양팔 자세와 팔로워 목표를 확인합니다. 간격·경로를 수정했다면 **Apply target**을 누릅니다.
2. READY에서 **Arm motion**을 누릅니다. 연결·IK/FK·경로 검사 후 초기 정렬을 시작합니다.
3. 정렬과 새 캘리브레이션 확보 동안 리더·카메라·베이스를 정지 상태로 유지합니다.
4. **LIVE CALIBRATION READY / ACTIVE**가 되면 리더를 움직여 추종합니다.

프로그램이 리더 자체를 구동하지는 않습니다. 기본 리더는 UR10e이며 `--leader ur5e`로 역할을 바꿀 수 있습니다. 기본 간격은 20 cm이므로, 위 예시처럼 `--gap 0.30`을 지정하면 30 cm를 사용합니다.

실행하면 **Follower TCP position error** 그래프 창도 자동으로 열립니다. 팔로워 베이스 좌표에서 `목표 TCP 위치 − 현재 TCP 위치`를 계산하고, 전체 거리 오차 `sqrt(ex² + ey² + ez²)` 하나를 **mm**, 최근 **30초**의 선으로 표시합니다. 그래프는 흰 배경에 진한 파란 선을 사용하고 세로축은 0부터 시작합니다. 초기 정렬은 선택한 정렬 끝점, 추종은 허용 보정을 포함해 제어기가 채택한 목표를 사용합니다. 현재 위치는 RTDE 값이며, 목표는 캘리브레이션 기반 추정값입니다. 그래프만 닫아도 제어는 계속되며 **Position error graph** 버튼으로 다시 열 수 있습니다. 미리보기와 시뮬레이션에서도 표시됩니다.

### 좌표변환 목표만으로 추종

기본 `--follow_law feedforward`는 리더 운동 보상과 자세 오차 보정 속도를 합칩니다. `--follow_law pose`는 매 제어 주기에 최신 베이스 변환과 리더 플랜지 자세로 **팔로워 TCP 목표를 다시 계산하고, 현재 TCP와의 위치·방향 오차로 하나의 속도 명령을 만듭니다.** 리더·베이스·상대 경로의 속도를 별도로 더하지 않습니다. 오실레이션 억제에는 팔로워의 실제 측정 속도를 사용합니다.

최근 좌표를 바로 반영하는 실행 예시는 다음과 같습니다.

```bash
./run_dual_control.sh --mode follow \
  --ur5e_ip 169.254.186.10 \
  --ur10e_ip 169.254.186.6 \
  --ur5e_tcp_offset 0 0 0 0 0 0 \
  --ur10e_tcp_offset 0 0 0 0 0 0 \
  --gap 0.30 --show_seg \
  --filter_alpha 1.0 --base_pose_window 0 \
  --initial_align --execute --approach_speed 0.01 \
  --follow_law pose
```

이 모드는 `--method separated --base_motion_mode pose --leader_response bounded`에서 사용합니다. 세 옵션은 기본값이며 다른 조합은 실행 전에 거절합니다. 초기 정렬·캘리브레이션 검사·복구·15 cm 접근의 절차는 같습니다. 접근 버튼은 상대 목표를 진행시키고, 그 목표 오차로 추종합니다.

위치·방향 이득, 데드밴드, 보정 속도 상한, 제동 거리 제한과 최종 TCP·관절 속도·가속도·공간 제한을 그대로 사용합니다. 기본 보정 상한은 **100 mm/s / 0.15 rad/s**이고 최종 명령 제한이 추가로 적용됩니다. 따라서 한 속도 항으로 바꾸는 것만으로 더 빨라지는 것은 아니며, 움직이는 목표에 대한 추종 지연이 생길 수 있습니다. 기존 방식으로 돌아가려면 `--follow_law feedforward`를 사용하거나 이 옵션을 생략합니다.

`--path hold`는 **리더에 대한 상대 자세를 유지**한다는 뜻입니다. 리더가 움직이면 목표도 함께 움직입니다. 상대 작업을 추가하려면 다음 옵션을 사용합니다.

| 작업 | 추가 옵션 |
| --- | --- |
| 마주보기 상대 자세 유지 | `--path hold` (기본) |
| 리더 Z 방향으로 5 cm 접근 | `--path line --path_delta 0 0 -0.05 0 0 0 --path_duration 10` |
| 반지름 2 cm 원 경로 | `--path circle --path_radius 0.02 --path_duration 10` |

GUI의 **Pause task**는 상대 경로 진행만 멈춥니다. 리더 추종까지 정지하려면 **STOP / Exit**, Space, `q`, Esc 또는 Ctrl+C를 사용합니다.

### 추종 중 15 cm까지 접근

기본 `separated + pose + bounded`, `--path hold`의 ACTIVE 상태에서 첫 추종 명령이 완료되면 **Approach to 15 cm** 버튼을 사용할 수 있습니다. 초기 정렬을 사용한다면 정렬과 캘리브레이션 전환을 먼저 마쳐야 합니다.

1. **Approach to 15 cm**: 리더를 따라가면서 플랜지 간격 목표를 15 cm까지 서서히 줄입니다.
2. **Pause approach**: 접근 목표의 진행과 목표 간격을 고정하고 리더 추종은 계속합니다.
3. **Resume approach**: 멈춘 지점부터 접근을 이어갑니다.

`--approach_speed 0.005`가 기본이며, 리더 기준 접근 목표의 최고 속도 **5 mm/s**를 뜻합니다. 리더 추종이 합쳐진 팔로워 전체 TCP 속도는 별도입니다. 현재 명령의 상대 X/Y와 방향은 유지하고 리더 플랜지 Z축 방향의 간격만 줄입니다.

일시정지는 팔을 순간 정지시키는 동작이 아닙니다. 고정된 간격 목표를 향한 잔여 오차 보정과 리더 추종은 가감속 한계 안에서 계속됩니다.

GUI는 명령 간격·추정 간격·진행률을 따로 표시합니다. **15 cm target set**은 목표 설정 완료이며 실제 접촉이나 삽입 완료 판정이 아닙니다. 15 cm는 플랜지 기준 간격으로 공구 끝의 삽입 깊이와 다릅니다. 핀·홀 공구를 사용한다면 실제 TCP 오프셋을 지정해야 하며, 현재 힘 감지·삽입 판정·자동 후퇴 기능은 없습니다.

일시적인 영상 오류나, 제동 없이 새 좌표로 전환하는 동안에는 접근 진행만 잠깐 보류하고 정상 좌표가 준비되면 자동으로 이어갑니다. GUI와 터미널에 자동 진행 대기를 표시합니다. 영상 관측을 위해 실제 제동에 들어가거나 복구 정지가 발생하면 **Resume approach**를 눌러야 다시 접근합니다. 직접 누른 **Pause approach**도 자동 해제되지 않습니다. 복구 정지 후 재Arm해도 접근은 일시정지 상태를 유지하며, **Apply target**은 접근 상태를 초기화합니다.

## 추종 반응과 제약

기본 `--base_motion_mode pose`에서는 영상으로 베이스 좌표 관계를 갱신하고, 실제 운동 속도는 관절·RTDE 측정값으로 계산합니다. 영상에서 추정한 베이스 위치의 차분을 필수 보상 속도로 사용하지 않습니다. 속도 추정·예측을 포함한 연구 비교는 `--base_motion_mode velocity`를 명시합니다.

UR10e PnP 출력의 **Z축 180° / 로컬 Z +5 mm 변환은 기본 ON**입니다. 기존 실행 명령에 추가 옵션 없이 적용되며, 시작 로그의 `UR10e PnP output correction: ON`으로 확인할 수 있습니다. `--no_ur10e_output_correction`을 지정하면 이 출력 변환을 끕니다. 이 설정은 명목 FK·Jacobian의 좌표 정의와 별개이며, 보정 ON/OFF에 따라 GUI 축과 제어 목표 방향도 달라집니다.

ACTIVE 추종에서는 최근 5개 영상의 **상대 베이스 좌표**를 모아 대표값을 선택하고, 5 mm / 0.5° 안의 작은 변화는 따라가지 않습니다. 그 밖의 변화는 0.35초 시정수와 20 mm/s·0.05 rad/s의 좌표 갱신 상한으로 부드럽게 반영합니다. 이 상한은 베이스 추정 좌표에만 적용하며, 기본 `feedforward`의 리더 관절값으로 계산한 추종 속도는 그대로 사용합니다. 실제 베이스를 빠르게 옮기면 좌표 반영은 지연될 수 있습니다. 설정은 `--base_pose_*` 옵션이며, 필터는 실제 베이스가 고정됐다고 판정하거나 캘리브레이션 정확도를 보장하지 않습니다.

| 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `--follow_max_speed` | 1.0 m/s | 팔로워 TCP 선속도 상한 |
| `--feedback_speed` | 0.10 m/s | 위치 오차를 따라잡는 보정 속도 상한 |
| `--feedback_angular_speed` | 0.15 rad/s | 방향 오차 보정 상한 |
| `--kp_linear`, `--kp_angular` | 3, 3 | 오차 보정의 비례 계수 |
| `--follow_damping_time` | 0.12 s | 실제 상대 속도를 반영한 목표 접근 감쇠; 0은 기존 보정 |
| `--max_joint_speed` | 0.2 rad/s | 관절별 속도 상한 |
| `--max_joint_accel` | 0.75 rad/s² | 관절별 가속도 상한 |
| `--max_travel` | 0.30 m | 최초 시작 위치 기준 이동 반경 |

목표 부근에서는 측정 상대 속도로 앞으로의 접근을 미리 반영해 오차 보정을 줄입니다. 빠르게 따라가다 목표를 넘는 현상을 줄이기 위한 기능입니다. 실제 명령에는 TCP·관절 속도, 가속도와 기하 검사가 함께 적용되므로 선속도 상한이 곧 실제 운전 속도는 아닙니다.

다음 추종 명령을 만들기 어려운 순간에는 검증된 감속 후보를 사용하고 다음 제어 주기에 다시 추종을 시도합니다. 이 과정은 새 Arm을 요구하지 않습니다. 명령 후보의 반복 기하 계산은 같은 검사 결과를 유지하도록 최적화했습니다. 감속도 실행할 수 없거나 실제 상태·통신에 문제가 있으면 정지 절차를 따릅니다.

기본 추종에서 단발성 영상 추정이 튀면 해당 프레임을 건너뜁니다. **마지막 정상 영상의 촬영 나이가 0.5초 이내이고, 거절이 0.3초 이내인 동안** 기존 좌표로 추종을 이어갑니다. 다음 정상 프레임은 안정화 대기 없이 받아들이며, 지속되는 이상이나 촬영 나이 만료에는 감속·관측으로 전환합니다. 최초 이상 감지부터 총 2초의 복구 기한은 연장하지 않습니다. 최소 3개의 신선한 프레임이 0.15초 이상 일관되면, 기존 기준의 15 cm / 15° 안에서 새 좌표로 부드럽게 전환하며 새 Arm 없이 추종합니다. 짧은 유예 중 새 좌표를 확보하면 제동을 끼워 넣지 않습니다.

이 동작은 기본 `separated + pose + bounded` 실제 제어에서 적용합니다. `--calibration_recovery_timeout 2`는 이상 감지 후 관측 기한, `--calibration_blend_time 0.5`는 좌표 전환 최소 시간입니다. 기한 안에 유효한 측정을 확보하지 못하거나 실제 상태·통신·형상 검사에서 실패하면 정지하며, 다시 Arm해야 합니다. `--no_live_calibration_recovery`는 이 관측 기능을 끕니다.

모든 수치의 의미와 대기·정지 원인은 [현재 구현 문서의 기본 설정과 상태 해석](docs/CURRENT_IMPLEMENTATION.md#7-주요-기본-설정)을 참고하세요.

## 실제 상태에서 시작하는 시뮬레이션

```bash
./run_simulation.sh \
  --ur5e_ip 169.254.186.10 \
  --ur10e_ip 169.254.186.6 \
  --gap 0.30 --show_seg --sim_motion leader
```

관절각·TCP·캘리브레이션을 한 번 취득한 뒤 카메라와 RTDE 수신 연결을 닫습니다. 이후 GUI에서 **Start simulation**, **Pause simulation**, **Reset simulation**을 사용합니다. 실제 로봇 명령은 보내지 않습니다. 초기 차이가 크면 가상 초기 정렬을 수행합니다.

가상 추종 중에도 같은 **Approach to 15 cm / Pause approach / Resume approach** 버튼을 사용합니다. **Pause approach**는 접근 목표를 고정하고, **Pause simulation**은 가상 시간과 전체 운동을 멈춥니다.

저장된 상태는 카메라·로봇 연결 없이 다시 사용할 수 있습니다. 터미널에 출력된 실제 파일 경로를 지정합니다.

```bash
./run_simulation.sh \
  --initial_state "runs/simulation/실행ID/initial_state.json" \
  --gap 0.30 --sim_motion leader
```

`--sim_motion`은 `none`, `leader`, `bases`, `both`를 지원합니다. 시뮬레이션에는 `--execute`나 `--initial_align`을 추가하지 않습니다. 관절속도를 적분하는 기구학 시뮬레이션이며 실제 관성·접촉·서보 지연·PnP 노이즈는 재현하지 않습니다.

## 로그와 코드 위치

실제 추종은 `runs/<실행ID>.jsonl`, 시뮬레이션은 `runs/simulation/<실행ID>/`에 설정·상태·명령·정지 원인을 기록합니다. GUI의 오차는 추정 좌표에서 계산한 값이며 독립 계측으로 검증한 물리적 오차와는 구분합니다.

```bash
python3 -m urcal.tools.analyze_log runs/실험.jsonl
python3 -m urcal.tools.preview_gui
python3 -m urcal.tools.simulate --output_dir runs/synthetic-comparison --scenario bases --duration 20 --seed 7
```

| 폴더·파일 | 역할 |
| --- | --- |
| `urcal/inference/` | 카메라·신경망·투표·EMPnP 포즈 추정 |
| `urcal/control/` | CLI, RTDE, 캘리브레이션 검사, 초기 정렬, GUI |
| `urcal/cooperative/` | 상대 목표·추종 명령·운동 및 기하 검사·실행 로그 |
| `urcal/simulation/` | 초기 상태 취득·저장과 가상 제어 |
| `urcal/tools/` | 화면 데모·합성 비교·로그 분석 |
| `configs/workspace.example.json` | 작업공간·캡슐 설정 예시 |
| `tests/` | 단위·회귀·모의 장비 검증 |
| `reference/` | 추론 최적화 이전 코드 비교 자료 |
| `docs/Research_Implementation_Plan.txt` | 연구 계획 원문 |
| `MANIFEST.json`, `SHA256SUMS` | 파일 메타데이터·무결성 목록 |

오프라인 검사는 다음과 같이 실행합니다. 실제 로봇에서의 성능 측정을 대신하지는 않습니다.

```bash
PYTHONPATH=tests:. python3 -B -m unittest discover -s tests -p 'test_*.py'
```
