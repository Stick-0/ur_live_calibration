# UR5e / UR10e 실시간 캘리브레이션과 양팔 상대 제어

RealSense D435 RGB 영상과 UR RTDE 관절값을 이용해 카메라 기준 로봇 베이스
포즈를 매 프레임 추정하고, 이를 이용해 플랜지 자세를 유지하거나 상대 경로를
추종하는 실행 묶음입니다. 모델별 가중치와 실행 스크립트가 포함되어 있으며,
기본 추론은 중복 연산을 줄인 최적화 버전입니다.
최적화 전 Python 코드 6개는 `reference/`에 보관했습니다.

처리 흐름은 RGB → 신경망/투표 기반 2D 키포인트 검출 → 관절값으로 3D
키포인트 계산 → EMPnP 포즈 추정입니다. 카메라 내부 파라미터와 왜곡 계수는
RealSense에서 읽습니다. 기본 입력 크기는 640×480, 카메라 설정은 30 FPS입니다.
실제 추론 처리 FPS는 하드웨어에 따라 달라집니다.

| 하고 싶은 작업 | 실행 | 움직이는 팔 |
| --- | --- | --- |
| UR5e 또는 UR10e의 카메라 기준 포즈 확인 | `run_ur5e.sh` / `run_ur10e.sh` | 없음 |
| 같은 영상에서 왼쪽 UR5e·오른쪽 UR10e 포즈 확인 | `run_dual.sh` | 없음 |
| 카메라 기준 한 위치에서 두 플랜지를 마주보게 유지 | `run_dual_control.sh --mode hold` | 양팔 |
| 리더에 대한 상대 자세 유지·직선·원 경로 추종 | `run_cooperative.sh` 또는 `run_dual_control.sh --mode follow` | 팔로워만 |
| 장비 없이 세 제어 방식 비교 | `python3 -m urcal.tools.simulate` | 없음 |

제어 프로그램도 기본값은 미리보기입니다. 실제 구동에는 `--execute`, 실제 TCP offset,
GUI/영상 창에서의 명시적인 Arm이 모두 필요합니다. 기본 역할은 **UR10e 리더 / UR5e 팔로워**,
플랜지 원점 간 초기 간격은 **30 cm**입니다.

## 포함 파일

```text
ur_live_calibration/
├── README.md
├── requirements.txt
├── run_ur5e.sh
├── run_ur10e.sh
├── run_dual.sh                   # 같은 프레임에서 두 로봇 추론
├── run_dual_control.sh           # 고정 자세 / 리더–팔로워 제어
├── run_cooperative.sh            # 이동 정반 협업 제어 실행
├── urcal/                        # 실행 코드 패키지
│   ├── paths.py                  # 작업 디렉터리와 무관한 묶음 루트 BUNDLE
│   ├── inference/                # 기존 포즈 센서
│   │   ├── infer_live.py          # UR5e, 카메라·RTDE 공통 유틸리티
│   │   ├── infer_live_ur10e.py    # UR10e, 선택적 ROS 2 출력
│   │   ├── infer_live_dual.py     # 같은 영상의 좌우 분할 추론
│   │   ├── infer_live_10.py       # UR10e 진입점 별칭
│   │   ├── network3.py           # ResNet·Transformer 키포인트 신경망
│   │   ├── vote2.py              # 2D 키포인트와 공분산 투표
│   │   └── my_pnp.py             # EMPnP
│   ├── control/                  # 공통 제어 기반과 카메라 고정 hold
│   │   ├── runtime.py            # 상태 수신, RTDE 세션, 실행 옵션
│   │   ├── geometry.py           # 좌표변환·플랜지·TCP·마주보기 목표
│   │   ├── policy.py             # 측정 유효성·안정화·hold 속도 제한
│   │   ├── vision.py             # 별도 프로세스의 영상/관절 샘플링
│   │   └── gui.py                # 목표 편집·Arm·Pause·Stop GUI
│   ├── cooperative/              # 이동 정반 상대 작업 제어
│   │   ├── runtime.py            # 제어 루프·명령 성공 후 진행률 확정
│   │   ├── controller.py         # 상태·목표·보상·피드백·검사 통합
│   │   ├── state.py              # 시간 정합·SE(3) 필터·단기 예측
│   │   ├── task.py               # 상대 경로와 작업 속도 비율 선택
│   │   ├── kinematics.py         # nominal FK·Jacobian·IK·캡슐 검사
│   │   └── logging.py            # 비동기 JSONL 기록
│   └── tools/
│       ├── analyze_log.py         # 추정/독립 기준 오차·완료시간 분석
│       └── simulate.py            # 같은 합성 조건으로 세 방식 비교
├── docs/Research_Implementation_Plan.txt # 연구 진행 계획 원문
├── configs/workspace.example.json # 작업공간·캡슐 설정 예시
├── reference/                     # 최적화 전 코드 6개
├── tests/                         # 수학·상태·모의 장비·원본 동등성 검사
├── trained_models/
│   ├── v3m2a3/best.pt              # UR5e
│   └── ur10e_2/best.pt             # UR10e
├── torch_cache/hub/checkpoints/
│   └── resnet18-f37072fd.pth       # 모델 초기화용 ResNet 가중치
├── runs/                         # 실행 시 생성되는 실험 로그, Git 제외
├── MANIFEST.json                  # 파일 출처, 가중치 메타데이터
└── SHA256SUMS                     # 파일 무결성 확인
```

루트에는 실행 스크립트와 안내·의존성·무결성 파일을 두고, Python 구현은 `urcal/`로
모았습니다. 기존 `run_*.sh` 명령은 그대로 사용합니다. Python 파일을 직접 실행하던 경우에는
저장소 루트에서 `python3 -m urcal.inference.infer_live_dual`처럼 모듈 경로로 실행하세요.
가중치와 캐시는 이동하지 않았고, `reference/`는 원본 비교용으로 분리해 유지합니다.

## 모듈을 따라 이해하는 처리 흐름

1. [추론 진입점](urcal/inference/infer_live_dual.py)이 영상과 관절값을 받습니다.
   [network3.py](urcal/inference/network3.py)의 ResNet·Transformer가 7개 키포인트별
   segmentation과 2D 방향 벡터장을 출력합니다.
   [vote2.py](urcal/inference/vote2.py)는 픽셀들의 방향으로 키포인트 위치·공분산·유효 여부를 계산합니다.
2. 모델별 FK가 관절값으로 로봇의 3D 키포인트를 계산하고,
   [my_pnp.py](urcal/inference/my_pnp.py)가 2D–3D 대응과 내부 파라미터로
   카메라–학습 모델 베이스 변환을 추정합니다. 단일 UR5e와 UR10e 실행은 각각
   [infer_live.py](urcal/inference/infer_live.py), [infer_live_ur10e.py](urcal/inference/infer_live_ur10e.py)이며,
   [infer_live_10.py](urcal/inference/infer_live_10.py)는 UR10e 별칭입니다.
3. 제어 실행에서는 [vision.py](urcal/control/vision.py)가 이 포즈 센서를 별도 프로세스에서 실행합니다.
   촬영 시각과 관절 샘플 시각을 보존하고, 원래 검출점의 재투영 오차를 다시 계산해 양팔 측정을 함께 전달합니다.
   [policy.py](urcal/control/policy.py)는 누락·시간 역전·노후화·과도한 점프·낮은 품질을 거부합니다.
4. 카메라 고정 모드는 [control/runtime.py](urcal/control/runtime.py)의 고정 목표 제어를 사용합니다.
   상대 작업은 [state.py](urcal/cooperative/state.py)에서 상태를 시간 정합하고,
   [controller.py](urcal/cooperative/controller.py)와 [task.py](urcal/cooperative/task.py)에서
   경로·보상·오차 피드백·허용 작업 속도를 계산합니다.
5. [kinematics.py](urcal/cooperative/kinematics.py)가 상대 작업의 관절·도달성·공간 조건을 검사합니다.
   공통 RTDE 세션은 제어기 상태와 목표를 확인한 뒤 `speedL`로 TCP 속도를 보냅니다.
   연구용 상대 작업에서는 현재 목표와 선택한 다음 목표를 모두 확인합니다.
   [cooperative/runtime.py](urcal/cooperative/runtime.py)는 전송 성공 후에만 경로 진행률과 명령 이력을 확정합니다.
6. [gui.py](urcal/control/gui.py)는 별도 프로세스에서 목표와 시작·정지 입력을 다룹니다.
   [logging.py](urcal/cooperative/logging.py)가 실험을 기록하고,
   [analyze_log.py](urcal/tools/analyze_log.py)가 결과를 분석합니다.
   [simulate.py](urcal/tools/simulate.py)는 같은 상위 제어기를 합성 상태와 연결해 장비 없이 검사합니다.

좌표 계산은 [geometry.py](urcal/control/geometry.py), 데이터 위치는
[paths.py](urcal/paths.py)가 공통으로 관리합니다. 따라서 영상, GUI, 제어 프로세스가
같은 가중치·좌표 정의·실행 루트를 사용합니다.

## 포함 가중치

| 로봇 | 원본 가중치 | 학습 epoch | 확인한 키포인트 |
| --- | --- | --- | --- |
| UR5e | `trained_models/v3m2a3/best.pt` | 85 | `ur5e_link0` ~ `ur5e_link6` |
| UR10e | `trained_models/ur10e_2/best.pt` | 73 | `ur10e_link0` ~ `ur10e_link6` |

두 체크포인트 모두 640×480 입력과 ImageNet 정규화로 학습되었습니다.
기본 옵션을 사용하고 `--no_normalize`는 지정하지 마세요.
UR5e는 저장소의 여러 가중치 중 로봇 종류를 메타데이터로 확인한 `v3m2a3`를
선택했습니다. 다른 UR5e 가중치를 쓰려면 실행 시 `--checkpoint`로 바꿀 수 있습니다.
기구학이 모델별로 다르므로 실행 코드와 가중치는 같은 로봇용으로 사용하세요.

## 환경 준비

GitHub에서 내려받는 경우 가중치 파일을 받기 위해 Git LFS가 필요합니다.
Ubuntu에서는 `sudo apt install git-lfs`로 설치할 수 있습니다.

```bash
git lfs install
git clone https://github.com/Stick-0/ur_live_calibration.git
cd ur_live_calibration
git lfs pull
sha256sum -c SHA256SUMS
```

저장소는 비공개이므로 접근 권한이 있는 GitHub 계정 인증이 필요합니다.
`.pt`와 `.pth` 파일은 Git LFS로 관리합니다. 일반 ZIP 다운로드 대신 위의
clone/pull 절차를 사용하면 실제 가중치까지 받을 수 있습니다.

이미 의존성이 설치된 Python 3.10 환경이 있으면 해당 환경을 활성화해 사용하세요.
새 환경에서는 이 폴더 전체를 복사하고 의존성을 설치합니다.
Python 환경 자체는 이 묶음에 포함하지 않았습니다.

```bash
cd /복사한/경로/ur_live_calibration
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU를 사용하려면 해당 PC의 드라이버에 맞는 PyTorch/CUDA 환경이 필요합니다.
코드는 CUDA가 사용 가능하면 GPU, 그렇지 않으면 CPU를 선택합니다.
OpenCV 화면을 띄울 수 있는 데스크톱 세션, D435 USB 연결 및 카메라 접근 권한,
UR 컨트롤러와 RTDE 통신이 가능한 네트워크가 필요합니다.

## 실행

아래 IP는 현재 배치의 예시이며 실제 로봇 컨트롤러 IP에 맞추세요. 사용하는 로봇에 맞춰
아래 두 명령 중 하나를 실행합니다.

```bash
# UR5e
./run_ur5e.sh --robot_ip 169.254.186.10 --show_seg

# UR10e
./run_ur10e.sh --robot_ip 169.254.186.6 --show_seg
```

실행 스크립트는 자체 위치를 기준으로 코드, 체크포인트 및 ResNet 캐시를 찾으므로
다른 작업 디렉터리에서도 실행할 수 있습니다. 다른 Python 실행 파일을 지정하려면
`ROBOTPOSE_PYTHON=/경로/python ./run_ur5e.sh ...` 형식을 사용하세요.
인자 없이 실행하거나 `--help`를 주면 장비 연결 없이 도움말을 표시합니다.

segmentation 디버그 창이 필요 없으면 `--show_seg`를 생략하세요. 최적화 버전은
이 창이 꺼져 있을 때 전체 segmentation 확률 맵의 CPU 복사를 생략합니다.
실행 중 `s`로 다시 켤 수 있습니다.

화면에는 키포인트, 추정 좌표축, `EMPnP OK/FAIL`, 처리 FPS가 표시됩니다.
`s`는 segmentation 창 전환, `q` 또는 `Esc`는 종료입니다.
`EMPnP OK`는 계산 성공 표시이며 실제 캘리브레이션 정확도를 보증하지 않습니다.

로봇 연결 없이 정지한 로봇의 관절값을 직접 줄 수도 있습니다. D435는 필요합니다.
아래 숫자는 예시이므로 영상에 보이는 로봇의 실제 관절값(라디안)으로 바꾸세요.

```bash
./run_ur5e.sh --joints 0 -1.57 1.57 -1.57 -1.57 0 --show_seg
```

순서는 shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3입니다.
UR10e도 같은 `--joints` 옵션을 지원합니다.

## 같은 카메라에서 UR5e와 UR10e 함께 추론

`run_dual.sh`는 D435에서 프레임을 한 번 읽고, 왼쪽 영역에 UR5e 모델,
오른쪽 영역에 UR10e 모델을 순서대로 실행합니다. 두 로봇은 화면의 각 영역 안에
들어오도록 배치하세요. 로봇 종류를 자동으로 판별하거나 경계 너머로 추적하지 않습니다.

```bash
# IP는 각 로봇 컨트롤러 주소로 변경
./run_dual.sh --ur5e_ip 169.254.186.10 --ur10e_ip 169.254.186.6 --show_seg

# 추론 영역의 경계를 왼쪽에서 영상 너비의 45% 위치로 변경 (기본값 0.5)
./run_dual.sh --ur5e_ip 169.254.186.10 --ur10e_ip 169.254.186.6 --split_ratio 0.45

# 정지한 로봇은 실제 관절값 6개를 라디안으로 직접 지정 가능
./run_dual.sh \
  --ur5e_joints 0 -1.57 1.57 -1.57 -1.57 0 \
  --ur10e_joints 0 -1.57 1.57 -1.57 -1.57 0
```

각 로봇에 IP 또는 관절값이 필요하며, 한쪽만 RTDE로 연결하는 방식도 가능합니다.
두 관절값은 카메라 프레임을 받은 직후, 신경망 실행 전에 읽습니다. 카메라 노출과
RTDE 사이의 하드웨어 시간 동기화는 수행하지 않습니다.

영역별 영상은 종횡비를 유지하면서 여백을 추가해 640×480으로 입력합니다.
투표에서는 여백을 제외하고, 특징점과 공분산을 원본 영상 좌표로 복원합니다.
두 EMPnP 계산에 원본 카메라 내부 파라미터와 왜곡 계수를 공통으로 사용하므로
각 결과는 같은 카메라 좌표계에서 표현한 로봇별 학습 모델 베이스 포즈 `T_C_M`입니다.
제어에 쓸 때는 아래의 모델–제어기 베이스 변환을 추가합니다.
각 로봇의 FK와 이전 포즈는 별도로 관리하며, 투표 실패 특징점은 PnP에서 제외합니다.
유효 특징점이 4개 미만인 쪽은 `FAIL`로 표시하고 다른 쪽은 계속 처리합니다.
UR10e의 기존 학습 좌표 정의와 5 mm Z 오프셋도 유지합니다.

화면에 두 로봇의 특징점·좌표축·상태와 전체 처리 FPS를 표시합니다.
가운데 구분선은 표시하지 않습니다. `k`로 두 로봇의 키포인트 점과 이름 표시를
켜고 끌 수 있으며, `--hide_keypoints`를 추가하면 숨긴 상태로 시작합니다.
`s`로 두 영역의 segmentation 창을 전환하고 `q`/`Esc`로 종료합니다.
이 실행 파일은 화면 표시용이며 ROS 발행이나 포즈 파일 저장은 하지 않습니다.
두 모델을 메모리에 함께 올리고 프레임마다 순차 실행하므로, 단일 모델 실행보다
메모리와 처리 시간이 더 필요합니다. `--device cpu` 또는 `--device cuda`로
실행 장치를 선택할 수 있으며 기본값은 자동 선택입니다.

가중치를 바꾸려면 `--ur5e_checkpoint`, `--ur10e_checkpoint`를 사용하세요.
Git LFS 포인터만 있는 경우에는 장비 연결 전에 `git lfs pull` 안내와 함께 종료합니다.
분할 입력과 실제 두 로봇 영상에 대한 정확도·속도는 실장비에서 확인해야 합니다.
장비/가중치 없이 좌표 복원, 투표, 독립 포즈 처리 등을 검사할 수 있습니다.

```bash
python -B -m unittest discover -s tests -p test_dual_inference.py -v
```

## 실시간 캘리브레이션을 반영한 양팔 플랜지 제어

`run_dual_control.sh`는 위의 왼쪽 UR5e / 오른쪽 UR10e 추론을 내부에서 실행합니다.
기존 `run_dual.sh`는 종료한 뒤 사용하세요. 두 프로그램이 카메라를 동시에 열면 안 됩니다.
이동 중에도 각 팔이 지정된 좌우 영역에 보여야 합니다. 고정 모드는 추정 실패 시 멈추고,
연구용 추종 모드는 아래에 설명한 짧은 측정 공백만 보완한 뒤 계속 유효하지 않으면 멈춥니다.
기본 실행은 **수신·목표 미리보기만** 수행하며 로봇 제어 스크립트를 업로드하지 않습니다.
`--mode follow`는 아래 연구 계획 기반 협업 모듈을 사용합니다. `--mode hold`는 기존 카메라 고정 목표 모드를 유지합니다.

```bash
# 고정 모드: 카메라 좌표계의 고정된 위치·방향으로 두 플랜지를 유지
./run_dual_control.sh --mode hold \
  --ur5e_ip 169.254.186.10 --ur10e_ip 169.254.186.6 --show_seg

# 추종 모드: UR10e 리더 / UR5e 팔로워 (기본값)
./run_dual_control.sh --mode follow \
  --ur5e_ip 169.254.186.10 --ur10e_ip 169.254.186.6 --show_seg
```

두 모드의 기본 마주보기는 플랜지의 **+Z 축이 상대 플랜지 원점을 향하고 +X 축을 공유**하도록
목표를 만듭니다. 팔로워 모드의 초기 목표는 리더 플랜지의 +Z 방향으로 `gap`만큼 떨어진
위치에 팔로워를 배치하고, 자세는 리더 기준 X축으로 180도 회전시킵니다.
`--leader ur5e`로 역할을 바꿀 수 있습니다. 리더에는 제어 인터페이스를 만들지
않으므로 기존 프로그램 또는 작업자가 별도로 움직입니다. 자동 freedrive 전환은 없습니다.

기본 플랜지 원점 간격은 **30 cm**입니다. `--gap 0.30` 또는 GUI에서 바꿀 수 있습니다.
고정 모드의 목표는 별도 GUI에서 입력합니다.

1. 최초 유효 상태의 두 플랜지 중점·UR5e→UR10e 방향이 입력칸에 초기값으로 표시됩니다.
2. `X / Y / Z`는 **카메라 좌표계, m** 단위입니다. 카메라 X는 오른쪽, Y는 아래, Z는 앞쪽입니다.
3. `Roll / Pitch / Yaw`는 **도(deg)** 단위이며 `Rz(yaw) Ry(pitch) Rx(roll)` 순서의 방향입니다.
   이 방향의 +Z를 따라 UR5e는 중점에서 −간격/2, UR10e는 +간격/2에 놓입니다.
4. `Apply target`을 누르면 목표가 갱신되고 영상의 `TARGET` 좌표축에서 확인할 수 있습니다.
   입력 중에는 목표가 바뀌지 않습니다. 고정 모드는 Apply 전에는 시작할 수 없습니다.
5. 구동 중에는 입력칸을 잠급니다. 목표를 바꾸려면 정지·종료 후 다시 미리보기로 실행하세요.

팔로워 모드의 GUI에서는 간격과 상대 경로를 설정합니다. 기준 위치·방향은 실시간 UR10e 플랜지에서 정합니다.
고정 모드는 적용한 목표를 카메라 좌표에 고정하고, 캘리브레이션 갱신에 따라 각 로봇 base의 명령을 다시 계산합니다.
CLI의 `--center_pose X Y Z RX RY RZ`로 고정 목표를 미리 지정할 수도 있습니다.
이 CLI 옵션은 XYZ가 m, RX/RY/RZ가 라디안 **회전 벡터**입니다. GUI의 Euler 각과 구분하세요.
카메라를 고정해 사용해야 이 모드가 물리 공간의 한 자세를 유지합니다.
GUI에는 Tkinter가 필요합니다. Ubuntu에서 없다면 `sudo apt install python3-tk`로 설치하세요.

영상 창에는 `TARGET` 좌표축과 `WAIT`/`READY`/`ACTIVE` 상태를 표시합니다.
`k`는 키포인트, `s`는 segmentation 표시 전환입니다. **스페이스 또는 q/Esc는
정지 후 종료**, GUI의 `Stop` 또는 창 닫기, 터미널의 Ctrl+C도 정지 후 종료합니다.
종료나 오류 후 자동 재시작하지 않습니다.

실제 제어는 `--execute`와 **양쪽 로봇에 설정된 실제 TCP offset**을 명시한 뒤,
목표를 확인하고 GUI의 `Arm` 또는 영상 창의 `a`로 시작합니다. `--execute` 없이 `a`를 누르면
미리보기 상태를 유지합니다. 아래 예시는 양쪽 활성 TCP offset이 모두 0인 경우에만 맞습니다.
공구를 사용한다면 로봇 Installation의 값을 **m / 회전 벡터 rad**로 바꾸세요.

```bash
./run_dual_control.sh --mode follow \
  --ur5e_ip 169.254.186.10 --ur10e_ip 169.254.186.6 \
  --ur5e_tcp_offset 0 0 0 0 0 0 \
  --ur10e_tcp_offset 0 0 0 0 0 0 \
  --execute
# READY에서 목표를 검토한 뒤 Arm 또는 영상 창에서 a.
# 고정 모드는 --mode hold로 바꾸고 GUI에서 Apply target 후 시작합니다.
```

구동할 팔은 UR의 Remote Control 및 모션이 가능한 상태여야 합니다. `a`를 누르면
해당 팔에 ur_rtde 제어 스크립트가 올라가므로 그 팔의 다른 모션 프로그램과 함께
실행하지 마세요. 제어하는 팔의 TCP offset은 실제 설정과 매 주기 비교합니다.
리더의 offset은 수신 API에서 조회할 수 없어 명시한 값을 사용하고, 두 팔 모두
관절 기반 nominal FK와 실제 TCP에서 복원한 플랜지를 비교해 큰 불일치를 거부합니다.

추론과 GUI는 각각 별도 프로세스, 제어는 기본 25 Hz 루프에서 동작합니다.
GUI의 정지 입력은 추론 완료를 기다리지 않고 공유 이벤트로 전달됩니다. 캡처 스레드는
추론 중에도 최신 프레임을 받아 두 팔의 관절값을 함께 기록합니다. 수신 시각과
카메라 타임스탬프로 노후화·관절 샘플 시차를 검사하지만, 하드웨어 동기화는
아니며 디바이스 시계의 노출 시각은 호스트 시계로 근사 변환합니다.

### 좌표계와 플랜지·TCP 관계

`T_A_B`는 B 좌표의 점을 A 좌표로 옮기는 4×4 강체변환입니다.
`C`는 카메라, `M`은 학습 모델 베이스, `B`는 UR 제어기 베이스,
`F`는 제어기의 TCP offset이 0일 때의 말단 좌표계, `TCP`는 실제 활성 공구 좌표계입니다.
여기서 플랜지 `F`는 UR `tool0` 관례를 따르며 ROS의 이름이 `flange`인 부착 프레임과
축 정의가 같다고 가정하면 안 됩니다. 내부 이동 단위는 m, 회전 벡터는 rad입니다.

```text
T_C_B = T_C_M · T_M_B
T_B_F = T_B_TCP · inverse(T_F_TCP)

카메라 고정 목표: T_B_TCP* = inverse(T_C_B) · T_C_F* · T_F_TCP
상대 추종 목표:   T_Bf_TCP* = inverse(T_C_Bf) · T_C_Bl · T_Bl_Fl · D(u) · T_Ff_TCP
```

상대식의 `l`은 리더, `f`는 팔로워, `D(u)`는 리더 플랜지 기준의 목표 팔로워 플랜지입니다.
초기 마주보기 목표는 `D(0) = TransZ(gap) · RotX(π)`이며, 측정된 현재 상대 자세로
매번 다시 설정하지 않습니다. 30 cm는 **플랜지 원점 사이** 거리이고 TCP나 공구 끝의 여유 거리가 아닙니다.
직선·원 경로에서는 `D(u)`가 변하므로 최초 마주보기의 정확한 위치 관계도 경로에 따라 변합니다.

UR5e의 `T_M_B`는 항등변환이고, UR10e는 +5 mm Z 이동과 Z축 π 회전입니다.
이 연결은 이 저장소의 학습/FK 정의에 맞춘 값입니다. 두 팔의 실제 활성 TCP offset을 적용하므로
공구를 장착한 상태에서도 플랜지 목표와 명령 TCP를 구분합니다.

기존 `reproj_mean` 대신 **EM 보정 전 원래 검출점**에 대한 재투영 오차를 다시 계산합니다.
고정 모드의 필터는 위치 선형 보간과 SO(3) 회전 보간을 사용합니다.
연구용 팔로워 모드의 시간 정합·필터·예측은 아래 설명을 따릅니다.

기본 구동 제한은 다음과 같습니다. 이 값들은 충돌 없는 경로를 보장하지 않습니다.

| 항목 | 기본값 |
| --- | --- |
| 양쪽 캘리브레이션 연속 유효 프레임 | 서로 다른 촬영 샘플 5개 |
| 캘리브레이션 최대 나이 | 0.5초 |
| RTDE 수신 최대 정체 시간 | 0.15초 |
| 유효 특징점 | 각 5개 이상 |
| 원래 검출점 재투영 오차 | 평균 3 px, 최대 8 px |
| 프레임 간 캘리브레이션 변화 | 15 mm / 2도 이하, 상대 베이스도 검사 |
| 선속도 / 각속도 상한 | 0.02 m/s / 0.1 rad/s |
| 명령 가속도 / 각가속도 상한 | 0.1 m/s² / 0.5 rad/s² |
| 현재 TCP와 현재 제어 목표 차이 | 0.10 m / 25도 이하 |
| 시작 TCP 위치로부터 이동 영역 | 반경 0.15 m |
| 최소 플랜지 원점 간 거리 | 0.15 m |

처음부터 대략 마주보는 배치에서 사용해야 합니다. 목표가 너무 멀거나 자세 차이가
크면 작은 속도로 계속 접근시키는 대신 `WAIT`로 거부합니다. 구동 중에는 캘리브레이션
누락·급변·지연, 상태 갱신 정지, 제어기 안전 상태 변화, 도달 불가 목표, 명령 오류가
발생하면 제어하는 팔 모두에 정지를 요청하고 종료합니다. 연구용 follow는 아래의 제한된 측정 공백 보완을 적용합니다.
통신 watchdog도 설정합니다.
제어 스크립트는 Arm 시 한 번만 올리고, RTDE 재연결 시 자동 업로드를 끕니다.
따라서 watchdog으로 멈춘 스크립트를 재연결이 다시 시작하지 않습니다.
SDK 내부의 짧은 재시도와 OS 스케줄링 지연은 남으므로, 호스트의 시간 제한이 엄밀한
실시간 정지 시간을 보장하지는 않습니다. watchdog 설정은 기본 5 Hz입니다.
기존 hold에는 충돌 모델이 없습니다. 연구용 follow는 아래의 nominal 링크/공구 캡슐과 설정한 장애물 여유를 검사합니다.
이 근사 검사는 실제 형상 기반의 충돌 없는 경로를 보장하지 않습니다.
15 cm는 플랜지 원점 기준이며 장착 공구의 여유 거리를 뜻하지 않습니다.
nominal FK 검사는 공장 기구학 보정이나 캘리브레이션의 실측 정확도 검증을 대신하지 않습니다.

기하학·필터·속도 제한 및 모의 RTDE/영상 테스트는 장비 없이 실행할 수 있습니다.
실제 양팔 이동, 제어 주기, 추종 오차와 정지 동작은 이 코드 작성 과정에서 시험하지 않았습니다.

```bash
python3 -B -m unittest discover -s tests -p 'test_dual_control_*.py' -v
```

API/좌표 정의 근거: [ur_rtde API](https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html),
[UR robot frames](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/rolling/doc/ur_description/doc/robot_frames.html),
[RTDE 재연결·스크립트 업로드 구현](https://gitlab.com/sdurobotics/ur_rtde/-/raw/master/src/rtde_control_interface.cpp).

## 연구 계획 기반: 이동 정반 보상과 상대 작업 진행

[연구 구현 계획](docs/Research_Implementation_Plan.txt)의 상태처리·목표 생성 모듈을 추가했습니다.
원문과 기존 네트워크/voting/PnP는 수정하지 않았습니다. 두 팔은 같은 이미지에서 추론하며,
UR10e 리더 / UR5e 팔로워와 30 cm 마주보기 목표를 기본으로 사용합니다.
RTDE 제어 스크립트와 `speedL` 전송 방식도 기존 것을 공유합니다.

```text
기존 포즈 센서 → 촬영 시각·수신 시각 분리 → 상대 상태/단기 예측
             → 고정 상대 목표 또는 상대 작업 경로
             → 보상 + 오차 피드백 + 작업 속도 비율(ρ) × 작업 운동
             → 관절·공간·도달성·충돌 여유 검사 → 기존 RTDE 제어
```

| 계획 단계 | 구현 |
| --- | --- |
| 1. 상대 자세 유지 | `--path hold`: 설정한 상대변환을 유지하며 현재 측정값으로 다시 설정하지 않음 |
| 2. 정반 이동 보상 | 양쪽 베이스 이동과 리더 팔 운동을 반영해 팔로워 base의 목표를 갱신 |
| 3. 시간 정합·필터·예측 | 촬영 시각 기반 SE(3) 필터, 위치·회전 속도와 제한된 단기 예측 |
| 4. 상대 경로·진행률 | `line`/`circle`, 별도 진행률, 작업 일시정지 중에도 보상 유지 |
| 5. 운동 여유 관리 | 진행률을 먼저 선택하고 전체 명령을 검사; 보상 자체가 불가능하면 중단 |

이 표는 소프트웨어 구현 범위입니다. 정반을 실제로 움직이며 각 단계를 검증한 실험 결과를 뜻하지 않습니다.
두 팔이 보상을 나누어 수행하는 기능, 힘 제어·접촉·물체 인계 제어는 포함하지 않습니다.

같은 `hold`라는 이름의 두 기능은 기준 좌표와 구동 대상이 다릅니다.

| 옵션 | 고정하는 것 | 베이스/리더가 움직이면 |
| --- | --- | --- |
| `--mode hold` | 카메라 좌표계의 두 플랜지 목표 | 양팔이 각각 그 카메라 목표를 유지하도록 움직임 |
| `--mode follow --path hold` | 리더 플랜지 기준의 상대 목표 `D` | 팔로워가 리더를 따라 움직이며 같은 상대 관계를 유지 |

카메라 고정 모드가 물리 공간의 고정 자세를 뜻하려면 카메라도 고정되어 있어야 합니다.
상대 경로의 `hold`와 `Pause task`는 팔을 정지시키는 명령이 아닙니다.

미리보기 실행 예시는 다음과 같습니다. 기존 카메라 프로그램은 종료한 뒤 실행하세요.

```bash
# 정반이 움직여도 30 cm 상대 자세 유지
./run_cooperative.sh --method separated --path hold \
  --ur5e_ip 169.254.186.10 --ur10e_ip 169.254.186.6

# 리더 플랜지 기준 Z 방향으로 5 cm 접근하는 상대 경로
./run_cooperative.sh --method separated --path line \
  --path_delta 0 0 -0.05 0 0 0 --path_duration 10 \
  --ur5e_ip 169.254.186.10 --ur10e_ip 169.254.186.6

# 반지름 2 cm, 리더 플랜지 XY 평면의 한 바퀴 경로
./run_cooperative.sh --method separated --path circle --path_radius 0.02 \
  --ur5e_ip 169.254.186.10 --ur10e_ip 169.254.186.6
```

`run_dual_control.sh --mode follow`도 같은 모듈을 실행합니다. GUI에서 간격·경로·이동량·반지름·시간을
바꾸고 `Apply target`으로 적용할 수 있습니다. 경로 이동량은 **리더 플랜지 좌표계**입니다.
GUI의 이동량은 cm, 회전 증분은 **회전벡터 성분의 도(deg)**입니다. CLI는 m/rad입니다.
기존 카메라 고정 hold GUI의 Roll/Pitch/Yaw Euler 입력과 구분하세요.
원 경로는 설정한 시작 목표에 `[r(cosθ−1), r sinθ, 0]`을 더해 한 바퀴 돌며 시작점으로 돌아옵니다.
원 경로의 상대 방향은 고정이고, 직선 경로의 회전 증분은 리더 플랜지 축에서 시작 방향에 적용됩니다.
선·원 경로 모두 `s(u)=10u³−15u⁴+6u⁵`를 사용해 경로 성분의 명목 속도·가속도가 시작과 끝에서 0이 됩니다.
베이스/리더가 움직이는 동안에는 경로 끝에서도 보상 운동이 남을 수 있습니다.

실제 구동은 위 명령에 `--execute`, `--ur5e_tcp_offset X Y Z RX RY RZ`,
`--ur10e_tcp_offset X Y Z RX RY RZ`를 실제 설치값으로 추가한 다음 `Arm motion`으로 시작합니다.
미리보기는 진행률을 증가시키지 않습니다. 구동 중에도 전송이 성공한 명령만 진행률에 반영합니다.
`Pause task`는 가속도 한계 안에서 작업 속도를 줄여 진행률을 멈춥니다. 이 과정에서 잠깐 경로가 더 진행할 수 있으며,
진행률이 멈춘 뒤에도 베이스/리더 운동 보상은 계속합니다. `STOP / Exit`는 로봇 제어 자체를 종료합니다.

### 세 가지 비교 방식

| `--method` | 상태 처리 | 명령 생성 |
| --- | --- | --- |
| `direct` | 원본 포즈를 공통 촬영 시각에 정합; 현재 시각 외삽 없음 | 목표 오차의 P 피드백; 기본 명목 속도로 경로 진행 |
| `predict` | SE(3) 필터와 현재 시각 단기 예측 | direct와 동일한 P 피드백·기본 경로 진행 |
| `separated` (기본) | predict와 동일 | 베이스/리더 보상 + P 피드백 + 작업 feedforward; 오차와 운동 여유에 따라 경로 감속 |

세 방식은 포즈 센서, 피드백 이득, 저수준 RTDE 제어와 물리 한계 검사를 공유합니다.
비교 방식 간 차이는 후속 상태 처리와 상위 목표/속도 생성입니다. `direct`/`predict`는 보상 feedforward와
오차 기반 자동 작업 감속을 사용하지 않습니다. 두 방법도 목표를 실시간으로 다시 계산하므로 움직이는 리더를
오차 피드백으로 추종합니다. 사용자 일시정지, 측정 공백 처리, 한계·기하 검사와 실행 불가 시 중단은 공통입니다.
방식은 실행 시 `--method`로 선택하며 구동 중 GUI에서 전환하지 않습니다.

같은 영상의 두 추정값은 완료 시각과 무관하게 같은 촬영 시각을 사용합니다. 별도 비동기 입력 API도 제공하며,
direct는 공통 과거 시각으로 보간하고 보간 구간이 없으면 거부합니다. predict는 두 상태를 현재 시각으로
예측합니다. 출력은 상대변환, 회전하는 리더 좌표계의 영향을 포함한 상대 속도, 마지막 측정 이후 경과시간입니다.
필터는 공간 twist 기반 SE(3) 보정이며, 공통 강체 운동에서 일정한 상대변환이 유지되도록 구성했습니다.
상대 베이스는 `inverse(T_C_Bl) · T_C_Bf`로 계산하고, 실제 상대 플랜지는 각 베이스에 관절/제어기 FK를
연결해 계산합니다. 베이스 속도는 카메라 좌표의 베이스 원점 선속도와 각속도이며, 상대 속도 계산에는
회전하는 리더 좌표계의 영향도 포함합니다. 품질 검사와 상태 갱신은 양팔 쌍으로 적용하므로
한쪽 측정만 반영된 상태로 다음 주기를 진행하지 않습니다.

기본 예측 한계는 촬영 이후 0.5초입니다. 구동 중 잠깐의 잘못된 측정은 마지막 유효 상태로 보완하면서 작업 진행을
감속·정지시키지만, 무효 측정이 0.3초 지속되거나 측정/예측 수명이 끝나면 보상도 중단합니다.
예측으로 갑작스러운 정반 이동을 정확히 알아내거나 실제 이동과 추정 오류를 항상 구분하지는 못합니다.
`--velocity_alpha`, `--prediction_horizon`, `--max_invalid_duration` 등은 실험 설정과 함께 로그에 남습니다.

### 실행 가능성 검사

`separated`의 속도 명령은 다음 성분으로 구성됩니다. 모든 성분은 **팔로워 base에서 표현한 TCP의
선속도·각속도**이며, TCP offset이 있을 때 회전에 따른 공구 끝의 선속도도 포함합니다.

```text
V_command = V_base + V_leader + β V_feedback + ρ V_task
u_next = min(1, u + ρ · dt / path_duration)
```

`V_base`는 두 베이스의 상대 운동 보상, `V_leader`는 리더 관절 운동에 따른 목표 이동,
`V_feedback`은 현재 TCP 목표 오차 보정, `V_task`는 상대 경로의 명목 시간 미분입니다.
`ρ`는 작업 속도 비율, `u`는 별도로 저장하는 경로 진행률입니다. `hold` 경로는 `u`를 증가시키지 않습니다.
`direct`/`predict`는 앞의 보상 두 항과 작업 feedforward를 명령에 더하지 않습니다.

전체 명령을 만든 뒤 단순히 잘라 보내지 않습니다. 필요한 보상은 축소하지 않고 허용되는 작업 속도 비율을 찾습니다.
피드백에는 별도의 deadband·크기·변화율 제한이 있으며, 실행 가능한 조합을 찾기 위해
피드백 비율 `β`를 1, 0.75, 0.5, 0.25, 0 순서로 시도합니다.
각 피드백 후보에서 Cartesian 속도/가속도와 Jacobian으로 계산한 관절 속도/가속도의 가능한 `ρ` 구간을
먼저 계산합니다. 총 명령은 측정된 현재 속도와 직전 전송 명령 양쪽에 대해 가속도 제한을 확인하고,
필수 보상 자체는 이전 보상 이력에 대해 따로 검사합니다.

그 구간 안에서 최대 9개 작업 비율 후보를 평가해 다음 목표의 국소 IK, 관절 위치, 작업공간,
팔 사이·장애물 여유를 확인합니다. 각 명령의 반 주기와 한 주기 뒤 자세도 샘플링합니다.
속도·가속도 구간은 수학적으로 계산하지만 기하 검사는 유한 후보 검색이므로,
모든 가능한 경로를 찾거나 연속 시간의 충돌 없음을 증명하지는 않습니다.
일시정지/감속 시 즉시 `ρ=0`으로 바꾸면 가속도 한계를 넘는 경우에는 잠깐 더 진행하며 감속합니다.
보상 자체가 속도·가속도·기하학 한계를 넘으면 `compensation_infeasible`로 중단합니다.
측정 만료, 추종 오차, 특이점, 도달 불가, 충돌 여유 부족은 별도 원인으로 기록합니다.

| 연구용 추가 설정 | 기본값 |
| --- | --- |
| 관절 명령 속도 / 가속도 | 각 0.2 rad/s / 0.5 rad/s² |
| Jacobian 최소 특이값 | 0.01 (m/rad를 포함한 nominal Jacobian 기준) |
| 위치 / 회전 피드백 deadband | 1 mm / 0.3도 |
| 작업 감속 시작 오차 | 1 cm / 5도 |
| 목표 도달 판정 | 경로 진행률 1 + 추정 오차 5 mm / 2도 이내 0.5초 유지 |
| 링크 캡슐 반지름 | UR5e 6 cm / UR10e 8 cm |
| 공구 캡슐 반지름 / 추가 여유 | 4 cm / 2 cm |

위 반지름은 **실험용 근사 모델 설정**이며 실제 로봇/공구 형상을 검증한 값이 아닙니다.
`--workspace_config configs/workspace.example.json`으로 관절 위치 제한, base 기준 TCP 박스, 링크/공구 반지름,
카메라 기준 구형 장애물을 지정할 수 있습니다. 예시에는 실제 작업장 장애물이 들어 있지 않습니다.
[설정 예시](configs/workspace.example.json)를 실제 배치에 맞춰 별도 파일로 작성해 사용하세요.
팔 사이 캡슐과 설정한 장애물은 검사하지만, CAD mesh와 자기 충돌의 완전한 모델은 없습니다.
nominal IK/Jacobian 검사는 공장 보정을 대신하지 않으며, 전송 직전 로봇 제어기로 현재·다음 목표를 추가 검사합니다.
실험 시작 전에 정반을 멈추고 두 플랜지를 목표 근처에 배치해야 합니다.

정반은 외부에서 움직이는 비구동 장치이므로 **로봇 Stop과 별도로 정반 이동도 중지**해야 합니다.
팔로워 모드의 Stop은 제어 중인 팔로워에 적용되며, 외부에서 실행 중인 리더 프로그램을 제어하지 않습니다.
실험에서는 외부 이동을 멈출 담당자/장치를 정하고, 중단 로그를 확인한 뒤 재배치·미리보기부터 다시 시작하세요.

### 기록과 검증

기본 로그는 `runs/<UTC시각>-<방식>-<고유값>.jsonl`입니다. `--log 경로`로 지정할 수 있으며 기존 파일은 덮어쓰지 않습니다.
촬영/수신/제어 시각, 원본 포즈와 품질, 상대 상태와 속도, 관절 위치·속도, TCP, 생성 목표,
보상/피드백/작업 성분, 추정 상대 오차, 명령 전·후 진행률, 전송 여부, 완료와 중단 원인을 기록합니다.
디스크 쓰기는 별도 스레드에서 처리하며 큐 초과나 쓰기 실패 시 실험을 중단합니다.

```bash
python3 -m urcal.tools.analyze_log runs/실험.jsonl
python3 -m urcal.tools.analyze_log runs/실험.jsonl \
  --reference independent_reference.jsonl --output report.json

# 장비 없는 합성 비교: 세 방식에 동일한 베이스 운동·지연·노이즈 조건 적용
python3 -m urcal.tools.simulate --output_dir runs/synthetic-comparison \
  --scenario bases --duration 20 --seed 7
```

독립 기준 파일은 각 줄에 `{"at": 제어와_같은_단조시계_초, "relative_pose": 4x4행렬_또는_pose6}`를 넣습니다.
상대변환은 리더 플랜지 좌표로 표현한 팔로워 플랜지 포즈 `T_Fl_Ff`이며,
팔로워 좌표를 리더 좌표로 옮깁니다. pose6 단위는 m/회전벡터 rad입니다.
모션캡처 등 외부 장치의 시계는 먼저 정합해야 하며, 기본 ±0.03초 안의 기준만 사용합니다.
제어에 사용한 추정 오차와 독립 기준 오차를 별도 출력하며, 기준 파일이 없으면 실제 오차를 만들어 내지 않습니다.

경로 진행률 완료와 추정 목표 도달·유지 완료는 구분합니다. `path_finished`는 기준 경로의 진행률이 1인 상태이고,
`finished`는 그 이후 추정 상대 오차가 위치·회전 허용값 안에 유지된 시간이 `finish_dwell`을 충족한 상태입니다.
미리보기나 전송 실패 주기는 이 유지 시간에 포함하지 않습니다. 완료 후에도 종료하지 않고 마지막 상대 목표를 유지합니다.
따라서 `finished`도 독립 실측에 의한 정확도 판정은 아닙니다. 감속 비율은 미완료 작업 구간에서만 집계하고,
완료 이후 상대 자세 유지 시간은 별도입니다. 세 방법의 오차뿐 아니라 완료시간·진행률도 함께 비교해야 합니다.
시뮬레이터는 Cartesian 6자유도 속도 적분 모델입니다. 독립 합성 기준은 제공하지만 실제 UR 동역학,
정반·공구 충돌, 영상 정확도 또는 실험실의 실제 오차를 검증하는 도구는 아닙니다.

실제 카메라와 양팔을 구동한 검증은 수행하지 않았습니다. 측정된 제어 주기·정지 지연·독립 기준 오차는
실험 장비에서 단계별로 확인해야 합니다. 단위·회귀 테스트는 `python3 -B -m unittest discover -s tests -v`로 실행합니다.
패키지 정리 후 전체 278개 검사 중 275개가 통과했고 CUDA 전용 3개는 건너뛰었습니다. 시간 정합, 회전 예측,
보상/작업 분리, 실행 가능성, 전송 실패 시 진행률 보존, 로그 분석과 모의 장비 정지를 포함합니다.
추가로 저장소 밖 작업 디렉터리에서 5개 실행 스크립트와 두 원본 추론 도움말 실행을 확인했고,
이동한 모듈로 두 실제 체크포인트의 CPU `strict=True` 로드가 성공했습니다.
패키지 정리 과정에서 핵심 네트워크·투표·PnP 코드, 원본 비교 파일과 가중치의 내용을 보존했습니다.
실제 nominal FK와 캡슐 검사를 사용한 정지 자세에서 `prepare()` 계산은 방식별 중앙값 10.90~14.68 ms였습니다
(3회 준비 후 각 10회, 최대 14.75 ms). 영상 추론·RTDE 통신을 포함한 전체 주기나 최악 실행 시간은 아닙니다.

## UR10e ROS 2 포즈 출력

UR10e 코드는 선택적으로 `geometry_msgs/PoseStamped`와 재투영 오차를 발행합니다.
설치된 ROS 2 배포판의 환경을 source하고, `rclpy`, `geometry_msgs`, `std_msgs`가
import 가능한 Python 환경에서 실행하세요. 일반 venv에 pip 패키지만 설치하는
것으로는 ROS 2 Python 환경이 구성되지 않습니다.

```bash
./run_ur10e.sh --robot_ip 169.254.186.6 --show_seg \
  --ros_pose_topic /robotpose/ur10e/pnp_pose \
  --ros_frame_id camera_color_optical_frame \
  --ros_reproj_topic /robotpose/ur10e/reproj_error

# ROS 2 환경을 준비한 다른 터미널
ros2 topic echo /robotpose/ur10e/pnp_pose
```

포즈는 코드의 FK/학습 데이터 베이스 좌표를 카메라 좌표로 변환하는 `T_C_B`입니다.
이동 단위는 m, 자세는 quaternion입니다. 원본 UR10e 코드는 학습 데이터와 맞추기
위해 베이스에 0.005 m의 Z 오프셋을 적용합니다. UR 제어기 좌표와 연결할 때는
원본 코드의 베이스 좌표 정의까지 확인해야 합니다.
UR5e 원본 코드에는 ROS 발행 옵션이 없습니다. 두 코드 모두 기본 실행에서는
화면을 표시하며 포즈를 파일에 자동 저장하지 않습니다.

## 복사본 확인

폴더를 옮긴 뒤 다음 명령으로 파일 무결성을 확인할 수 있습니다.

```bash
sha256sum -c SHA256SUMS
```

실행 스크립트가 동봉한 `torch_cache`를 사용하므로 ResNet 초기화 파일을
추가로 다운로드할 필요가 없습니다. URDF, CAD mesh, 학습 데이터는 이 추론 경로에서
읽지 않습니다. 파일 출처와 원본 대비 조정 사항은 `MANIFEST.json`에 기록합니다.

## 확인한 범위

- 두 로봇 체크포인트, ResNet 파일 및 `reference/` 코드의 원본 일치 여부를 확인했습니다.
- 두 실행 스크립트의 문법 및 `/tmp`에서 기본/원본 버전 도움말 실행을 확인했습니다.
- 기존 Python 3.10 환경에서 필수 패키지 import와 동봉한 캐시만 사용하는 모델
  초기화를 확인했습니다. 두 체크포인트 모두 신경망에 `strict=True`로 로드됩니다.
- 초기 추론 최적화 회귀 테스트 19개 중 CPU 검사 16개가 통과했습니다. CUDA 접근이 불가능하여
  GPU 검사 3개는 건너뛰었습니다. 원본 대비 비교 허용 오차는 0입니다.
- EMPnP는 두 로봇의 FK에서 생성한 144개 사례에서 성공 결과 140개와 초기화 실패
  4개가 원본과 동일했습니다. 4~7개 키포인트, 왜곡, 잡음, 이상치를 포함합니다.
- 실제 저장 RGB 프레임 한 장을 두 체크포인트에 입력한 CPU FP32 검사에서 전체
  segmentation/vector 출력이 원본과 완전히 일치했습니다. 이는 구현 동등성 검사이며
  UR10e 실영상 정확도 평가를 의미하지 않습니다.
- 검증 환경의 패키지 버전은 `MANIFEST.json`에 기록했습니다. 실제 카메라/로봇 연결,
  GPU 추론 속도·동등성, 실환경 정확도 및 ROS 2 발행은 이번 작업에서 시험하지 않았습니다.

## 적용한 최적화

| 위치 | 변경 |
| --- | --- |
| [inference/vote2.py](urcal/inference/vote2.py) | 각도 구간별 픽셀 선택을 GPU reduction으로 처리. 구간 순서와 동점 시 첫 픽셀 선택을 유지 |
| [inference/vote2.py](urcal/inference/vote2.py) | 채널별 sigmoid·mask·픽셀 수 재사용, 이웃 선택용 픽셀 수를 한 번에 CPU로 전달 |
| [inference/](urcal/inference/)의 `infer_live*.py` | segmentation 창이 꺼져 있으면 디버그 확률 맵 생성·CPU 복사 생략 |
| [inference/network3.py](urcal/inference/network3.py) | eval 모드에서 같은 입력의 LayerNorm 결과 재사용. attention 연산 경로와 덧셈 순서 유지 |
| [inference/my_pnp.py](urcal/inference/my_pnp.py) | 고정 가중치 및 prior 곱 재사용, 동일 파라미터에 대한 연속 residual 요청 캐싱 |

입력 해상도, 가중치, FP16 설정, 투표 가설 수, 임계값, 난수 사용 순서,
EMPnP 반복 횟수·수렴 기준·RANSAC 초기화는 유지했습니다. 새로운 attention 커널,
해석적 Jacobian, 포즈 warm-start, 해상도 축소는 적용하지 않았습니다.

참고 CPU 측정에서 합성 입력(7개 키포인트, 120×160, 1 thread)의 **투표 단계만**
중앙값 35.182 → 5.822 ms였습니다. 합성 144개 사례의 **EMPnP 합계**는
414.475 → 386.586 ms였습니다. 전체 라이브 FPS나 GPU 개선율로 해석하면 안 됩니다.

## 회귀 검사 및 원본 비교

카메라/로봇 연결 없이 검사할 수 있습니다. CUDA가 사용 가능한 환경에서는
회귀 검사의 GPU 항목도 자동 실행됩니다.

```bash
python -B -m unittest discover -s tests -v

# 저장한 RGB 이미지로 두 실제 체크포인트의 전체 신경망 출력 비교
python -B tests/verify_checkpoints.py --image /경로/frame.png --device cpu

# GPU에서는 라이브와 같은 FP16/channels_last 설정으로 비교
python -B tests/verify_checkpoints.py --image /경로/frame.png --device cuda
```

최적화 전 버전으로 실행하려면 같은 명령 앞에 `ROBOTPOSE_REFERENCE=1`을 붙이세요.
가중치는 공통으로 사용하며 Python 코드만 `reference/`에서 불러옵니다.

```bash
ROBOTPOSE_REFERENCE=1 ./run_ur5e.sh --robot_ip 169.254.186.10
ROBOTPOSE_REFERENCE=1 ./run_ur10e.sh --robot_ip 169.254.186.6
```
