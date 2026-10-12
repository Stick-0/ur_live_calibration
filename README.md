# UR5e / UR10e 실시간 카메라–로봇 캘리브레이션

RealSense D435 RGB 영상과 UR RTDE 관절값을 이용해 카메라 기준 로봇 베이스
포즈 `T_C_B`를 매 프레임 추정하는 실행 묶음입니다. 모델별 가중치와 실행
스크립트가 포함되어 있으며, 기본 실행은 중복 연산을 줄인 최적화 버전입니다.
최적화 전 Python 코드 6개는 `reference/`에 보관했습니다.

처리 흐름은 RGB → 신경망/투표 기반 2D 키포인트 검출 → 관절값으로 3D
키포인트 계산 → EMPnP 포즈 추정입니다. 카메라 내부 파라미터와 왜곡 계수는
RealSense에서 읽습니다. 기본 입력 크기는 640×480, 카메라 설정은 30 FPS입니다.
실제 추론 처리 FPS는 하드웨어에 따라 달라집니다.

## 포함 파일

```text
ur_live_calibration/
├── README.md
├── requirements.txt
├── run_ur5e.sh
├── run_ur10e.sh
├── infer_live.py                  # UR5e 추론
├── infer_live_ur10e.py             # UR10e 추론, 선택적 ROS 2 출력
├── infer_live_10.py                # UR10e 기존 진입점 별칭
├── network3.py                    # 공통 신경망
├── vote2.py                       # 키포인트 투표
├── my_pnp.py                      # EMPnP
├── reference/                     # 최적화 전 코드 6개
├── tests/                         # 원본 대비 회귀 검사 및 가중치 검증
├── trained_models/
│   ├── v3m2a3/best.pt              # UR5e
│   └── ur10e_2/best.pt             # UR10e
├── torch_cache/hub/checkpoints/
│   └── resnet18-f37072fd.pth       # 모델 초기화용 ResNet 가중치
├── MANIFEST.json                  # 파일 출처, 가중치 메타데이터
└── SHA256SUMS                     # 파일 무결성 확인
```

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

이 PC에서는 기존 Python 3.10 환경을 사용할 수 있습니다.

```bash
cd /home/jm/Robotpose/ur_live_calibration
source ../.venv/bin/activate
```

다른 PC에서는 이 폴더 전체를 복사하고 Python 3.10 환경에 의존성을 설치하세요.
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

`192.168.0.10`을 실제 로봇 컨트롤러 IP로 바꾸세요. 사용하는 로봇에 맞춰
아래 두 명령 중 하나를 실행합니다.

```bash
# UR5e
./run_ur5e.sh --robot_ip 192.168.0.10 --show_seg

# UR10e
./run_ur10e.sh --robot_ip 192.168.0.10 --show_seg
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

## UR10e ROS 2 포즈 출력

UR10e 코드는 선택적으로 `geometry_msgs/PoseStamped`와 재투영 오차를 발행합니다.
설치된 ROS 2 배포판의 환경을 source하고, `rclpy`, `geometry_msgs`, `std_msgs`가
import 가능한 Python 환경에서 실행하세요. 일반 venv에 pip 패키지만 설치하는
것으로는 ROS 2 Python 환경이 구성되지 않습니다.

```bash
./run_ur10e.sh --robot_ip 192.168.0.10 --show_seg \
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
- 회귀 테스트 19개 중 CPU 검사 16개가 통과했습니다. CUDA 접근이 불가능하여
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
| `vote2.py` | 각도 구간별 픽셀 선택을 GPU reduction으로 처리. 구간 순서와 동점 시 첫 픽셀 선택을 유지 |
| `vote2.py` | 채널별 sigmoid·mask·픽셀 수 재사용, 이웃 선택용 픽셀 수를 한 번에 CPU로 전달 |
| `infer_live*.py` | segmentation 창이 꺼져 있으면 디버그 확률 맵 생성·CPU 복사 생략 |
| `network3.py` | eval 모드에서 같은 입력의 LayerNorm 결과 재사용. attention 연산 경로와 덧셈 순서 유지 |
| `my_pnp.py` | 고정 가중치 및 prior 곱 재사용, 동일 파라미터에 대한 연속 residual 요청 캐싱 |

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
ROBOTPOSE_REFERENCE=1 ./run_ur5e.sh --robot_ip 192.168.0.10
ROBOTPOSE_REFERENCE=1 ./run_ur10e.sh --robot_ip 192.168.0.10
```
