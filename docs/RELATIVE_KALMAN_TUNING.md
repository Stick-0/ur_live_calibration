# 상대 베이스 Kalman 추정: 잠정 튜닝 기록

작성 기준: 2026-11-02. 구현은 [relative_kalman.py](../urcal/cooperative/relative_kalman.py), 오프라인 분석기는 [tune_relative_kalman.py](../urcal/tools/tune_relative_kalman.py), 실행 설정은 [relative_kalman.provisional.json](../configs/relative_kalman.provisional.json)이다.

현재 설정은 **라벨 없는 실제 로그의 산포·촬영 지연을 참고하고 합성 정답으로 검증한 잠정값**이다. 사용자가 별도 라벨이 없음을 확인했다. 실제 정반의 위치·속도 정답은 없으므로, 실제 정반 이동에 대한 정확도나 추종 지연 개선을 입증한 결과로 해석하면 안 된다. 기존 로그·마커리스 모델·가중치를 변경하지 않고 분석했으며, 이 검증 과정에서 로봇 명령을 보내지 않았다.

## 무엇을 관측하는가

같은 촬영 시각의 두 베이스 추정에 고정된 모델→실제 베이스 변환을 적용한다. 팔로워의 실제 베이스를 (B_f), 리더를 (B_l)이라고 하면:

\[
G_k=B_f(t_k)^{-1}B_l(t_k),\qquad z_k=[G_{03},G_{13},\operatorname{atan2}(G_{10},G_{00})]^T.
\]

공면 조건을 통과한 상대 위치 (x,y)와 yaw만 사용한다. 공통 카메라 움직임은 이상적인 동일 시점 변환에서는 소거된다. 카메라 기준 두 베이스 속도를 따로 차분하지 않는다. 분석 도구는 공면이 아닌 기록을 임의 투영하여 표본을 만들지 않고 제외한다. 이 물리 좌표 정의는 과거 화면 출력 보정 토글과 독립적이다.

상태는 (s=[x,y,\psi,v_x,v_y,\omega]^T), 관측 행렬은 (H=[I_3\;0])이다. 촬영 시각 간격 (\Delta t)로 다음 모델을 사용한다.

\[
F=\begin{bmatrix}I&\Delta tI\\0&I\end{bmatrix},\quad
Q=\begin{bmatrix}\Delta t^4 A/4&\Delta t^3 A/2\\\Delta t^3 A/2&\Delta t^2 A\end{bmatrix},\quad
A=\operatorname{diag}(\sigma_a^2,\sigma_a^2,\sigma_\alpha^2).
\]

yaw 혁신은 \([-\pi,\pi)\)로 감싸고, 공분산 갱신은 Joseph 형식을 사용한다. 필터 관측 갱신은 새 촬영 시각에서만 일어난다. 조회 주기나 패킷 도착 간격을 새 관측으로 취급하지 않는다.

현재 시각으로의 예측에는 관측 갱신 후 속도 SNR과 연속 표본 수로 정한 축별 `prediction_gain`을 사용한다. 이 계수는 다음 관측까지 고정되며 예측 위치와 그 미분인 속도에 함께 적용된다. 따라서 XY 이동이 분명해도 잡음 수준의 yaw 속도를 따로 외삽하지 않는다. 예측은 0.175초까지 일정 속도, 이후 0.35초까지 속도를 선형 감쇠한다. 공분산·관측 나이로 정한 추가 계수 `lambda`는 실제 보상 속도에 한 번만 적용된다. `lambda`만으로 보상이 켜졌다고 판단하지 말고 **`prediction_gain`과 최종 `origin_velocity`를 함께 확인**해야 한다.

## 실제 로그에서 확인한 범위

보관된 JSONL 93개 중 11월 1일 기록 22개를 목록 점검하고, 같은 공면 처리·필터 설정으로 실행한 최신 4개를 상세 분석했다. 상세 표본은 원본 `measurement`, `alignment_vision`, `handoff_measurement`의 같은 촬영 쌍이다. 상위 게이트에서 거절했더라도 유효 점 수·변환이 정상인 쌍은 포함하여 거절 후 남은 자료만으로 잡음을 과소평가하지 않았다. 중복 촬영, 없는 포즈, 유효 점 부족, 공면 조건 위반은 제외 수로 기록한다.

파일명의 시간은 UTC이다. 예를 들어 `20261101T130805Z`는 한국시간 11월 1일 22:08:05이다. 다음 구간 시간은 각 파일의 **첫 유효 공면 촬영 쌍부터의 초**이다.

| 실행 파일의 UTC 시간·ID | 유효 쌍 | PREVIEW 구간 | ALIGNMENT 구간 | ACTIVE 구간 |
|---|---:|---|---|---|
| `130112Z-…-2479f01f` | 1,184 | 0–18.05초 | 23.72–26.15초 | 26.59–98.07초 |
| `130341Z-…-3206b63c` | 826 | 0–27.25초 | 33.22–39.46초 | 40.13–83.89초 |
| `130549Z-…-2cac12f2` | 860 | 0–2.97초 | 9.27–17.18초 | 17.78–91.36초 |
| `130805Z-…-3af9fd05` | 863 | 0–14.31초 | 20.55–28.69초 | 29.19–84.36초 |

관절 기록 차분은 **팔이 움직였다는 증거**로만 사용한다. 촬영 간격이 0.5초 미만이고 어느 관절이든 0.01rad/s를 넘으면 팔 운동 표본으로 분류한다. PREVIEW에서는 네 실행 모두 이 조건의 팔 운동이 없었다. 마지막 실행 ALIGNMENT 88표본 중 67표본, ACTIVE 597표본 중 532표본에서 팔 운동이 관측됐다. 이것만으로 베이스·정반이 고정됐다고 확정할 수는 없다.

ALIGNMENT의 “keep leader/bases stationary”는 운영 지시이며 독립 측정 라벨이 아니다. ACTIVE의 PnP 변화 역시 실제 정반 이동, 자세에 따른 추정 편향, 관절·영상 시각 불일치가 섞일 수 있다. 따라서 실제 “정반 고정·팔 고정 / 정반 고정·팔 운동 / 정반 이동” 세 조건을 이번 기록으로 확정 분리하지 않았다.

## 산포와 시간 지연에서 고른 값

각 구간 중앙값을 뺀 잔차의 `1.4826 × MAD`를 계산했다. yaw는 원형 평균 부근으로 펼친 뒤 처리한다. 다음은 **관측 산포이며 검증된 센서 잡음이나 절대 정확도가 아니다**.

| 실행 UTC 시간 | PREVIEW X/Y/yaw 산포 | ALIGNMENT X/Y/yaw 산포 |
|---|---|---|
| 13:01:12 | 0.84 / 1.65mm / 0.070° | 1.13 / 2.74mm / 0.071° |
| 13:03:41 | 0.74 / 1.33mm / 0.060° | 1.95 / 4.34mm / 0.130° |
| 13:05:49 | 0.61 / 2.18mm / 0.100° | 1.22 / 4.03mm / 0.159° |
| 13:08:05 | 0.86 / 1.38mm / 0.069° | 3.61 / 6.27mm / 0.228° |

PREVIEW만 기준으로 1–2mm 잡음을 가정하면 팔 운동 중의 산포를 크게 과소평가한다. 잠정 관측 표준편차는 프로토콜상 고정으로 기대한 PREVIEW·ALIGNMENT 최대 산포를 올림해 **7mm / 0.3°**로 설정했다. ACTIVE 산포를 잡음으로 학습하지 않았다. 공분산은 강건한 주변 스케일과 제한한 상관계수로 별도 기록하지만, 실행 R은 대각 행렬이며 XY에는 두 축 중 큰 표준편차를 적용한다.

마지막 ALIGNMENT의 잔차 1차 자기상관은 X 0.927, Y 0.893, yaw 0.767이다. 같은 실행 PREVIEW는 −0.023, −0.045, 0.015였다. 팔 운동 중 산포를 독립 백색 잡음만으로 설명하기 어렵다는 근거다. 이 수치로 편향의 원인을 확정하거나 저주파 변화를 실제 정반 운동으로 판정하지는 않는다. 시간 차분 잔차를 전부 R로 흡수하는 튜닝도 하지 않았다.

| 실행 UTC 시간 | 촬영→수신 중앙값 / 95백분위 | 제어 시점 관측 나이 중앙값 / 95백분위 |
|---|---|---|
| 13:01:12 | 144 / 177ms | 166 / 222ms |
| 13:03:41 | 153 / 190ms | 194 / 260ms |
| 13:05:49 | 158 / 201ms | 194 / 260ms |
| 13:08:05 | 154 / 196ms | 184 / 242ms |

이 과거 로그의 촬영 시각에는 당시 구현의 보수적인 한 프레임 시간 차감이 포함된다. 새 모드는 센서 시각을 매핑하고 한 프레임을 별도로 빼지 않으며, 시각 불확실성은 따로 기록한다. 따라서 위 값은 **과거 방식으로 측정한 보수적 참고치**이며 새 모드 실행 지연의 실측치가 아니다.

과거 타임스탬프 기준으로 0.15초 예측 지평이면 정상적인 관측도 제어 시점에 이미 지평을 넘는 경우가 많다. 최대 관측 나이 95백분위에 50ms 여유를 더해 50ms 단위로 올림한 **0.35초**를 잠정 채택했다. 이 값은 지연의 관측 근거가 있으며, 오래된 관측을 새 값으로 만드는 설정은 아니다. 기존 유효 나이 제한은 별도로 적용된다. 새 촬영 시각 매핑·관절 보간의 실제 효과는 과거 로그만으로 재현·검증하지 못한다.

## 현재 잠정 설정

JSON의 각도 단위는 rad이다. 아래 속도는 **상대 베이스 상태 추정의 제한**이며 팔로워 TCP·관절 명령의 속도 상한과 다르다.

| 설정 | 값 | 의미 |
|---|---:|---|
| `measurement_std_m` | 0.007m | XY 관측 표준편차 |
| `measurement_std_yaw` | 0.005236rad = 0.3° | yaw 관측 표준편차 |
| `acceleration_std_m` | 0.10m/s² | Q의 선가속도 표준편차 사전값 |
| `acceleration_std_yaw` | 0.15rad/s² | Q의 각가속도 표준편차 사전값 |
| `initial_velocity_std_m/yaw` | 0.05m/s, 0.15rad/s | 초기 속도 불확실성 |
| `prediction_horizon`, `gap_reset` | 각각 0.35초 | 예측 종료 / 장기 관측 공백 재초기화 |
| `min_velocity_samples` | 4 | 속도 신뢰도 누적의 최소 표본 수 |
| `innovation_gate` | 16.27 | 정규화 혁신 제곱거리 상한 |
| `max_step_m/yaw` | 0.15m / 15° | 관측 간 큰 상대 변화 거절 |
| `max_speed`, `max_yaw_rate` | 0.15m/s, 0.5rad/s | 추정 상대 속도 상한 |
| `stationary_window/sigma` | 0.25초 / 0.75 | 짧은 구간의 정지 판단 |
| `reversal_sigma` | 3.0 | 단발 잡음과 큰 반대 방향 변화 구분 |
| `velocity_snr_min/full` | 1.5 / 3.0 | 속도 SNR에 따른 예측 이득 증가 범위 |

따라서 (R=\operatorname{diag}(0.007^2,0.007^2,\operatorname{rad}(0.3)^2)), (A=\operatorname{diag}(0.10^2,0.10^2,0.15^2))이다. Q는 실제 가속도 정답으로 식별하지 않은 사전값이다. yaw Q를 0.30에서 0.15rad/s²로 낮추고, 낮은 SNR에서 위치 외삽까지 줄이는 방식을 합성 정지·이동 비교로 점검했다. 단일 합성 경로의 최저 RMSE를 찾아 Q를 자동 최적화하지 않았다.

## 합성 정답과 독립 seed 검증

합성 입력은 65–95ms의 불규칙 촬영 간격, 180ms 관측 나이, 위 R의 가우스 잡음을 사용한다. 경로는 2초 정지→4초 이동→2초 정지→4초 반전→정지이다. 이동 속도는 X 40mm/s, Y 10mm/s, yaw 0.035rad/s이며 yaw 경계도 넘는다. 필터는 실제 실행 모듈을 그대로 호출한다. 로봇 동역학·제어 게이트·IK·충돌 검사는 이 시험에 포함되지 않는다.

탐색 seed 2718은 [상세 비교](relative_kalman_tuning.synthetic.json)에 남겼다. Q 절반/두 배, yaw Q 절반, 짧은 예측 지평도 함께 기록하며, 선택한 값이 모든 항목의 최저 오차라고 주장하지 않는다. 특히 0.15초 지평은 180ms 나이에서 보상 속도가 항상 0이므로 낮은 산포만 보고 더 좋은 추종이라고 판정하지 않는다.

설정 선택에 사용하지 않은 seed 31415·57721·16180 결과를 합친 값은 다음과 같다. 비교 대상은 동일 코드·동일 측정 입력에서 yaw Q만 0.30rad/s²로 바꾼 경우다. 검증 후 설정을 다시 학습하지 않았다.

| 구간 | 채택값 XY RMSE | 채택값 yaw RMSE | yaw Q=0.30의 yaw RMSE |
|---|---:|---:|---:|
| 최초 정지 | 6.43mm | 0.188° | 0.200° |
| 이동 | 8.91mm | 0.434° | 0.454° |
| 중간 정지 | 4.34mm | 0.197° | 0.202° |
| 반전 | 7.47mm | 0.390° | 0.418° |
| 최종 정지 | 5.96mm | 0.202° | 0.225° |

독립 seed 세 개에서 관측 거절은 0회였다. 이동 구간 평균 X 보상은 +11.36mm/s, 반전 구간은 −10.89mm/s였다. 정지 전환 직후를 포함한 중간·최종 정지 구간은 +2.16 / −1.01mm/s이므로 정지 순간부터 잔여 보상이 완전히 0이라고 주장하지 않는다. 탐색 seed에서는 203표본 중 1표본이 거절됐다.

별도 20초 정지 백색 잡음 시험의 세 seed 합산 결과는 raw XY 9.69mm·yaw 0.301° → 필터 5.14mm·0.188°였다. 가짜 보상 RMS는 선속도 0.228mm/s, 각속도 0.002°/s였다. 단, 20mm·0.5°의 느린 상관 편향을 추가하면 XY 16.81→14.72mm, yaw 0.468→0.407°로 편향이 남았다. **Kalman 필터만으로 지속적인 추정 편향과 실제 정반 이동을 구분할 수는 없다.** 전체 수치와 seed별 결과는 [독립 합성 검증 JSON](relative_kalman_tuning.heldout.json)에 있다.

## 실제 로그 재생의 의미

실제 촬영 간격을 유지하고 관측 나이만 0.18초로 고정해 필터만 재생했다. 네 실행의 거절은 각각 16/1,184, 0/826, 0/860, 2/863이었다. PREVIEW 보상은 모두 0이었다. ALIGNMENT 선속도 보상 RMS는 마지막 실행에서 0.029mm/s이고 나머지는 0이었다. ACTIVE는 각각 9.80, 9.00, 6.49, 20.23mm/s였다. 첫 실행의 ACTIVE 12표본은 관측 유효 나이가 지나 결측값으로 남겼으며 이전 값을 새 측정인 것처럼 채우지 않았다.

이 ACTIVE 속도를 참값과 맞다고 판정하지 않았다. 실제 정반 정답이 없어 적절한 보상인지 추정 편향에 대한 보상인지 구분할 수 없기 때문이다. [실제 로그 재생 JSON](relative_kalman_tuning.replay.json)은 신호 발생·거절·결측 진단 자료이며, 하드웨어 추종 개선이나 과거 중단 상황 재현 자료가 아니다.

## 재현 및 향후 라벨 사용

프로젝트 루트에서 다음 명령을 실행한다. 원본 로그를 읽고 새 분석 파일만 작성한다.

```bash
python3 -B -m urcal.tools.tune_relative_kalman provisional \
  runs/20261101T130112Z-separated-2479f01f.jsonl \
  runs/20261101T130341Z-separated-3206b63c.jsonl \
  runs/20261101T130549Z-separated-2cac12f2.jsonl \
  runs/20261101T130805Z-separated-3af9fd05.jsonl \
  --output configs/relative_kalman.provisional.json \
  --report docs/relative_kalman_tuning.audit.json

python3 -B -m urcal.tools.tune_relative_kalman synthetic \
  --config configs/relative_kalman.provisional.json \
  --output docs/relative_kalman_tuning.synthetic.json

python3 -B -m urcal.tools.tune_relative_kalman validate-synthetic \
  --config configs/relative_kalman.provisional.json \
  --seeds 31415 57721 16180 \
  --output docs/relative_kalman_tuning.heldout.json

python3 -B -m urcal.tools.tune_relative_kalman replay \
  runs/20261101T130112Z-separated-2479f01f.jsonl \
  runs/20261101T130341Z-separated-3206b63c.jsonl \
  runs/20261101T130549Z-separated-2cac12f2.jsonl \
  runs/20261101T130805Z-separated-3af9fd05.jsonl \
  --config configs/relative_kalman.provisional.json \
  --output docs/relative_kalman_tuning.replay.json
```

[세그먼트 manifest 예제](../configs/relative_kalman.segments.example.json)는 현재 증거를 반영해 조건을 모두 `unknown`으로 남겼다. 이 예제를 `fit`에 넣으면 라벨이 있는 정지 베이스 훈련 구간이 없다는 오류가 정상적으로 발생한다. 실제 작업 기록이나 독립 기준 측정이 생겼을 때만 `condition`을 `base_stationary_arm_stationary`, `base_stationary_arm_moving`, `base_moving` 중 하나로 지정하고 근거를 `evidence`에 적는다. 구간은 `[start_s,end_s)`이고 같은 로그의 훈련·검증 구간은 겹칠 수 없다.

```bash
python3 -B -m urcal.tools.tune_relative_kalman fit \
  --segments configs/relative_kalman.segments.labelled.json \
  --output configs/relative_kalman.fitted.json \
  --report runs/relative_kalman_fit_report.json
```

위 명령의 `labelled.json`은 향후 근거가 있는 구간으로 별도 작성하는 파일이다. `fit`은 **베이스가 고정된 train 구간만** R에 사용하고, validation 구간은 R 산정에 사용하지 않는다. 팔 정지·팔 운동 조건을 따로 중심화하고 큰 주변 산포를 채택해 긴 정지 자료가 짧은 팔 운동 자료를 덮지 않게 한다. 검증용 고정 구간은 중앙값 주변 산포만 평가한다. 이동 구간은 독립 위치 정답 없이는 점수화하지 않으며 Q·예측 지평도 자동으로 학습하지 않는다.

회귀 검증은 `PYTHONPATH=tests:. python3 -B -m unittest test_relative_kalman_tuning -q`로 실행한다. 물리 좌표·공면 검증, yaw wrap, 거절 표본 포함, train/validation 분리, 무라벨 오용 방지, 관측 공백 결측 유지, 원본 로그 덮어쓰기 방지, 합성 재현성과 독립 seed를 검증한다.
