# RoboTwin π0.5 — Docker / A100 80GB × 4 / adapt_to_pi=False

다른 서버에서 GitHub 코드를 clone하고, 별도 데이터 ZIP을 복원한 뒤 Docker로
학습하는 프로젝트다. 기존 서버의 `adapt_to_pi=True` 학습과는 별도 실험이며
공식 `pi05_base`에서 새로 시작한다. 기존 True 체크포인트를 재사용하지 않는다.

다른 서버에서 이미지만 빌드해 전달받으려면
[빌드 담당자 전달 안내서](DOCKER_BUILD_HANDOFF_KO.md)를 사용한다.
GPU·학습 데이터 없이 `bash tools/build_export_image.sh`로 빌드·CPU 검사·이미지
내보내기를 수행하며, 최종 A100 서버에서는 받은 이미지를 `--no-build`로 실행한다.
GitHub Actions로 GHCR에 올린 이미지가 있다면 `docker pull`로 받아 같은 방식으로
실행한다 (아래 "전달받은 이미지 사용" 참고).

**중요:** 아래 "새 서버에서 시작"은 **이 서버에서 직접 이미지를 빌드하는** 경로다.
전달받은 tar 또는 GHCR 이미지가 이미 있다면 이 섹션의 `bash tools/start.sh`와
`docker compose build`를 실행하지 말 것 — 둘 다 전달받은 이미지를 조용히
재빌드해 덮어쓰며, SHA256SUMS/image_manifest.json의 image_id 검증이 무의미해진다.

## 실행 준비

- Linux x86_64, 한 서버의 **A100 80GB 4장**. 이 네 장이 컨테이너에 보여야 한다.
- Docker Engine, Compose v2.33 이상 (`docker compose run --pull`은 v2.33.0부터 지원), NVIDIA Container Toolkit, 호스트 GPU 드라이버.
  CUDA 12.8 wheel 환경이며 GPU 런타임 검사는 아래 명령으로 별도 수행한다.
- CPU RAM은 160GB 이상을 권장한다. 실제 최대치는 첫 모델 초기화 후 확인한다.
- 저장 공간: 원본 데이터·가중치 합계 약 **514.4 GiB**. 압축본을 함께 보관하고
  중간 체크포인트도 남기려면 **여유 공간 2TB 이상**을 권장한다.
- 저장소 접근 권한과 ZIP 또는 비공개 Hugging Face 데이터셋 접근 권한.
  비밀번호·토큰은 Git, Dockerfile, 데이터 ZIP에 넣지 않는다.

호스트의 드라이버와 Docker/NVIDIA 런타임 설치는 컨테이너 안에서 대신할 수 없다.
[NVIDIA 설치 문서](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 전달받은 이미지 사용 (빌드하지 않음)

전달받은 tar 또는 GHCR 이미지가 있으면 이 블록만 실행한다. `build:` 지시자가
있어도 `--no-build --pull never`가 재빌드·재다운로드를 모두 막는다.

```bash
git clone https://github.com/DONGSANSUNJAE/CVPR_VLA_2027.git
cd CVPR_VLA_2027/pi05_robotwin_docker

# (A) 전달받은 tar가 있는 경우
sha256sum -c /path/to/image_export/SHA256SUMS
docker image load --input /path/to/image_export/pi05-robotwin-linux-amd64.tar

# (A') 또는 GHCR에서 받는 경우 — ghcr_image.json의 registry_reference를 그대로 사용
docker pull ghcr.io/dongsansunjae/cvpr_vla_2027/pi05-robotwin-false@sha256:<digest>
docker image tag ghcr.io/dongsansunjae/cvpr_vla_2027/pi05-robotwin-false@sha256:<digest> \
  pi05-robotwin-false:60k

# 둘 다 공통: image_id가 image_manifest.json / ghcr_image.json과 일치하는지 확인
docker image inspect pi05-robotwin-false:60k --format '{{.Id}}'

python3 tools/restore_payload.py --archives /path/to/archives --output payload
python3 tools/check_payload.py

mkdir -p outputs
docker compose run --rm --no-deps --pull never train preflight
docker compose run --rm --no-deps --pull never train gpu-check
docker compose up -d --no-build --pull never train
docker compose logs -f --tail 50 train
```

## 새 서버에서 시작 (이 서버에서 직접 빌드)

전달받은 이미지가 없을 때만 쓴다. GitHub 저장소에서 학습 폴더로 이동한다.

```bash
git clone https://github.com/DONGSANSUNJAE/CVPR_VLA_2027.git
cd CVPR_VLA_2027/pi05_robotwin_docker

# 전달받은 payload-000.zip ... 파일들과 payload_manifest.json이 있는 폴더
python3 tools/restore_payload.py --archives /path/to/archives --output payload

# Docker 이미지 빌드와 백그라운드 학습 시작
bash tools/start.sh
```

첫 이미지 빌드는 잠긴 의존성을 다운로드하므로 인터넷이 필요하다. 이미지와
payload 준비 후 학습 과정에는 Hub 로그인이 필요 없다. `start.sh`는 로그를
보여주며, 여기서 Ctrl+C해도 백그라운드 컨테이너의 학습은 계속된다.

직접 나누어 검사하고 실행하려면:

```bash
mkdir -p outputs
docker compose build
docker compose run --rm train preflight
docker compose run --rm train gpu-check
docker compose up -d train
docker compose logs -f --tail 50 train
```

`preflight`는 전체 index·정규화·초기 가중치 해시와 50과제의 입력 변환을 검사한다.
`gpu-check`는 네 GPU의 BF16 연산·역전파·장치 간 통신을 검사한다. 이 검사를
통과해도 전체 모델의 첫 학습 업데이트까지 통과했다는 뜻은 아니다.

## 중지·재시작·지표

```bash
docker compose stop -t 1800 train
docker compose up -d train
tail -f outputs/logs/metrics.jsonl
cat outputs/logs/train_status.json
```

정상 SIGTERM이면 현재 업데이트 종료 후 체크포인트를 저장한다. 강제 kill이나
서버 전원 종료에서는 마지막으로 저장 완료된 체크포인트까지만 복구된다.
동일 코드·배치 설정으로 resume하며 출력 경로는 `outputs/checkpoints/`다.
기존 True 체크포인트나 다른 recipe의 출력 디렉터리를 넣으면 중단한다.
학습 실패 후 자동 재시작·재제출은 하지 않는다. Docker에는 Slurm 72시간 제한이 없다.

## 고정한 학습 조건

| 항목 | 값 |
|---|---|
| 데이터 | 50과제 × (clean 50 + randomized 500) = 27,500 에피소드 |
| 모델 | JAX π0.5 full FT, 공식 pi05_base |
| Aloha 변환 | **adapt_to_pi=False**, 학습·정규화·추론 동일 |
| 행동 | 팔 12개 관절 delta, 그리퍼 2개 absolute, horizon50 |
| 전역 배치 | 64 = microbatch4 × accumulation16 |
| 업데이트 | 60,000 |
| optimizer | AdamW β1=.9, β2=.95, eps=1e-8, wd=1e-10, 평균 gradient clip=1.0 |
| LR | warmup1,000; cosine peak2.5e-5 → 2.5e-6, decay60,000 |
| EMA / seed | .99 / 42 |
| 저장 | step10, 이후 1,000마다, 마지막·정상 중지 때 |
| 보존 | 최신 + 5,000의 배수; 전부 보관하면 60k까지 약500GiB 추가 예상 |

코드는 기존 실행기의 4 GPU FSDP·gradient accumulation 경로를 사용한다.
원 논문의 미공개 전처리까지 동일함을 주장하지 않는다. 4 GPU 각각에 전체 배치
64를 독립 학습시키는 방식이 아니라, 모델·optimizer를 분할하고 64표본당 한 번
optimizer를 갱신한다. microbatch별 난수는 원본 큰 배치 연산과 비트 단위로 같지 않다.

`False` 정규화는 전 학습 시작점 6,075,103개와 horizon50의 action303,755,150개를
다시 계산했다. 수치와 해시는 `app/assets/.../norm_stats_provenance.json`에 있다.

## 데이터 전달 경로

GitHub에는 코드·작은 설정·정규화만 올린다. 500GiB가 넘는 데이터와 가중치는
GitHub 일반 저장소나 Docker 이미지에 넣지 않는다. 이 프로젝트의 `archives/`는
독립 ZIP64 65개와 완료 manifest의 생성이 끝났다. 압축 파일 합계는
422,792,947,055 bytes(약393.8GiB)다. 각각 약8GiB 이하의 원본을 묶고,
손상·누락·해시 불일치가 있으면 복원을 중단한다. 복원 재실행 시 검증된 기존 파일은 재사용한다.

직접 SSH 전송이 가능하면 `rsync -avP archives/ USER@HOST:/TARGET/archives/`를 쓸 수 있다.
명령의 계정·호스트·대상 폴더는 사용자가 정한 값으로 바꾼다.

Hugging Face를 경유할 경우 `huggingface_hub`가 설치된 환경에서 다음을 사용한다.
먼저 `hf auth login` 등으로 본인의 비공개 저장소 권한을 설정한다. 도구는
공개 저장소로 업로드하지 않으며, 데이터 이전 권한을 가진 저장소를 지정해야 한다.

```bash
# 보내는 서버: 전체 패키지 완료 후
python tools/hub_payload.py upload --repo-id OWNER/PRIVATE_DATASET --directory archives
# 출력된 PIN_THIS_REVISION의 commit SHA를 받는 서버에서 사용
python tools/hub_payload.py download --repo-id OWNER/PRIVATE_DATASET \
  --revision COMMIT_SHA --directory archives
python3 tools/restore_payload.py --archives archives
```

Hub 업로드는 아직 수행하지 않았다. ZIP 안에는 학습 원자료와 공식 초기 가중치가
들어가며, 원 배포물의 라이선스·이용 조건은 그대로 적용된다.

## 검증 범위와 한계

검증 결과는 `VALIDATION_KO.md`에 기록한다. 준비 호스트에는 Docker 엔진과
A100 allocation이 없어 여기에서 Docker build/run 또는 A100 학습 속도를 실측하지
못했다. Docker용 잠금 의존성은 별도 Python3.11 환경에서 CPU로 검사한다.

이 저장소는 **학습 이전 패키지**다. `app/scripts/reproduce_checkpoint.py`는 수치
재현용이며, 기존 서버의 실제 RoboTwin 영상 평가 환경·시뮬레이터 자산은 이번
학습 payload에 포함하지 않았다. 수치 재현을 실제 영상 평가 완료로 보고하지 않는다.

## VESSL Cloud 커스텀 이미지로 사용

VESSL Cloud Workspace의 "Container image" 탭에서 **Custom**을 선택하면
"custom image must include SSHD and Python"이라는 안내가 뜬다. 이 이미지는
`openssh-server`·`curl`과 Python(시스템/venv 둘 다)을 포함하므로 그대로 쓸 수 있다.

- **Custom image URI**: GHCR에 올린 이미지 참조
  (`.github/workflows/build-image.yml` 실행 후 아티팩트의 `ghcr_image.json`에
  있는 `registry_reference`, 예: `ghcr.io/dongsansunjae/cvpr_vla_2027/pi05-robotwin-false@sha256:<digest>`).
  GHCR 패키지가 private이면 VESSL이 pull할 수 있어야 하므로 먼저 public으로
  바꾸거나 VESSL 쪽에 레지스트리 인증 수단이 있는지 확인한다.
- **Start/실행 command**: `idle` 모드로 띄운다 — `python /opt/pi05/entrypoint.py idle`.
  기본 CMD(`train`)로 그대로 두면 payload가 없는 workspace에서 즉시
  `FileNotFoundError`로 죽어 SSH 접속 자체가 안 된다. `idle`은 데이터·GPU를
  전혀 건드리지 않고 `sshd -D`만 foreground로 띄운다.
- VESSL이 공개 키를 어디에 주입하는지는 공식 문서에 명시돼 있지 않다. sshd는
  배포판 기본 설정(루트는 공개키 로그인 허용, 비밀번호 로그인은 기본 비활성)을
  그대로 쓰고, `/root/.ssh`를 0700으로 미리 만들어 둔다 — VESSL이 여기에
  `authorized_keys`를 쓰는 방식이라면 바로 동작한다.
- 이미 이 커스텀 이미지로 실행 중인 VESSL 컨테이너에서는 Docker를 다시 실행하지
  않는다. 복원이 끝난 `/HW/pi05_robotwin_false_60k/payload`를 사용해 아래 실행기를
  호출한다. 입력·GPU 검사 실패 시 학습을 시작하지 않으며 중복 실행을 차단한다.

  ```bash
  bash tools/start_vessl.sh
  tail -n 60 -f /HW/pi05_robotwin_false_60k/outputs/logs/launch.log
  ```

  코드·venv는 이미지의 `/opt/pi05`, `/opt/venv`를 사용한다. 체크포인트와 로그는
  `/HW/pi05_robotwin_false_60k/outputs`에 저장한다. 기존 경로가 다른 파일을 담고
  있으면 덮어쓰지 않고 중단한다. SSH 종료 후에도 계속 실행되지만 workspace의
  Pause/종료 후에는 계속되지 않는다. 현재 `/HW`는 object storage이므로 데이터
  읽기·체크포인트 저장 속도는 실제 실행에서 확인해야 한다. 실행기 구문과 경로
  연결·재실행·충돌 거부 검사는 통과했으며 실제 A100 학습 검증은 별도다.
- (참고) 이전 VESSL 제품 문서에는 Jupyter(`/usr/local/bin/jupyter`, 포트8888)도
  요구한다고 돼 있었다 — 지금 이미지에는 넣지 않았다. workspace UI에 Jupyter
  연결 탭도 있다면 알려주면 `jupyterlab`을 추가한다.

## 예상 완료 시각

실행 중인 VESSL 학습의 예상 완료 시각은 CPU 전용 `tools/training_eta.py`로
확인한다. 원본 학습 로그는 수정하지 않고 `train_status_eta.json`과
`metrics_eta.jsonl`을 같은 로그 폴더에 쓴다. 학습 재시작은 필요 없다.

```bash
nohup /opt/venv/bin/python -u tools/training_eta.py --watch \
  > /HW/pi05_robotwin_false_60k/outputs/logs/eta_monitor.log 2>&1 < /dev/null &
cat /HW/pi05_robotwin_false_60k/outputs/logs/train_status_eta.json
tail -n 5 /HW/pi05_robotwin_false_60k/outputs/logs/metrics_eta.jsonl
```

`estimated_remaining`은 남은 시간, `estimated_finish_kst`는 한국시간 완료
예정 시각이다. 관측 시작·프로세스 변경 뒤 첫 로그 구간은 제외하고 새 구간의
최소20개 업데이트가 모이면 최근 최대100개 업데이트의 가중 평균으로 계산한다.
따라서 보통 관측 시작 뒤30스텝가량부터 ETA가 나온다. 초기 컴파일은 제외한다.
마지막 체크포인트 저장 대기와 향후 중단 시간은 예측에 포함하지 않는다.
`metrics_eta.jsonl`은 원본 지표에 ETA를 붙인 **관측 기록**이다. 관측을 시작한
시점부터 기록하며, 중단 등 상태 변경 시 같은 스텝이 다시 기록될 수 있다.
학습 프로세스가 없거나 실패·장시간 진전 없음이면 완료 시각을 비운다.
상태 JSON이 오래된 경우 `launch.log`의 더 최신 상태 이벤트를 사용한다.
Pause로 이 관측기도 종료되면 다시 실행해야 한다.

## 코드 출처

- `app/openpi_snapshot`: 기존 실험에서 고정한 공개 OpenPI 코드. 원 라이선스 포함.
- 새 변경: False 데이터 설정·통계, 로컬 tokenizer 사용, raw HDF5 경로에서는
  사용하지 않는 LeRobot import 지연, Docker·payload 이전 도구.
- 공개 학습 코드: https://github.com/Physical-Intelligence/openpi
- 요청된 비교 논문: https://arxiv.org/abs/2603.22078
