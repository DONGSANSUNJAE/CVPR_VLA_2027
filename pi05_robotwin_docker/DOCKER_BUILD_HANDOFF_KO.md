# 다른 서버 담당자에게 전달할 Docker 이미지 빌드 요청

요청: 아래 GitHub의 RoboTwin π0.5 학습 환경을 **linux/amd64 Docker 이미지**로
빌드하고, CPU 검사 후 `docker save` 이미지 파일과 검증 기록을 전달해 주세요.
빌드 서버에서 학습을 시작할 필요는 없습니다. **빌드에는 GPU·학습 데이터·
초기 가중치·개인키·Hub 토큰이 필요하지 않습니다.**

저장소: https://github.com/DONGSANSUNJAE/CVPR_VLA_2027

작업 폴더: `pi05_robotwin_docker/`

이미지 태그: `pi05-robotwin-false:60k`

## 1. 빌드 서버 준비 조건

- Linux x86_64와 실제 동작하는 Docker Engine, Docker Buildx, Git, Python 3,
  `sha256sum`. `docker info`가 성공해야 합니다. Docker CLI만 있으면 안 됩니다.
- 인터넷: Docker Hub, Debian 패키지 저장소, PyPI 패키지 서버,
  `download.pytorch.org`, GitHub에 접근 가능해야 합니다.
- 여유 디스크 100GB, RAM 16GB 정도를 준비하는 것을 권장합니다.
  이는 빌드용 여유 권장치이며 실제 최대 사용량이나 이미지 크기의 실측값은 아닙니다.
- NVIDIA GPU/드라이버/Container Toolkit은 **빌드 서버에는 필요하지 않습니다.**
  GPU 학습을 실행할 최종 서버에 필요합니다.
- 기존 Python/PyTorch/CUDA 설치는 이미지에 사용하지 않습니다. Dockerfile이
  Python 3.11.11 기반 이미지를 사용하고 `requirements.lock`을 설치합니다.

Docker가 없다면 서버 OS에 맞게 설치하고 Buildx 사용을 확인해 주세요.
[Docker 설치](https://docs.docker.com/engine/install/),
[Buildx 설치](https://docs.docker.com/build/install-buildx/).

## 2. 빌드·CPU 검사·이미지 내보내기

새 clone을 사용합니다. 빌드 도구는 수정된 작업 폴더를 거절하며, 사용한 Git
commit을 이미지 라벨과 전달 파일에 기록합니다.

```bash
git clone https://github.com/DONGSANSUNJAE/CVPR_VLA_2027.git
cd CVPR_VLA_2027/pi05_robotwin_docker

docker info
docker buildx version

bash tools/build_export_image.sh
```

성공하면 `IMAGE_EXPORT_COMPLETE`가 출력되고 `build/image_export/`에 아래 파일이 생깁니다.
**이 디렉터리 전체를 전달**해 주세요.

```text
pi05-robotwin-linux-amd64.tar    Docker 이미지 자체
SHA256SUMS                     이미지와 기록 파일의 SHA-256
image_manifest.json            이미지 ID, 소스 commit, 검사 범위
image_inspect.json              Docker 이미지 메타데이터
source_commit.txt               정확한 소스 commit
docker_version.txt             빌드 서버의 Docker 버전
build.log                      빌드 로그
cpu_smoke.json                 CPU 검사 결과
cpu_smoke.stderr.log           CPU 검사 진단 출력
```

중간 실패 시 로그를 보존하고 오류를 전달해 주세요. 다시 시도할 때에는 새 출력
폴더를 지정할 수 있습니다: `bash tools/build_export_image.sh build/image_export_retry`.
실패를 없애려고 패키지 버전·학습 설정·체크섬 검사를 변경하지 마세요.

CPU 검사는 JAX·Flax·Torch·OpenCV·HDF5 import, 고정 버전, 학습 설정과 False 정규화
통계를 확인합니다. 네트워크·GPU·데이터·가중치를 사용하지 않습니다.
**CPU 검사 성공은 GPU 학습 성공이나 논문 성능 재현을 뜻하지 않습니다.**

빌드 도구의 핵심 명령은 `docker buildx build --platform linux/amd64 --load`,
`docker run --entrypoint ...` CPU 검사, `docker image save`입니다. 실행 중인
컨테이너 파일시스템만 저장하는 `docker export`로 대체하지 마세요.
[Buildx](https://docs.docker.com/reference/cli/docker/buildx/build/),
[이미지 save](https://docs.docker.com/reference/cli/docker/image/save/).

## 3. 변경하지 않을 환경·학습 설정

| 항목 | 고정값 | 확인 파일 |
|---|---|---|
| 플랫폼 | linux/amd64 | 빌드 스크립트 |
| Python | 3.11.11 | `Dockerfile` |
| JAX / jaxlib | 0.6.0 / 0.6.0 | `requirements.in`, `requirements.lock` |
| CUDA runtime | 12.8.90 | `requirements.in`, `requirements.lock` |
| Flax / NumPy | 0.10.2 / 1.26.4 | `requirements.in`, `requirements.lock` |
| Torch | 2.6.0+cpu; 데이터 처리 등에 사용, 학습 엔진은 JAX | `requirements.in` |
| 모델·초기화 | π0.5 full FT, 공식 pi05_base, bfloat16 | `app/configs/train_config.py` |
| 전처리 | **adapt_to_pi=False**, 재계산한 정규화 | 같은 설정 파일, `app/assets/` |
| 학습 데이터 | 50과제 × (clean 50 + randomized 500) = 27,500 | 별도 payload |
| 행동 | 팔 관절 delta, 그리퍼 absolute, horizon 50 | 설정 파일 |
| 배치·업데이트 | 64, optimizer 업데이트 60,000회 | 설정 파일 |
| 배치 구현 | microbatch 4 × 누적 16회, GPU 4장 FSDP | `docker/entrypoint.py` |
| AdamW | β1=.9, β2=.95, eps=1e-8, wd=1e-10, clip=1.0 | 설정 파일 |
| 학습률 | warmup 1,000, cosine 2.5e-5 → 2.5e-6, decay 60,000 | 설정 파일 |
| EMA / seed | .99 / 42 | 설정 파일 |
| 저장 | step 10, 1,000마다, 최종·정상 종료 시 | 설정 파일, 실행기 |
| 보존 | 최신 체크포인트 + 5,000의 배수 | 설정 파일 |

논문 https://arxiv.org/abs/2603.22078 의 3.2절에 공개된 주요 학습 조건에 맞춥니다.
논문에 명시되지 않은 adapter, warmup, horizon, EMA 등의 값과 분산 구현이
저자 구현과 완전히 같다고 주장하지 않습니다. 기존 True 학습 체크포인트는
이 False 실험의 resume에 사용하지 않습니다.

## 4. A100 서버로 따로 전송할 데이터

이미지 빌드 담당자에게 대용량 데이터를 보낼 필요는 없습니다. 원본 서버에서
**최종 A100 서버로** 아래 66개 파일을 따로 전송합니다.

```text
/scratch2/mailab_yonsei/HW/cvpr2027/pi05_robotwin_docker_false_60k/archives/
  payload-000.zip ... payload-064.zip
  payload_manifest.json
```

ZIP 합계 393.8GiB, 복원 후 약514.4GiB입니다. clean+random 학습 원자료,
지시문·인덱스, 공식 pi05_base, tokenizer가 포함돼 있습니다. 코드와 정규화는
GitHub/이미지에 포함됩니다. 기존 fine-tuned 체크포인트는 필요 없습니다.

## 5. 최종 A100 서버에서 이미지 불러오기·학습 시작

요구 조건: 한 Linux x86_64 서버의 **A100 80GB 4장**, 호환 NVIDIA 드라이버,
Docker Engine, Compose v2.30 이상, NVIDIA Container Toolkit. 이미지의 CUDA
라이브러리가 호스트 드라이버 설치를 대신하지 않습니다.
[NVIDIA Container Toolkit 설치](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

학습용 CPU RAM은160GB 이상, 압축본·데이터·중간 체크포인트를 함께 유지할
여유 디스크는2TB 이상을 권장합니다. 실제 GPU 학습 메모리·속도는 아직 실측하지
않았습니다. 서버가 임대 컨테이너라면 Docker 실행/이미지 사용 방식을 제공자가
지원하는지 확인해야 합니다.

아래에서는 이미지 전달 폴더를 `/data/pi05_image`, 데이터 ZIP 폴더를
`/data/pi05_archives`, 코드 위치를 `/data/CVPR_VLA_2027`로 가정합니다.
실제 쓰기 가능한 대용량 디스크 경로로 바꿔 사용하세요.

```bash
# 전달받은 이미지 및 기록 검증 후 Docker에 등록
cd /data/pi05_image
sha256sum -c SHA256SUMS
docker image load --input pi05-robotwin-linux-amd64.tar
docker image inspect pi05-robotwin-false:60k --format '{{.Id}}'
# 위 ID를 image_manifest.json의 image_id와 대조

# 빌드 때와 동일한 소스 commit 사용
cd /data
git clone https://github.com/DONGSANSUNJAE/CVPR_VLA_2027.git
cd CVPR_VLA_2027
PI05_CODE_COMMIT="$(cat /data/pi05_image/source_commit.txt)"
git checkout --detach "$PI05_CODE_COMMIT"
cd pi05_robotwin_docker

# 데이터·초기 가중치·tokenizer를 해시 검증하며 복원
python3 tools/restore_payload.py --archives /data/pi05_archives --output payload
python3 tools/check_payload.py

# 실제 데이터 준비 검사와 A100 4장의 기초 GPU 검사
docker compose run --rm --no-deps --pull never train preflight
docker compose run --rm --no-deps --pull never train gpu-check

# 받은 이미지를 그대로 사용하며 백그라운드 학습 시작
docker compose up -d --no-build --pull never train
docker compose logs -f --tail 50 train
```

이미지를 받은 서버에서는 빌드를 다시 실행하는 `tools/start.sh` 대신 위의
`up --no-build --pull never` 명령을 사용합니다. 이미지 로드·ID 확인이 성공해야
하며, 실행 로그에서 실제 첫 optimizer 업데이트와 체크포인트 저장을 확인해야 합니다.

CPU 데이터 검사는 전체 인덱스·정규화·base 해시 및50과제 입력을 검사합니다.
GPU 기초 검사도 전체 모델의 첫 학습 업데이트를 대신하지 않습니다.

## 6. 출력·중지·재개

```bash
tail -f outputs/logs/metrics.jsonl
cat outputs/logs/train_status.json
docker compose stop -t 1800 train
docker compose up -d --no-build --pull never train
```

데이터와 가중치는 read-only bind mount, 출력은 호스트 `outputs/`에 저장합니다.
체크포인트는 `outputs/checkpoints/` 아래에 생깁니다. 정상 종료 요청에서는
현재 업데이트 후 저장하며, 강제 종료 시 마지막 저장 완료분까지만 복원됩니다.
resume은 이 False 실험의 동일 설정 출력으로만 합니다.

이 패키지는 학습용입니다. 기존 실제 RoboTwin 시뮬레이터 영상 평가 환경·자산은
포함하지 않았습니다. 수치 재현 스크립트를 영상 평가 완료로 해석하지 않습니다.

## 준비 측 검증 상태

기존 CPU 데이터·의존성·새 clone 검사와 Compose 설정 검사는 통과했습니다.
전달용 셸 스크립트 문법 검사와 새 CPU 검사 스크립트의 별도 Python3.11.13
고정 의존성 환경 실행도 통과했습니다. 이것은 컨테이너 실행 결과가 아닙니다.
이 안내서 추가 시점에는 **실제 Docker 이미지 빌드/실행과 A100 학습은 미검증**입니다.
새 빌드·CPU 검사 도구의 성공 출력은 빌드 담당자 실행 후 확인해야 합니다.
버전 고정과 별도로 Debian apt 패키지는 날짜별 저장소 snapshot으로 고정하지
않았으므로 재빌드 이미지가 비트 단위로 같다고 보장하지 않습니다. 전달하는
이미지 자체는 SHA-256과 image ID로 식별합니다.
