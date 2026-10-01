# 이전 패키지 검증 범위

2026-10-01, 준비 서버에서 확인했다.

| 검사 | 결과 |
|---|---|
| 전체 False 정규화 재계산 | 27,500에피소드·6,075,103시작점·303,755,150 action 벡터 완료 |
| 분리된 Python3.11 환경 | requirements.lock으로 97개 패키지 설치, uv pip check 통과 |
| 행동 전처리 | False 유지, raw state 보존, 팔 delta/그리퍼 absolute, 역변환 수치 검사 통과 |
| ZIP 복원 | 작은 fixture 압축→해시검사→복원→재실행 통과, 손상 ZIP 거절 확인 |
| 실제 데이터 입력 | 50과제를 포함한 52개 관측의 모델 입력 shape·유한값 검사 통과 |
| 데이터·초기 가중치 | index/qpos/False norm 해시와 공식 base의 29개 파일 해시 확인 |
| 별도 Git clone 경로 | 새 clone과 독립 venv에서도 실제 데이터 preflight 통과 |
| Docker Compose | 공식 Compose v2.39.2 바이너리의 config 검사 통과 |
| 전달용 빌드 도구 | 셸 문법 검사 통과; 새 CPU 검사 스크립트가 별도 Python3.11.13 고정 의존성 환경에서 통과. 실제 컨테이너에서의 실행은 아직 미검증 |
| Docker base | Python3.11.11 slim-bookworm의 linux/amd64 manifest digest 고정 |
| 원래 학습 | 원본 frozen source hash 불일치 0, 작업2350262 유지 |
| 실제 Docker build/run | **미실행: 준비 서버에 Docker 엔진 없음** |
| A100 4장 GPU 실행 | **미실행: 목적지 서버/할당에 접근하지 않음** |
| 전체 대용량 ZIP | 65개 생성 완료, 55,032개 원본 파일 SHA-256 기록, 약393.8GiB |
| 게시 대상 | GitHub DONGSANSUNJAE/CVPR_VLA_2027의 pi05_robotwin_docker/; 데이터 Hub 업로드는 미실행 |

CPU 검사는 Docker용 동일 lock의 독립 venv에서 수행했으며 기존 venv의 .pth나
사용자 uv cache로 연결된 환경을 그대로 이전한 것이 아니다. 준비 Python은
3.11.13, Docker base는3.11.11이다. 실제 컨테이너/드라이버 검증의 대체 결과는 아니다.

이전 코드의 LeRobot import를 필요 시점으로 옮겼다. raw HDF5 factory를 사용하는
이번 실행은 LeRobot 데이터 변환이나 외부 데이터셋을 로딩하지 않는다. native
OpenPI 모델·loss는 유지하며 FSDP/gradient accumulation은 기존 4 GPU 경로다.

수치 재현용 코드와 실제 시뮬레이터 영상 평가는 구분한다. 이 패키지에서는
새 A100 학습 모델의 성공률·영상 재현을 아직 측정하지 않았다.
