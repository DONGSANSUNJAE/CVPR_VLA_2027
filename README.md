# CVPR_VLA_2027
CVPR_VLA_2027

## RoboTwin π0.5 full fine-tuning

[A100 80GB × 4 Docker setup](pi05_robotwin_docker/README_KO.md): clean + randomized 27,500 episodes, adapt_to_pi=False, batch64, 60k updates.

Code and normalization are in this repository. Data and official base weights are separate verified ZIP archives; no trained checkpoints or credentials are included.

[Validation scope](pi05_robotwin_docker/VALIDATION_KO.md): CPU and Compose checks passed; Docker build and A100 execution still require validation on the destination server.

[Docker image build handoff](pi05_robotwin_docker/DOCKER_BUILD_HANDOFF_KO.md): build on a CPU-only Docker host and transfer the verified image to the A100 server.
