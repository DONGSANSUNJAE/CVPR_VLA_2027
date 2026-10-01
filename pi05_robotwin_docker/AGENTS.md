# Portable RoboTwin pi0.5 run

User request 2026-10-01: prepare a GitHub-clonable Docker project for a different
server with four A100 80GB GPUs on one node. Full FT, clean+random 27,500 episodes,
batch64 and60k updates, MUST adapt_to_pi=False in training, normalization and
inference. Start from official pi05_base, never resume the old True-trained run.
Preserve the original server jobs and frozen sources. No external upload until
the user identifies the destination. Package code separately from data/weights.
Do not claim Docker/GPU validation if only CPU checks have run. This host has no
Docker engine. Never commit credentials, raw data, checkpoints, virtualenvs,
downloads, build caches or old run results to Git.
