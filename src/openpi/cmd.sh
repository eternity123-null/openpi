source /inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/openpi/.venv/bin/activate

uv run scripts/compute_norm_stats.py --config-name pi0_autolife

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_autolife --exp-name=scoop0825 --overwrite