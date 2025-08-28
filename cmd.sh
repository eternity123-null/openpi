source /inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/openpi/.venv/bin/activate

uv run scripts/compute_norm_stats.py --config-name pi0_autolife_pour

WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_autolife_scoop --exp-name=scoop19-27
WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_autolife_pour --exp-name=pour19-27
WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_autolife_scoop_finetune --exp-name=ft_scoop19-25_on_19-27
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_autolife_scoop --policy.dir=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/openpi/checkpoints/pi0_autolife_scoop/scoop19-27/45000
