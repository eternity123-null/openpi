source /inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/openpi/.venv/bin/activate

uv run scripts/compute_norm_stats.py --config-name pi0_autolife_pour_single_finetune

WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_autolife_scoop_single_finetune --exp-name=scoop_3_1931_ft
WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_autolife_pour_single_finetune --exp-name=pour_3_1931_ft
WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_autolife_scoop_labeled_finetune --exp-name=ft_scoop19-27_with_labeled
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_autolife_scoop_labeled_finetune --policy.dir=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/openpi/checkpoints/pi0_autolife_scoop_labeled_finetune/ft_scoop19-27_with_labeled_only/35000

uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_autolife_scoop_single_finetune --policy.dir=/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/openpi/checkpoints/pi0_autolife_scoop_single_finetune/scoop_1_1931_ft/49999

wandb sync wandb/offline-run-*-*

python /inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/openpi/.venv/lib64/python3.11/site-packages/lerobot/scripts/visualize_dataset_html.py \
 --repo-id pour_3_1931 \
 --root /inspire/ssd/project/robot-decision/public/zcd/autolife_lerobot/ \
 --force-override 1 \
 --local-files-only 1 \
 --tolerance-s 1

python /inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/openpi/.venv/lib64/python3.11/site-packages/lerobot/scripts/visualize_dataset.py \
--repo-id pour_3_1931 \
--root /inspire/ssd/project/robot-decision/public/zcd/autolife_lerobot/pour_3_1931 \
--episode-index 0