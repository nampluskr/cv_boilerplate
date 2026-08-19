# Wired by the task agent (PLAN-P4 SS4.3); model agents only touch their own model file plus
# their own config file, so this import list is set up ahead of time to match the exact
# registry keys/file paths PLAN-P4 SS4.3 assigns to each model agent.
from src.tasks.detection.models import custom_fcos, fasterrcnn_r50_fpn, yolov8n  # noqa: F401
