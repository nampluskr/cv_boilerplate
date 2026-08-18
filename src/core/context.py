import os
import platform
import random
import socket
import subprocess
import time

import numpy as np
import torch
import torchmetrics
import torchvision


class RunContext:
    """Holds runtime settings (seed, device, AMP, determinism) and per-run state."""

    def __init__(self, config, run_dir, allow_test_split=False):
        self.config = config
        self.run_dir = run_dir
        self.seed = config["runtime"]["seed"]
        self.device_name = config["runtime"]["device"]
        self.amp = config["runtime"]["amp"]
        self.deterministic = config["runtime"]["deterministic"]
        self.epochs = config["train"]["epochs"]
        self.grad_clip = config["train"]["grad_clip"]
        self.allow_test_split = allow_test_split
        self.nondeterministic_warnings = []
        self.started_at = None
        self.finished_at = None
        self.command_line = None

    @property
    def device(self):
        if self.device_name == "cuda":
            return torch.device("cuda:0")
        return torch.device("cpu")

    def setup_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.apply_determinism()

    def apply_determinism(self):
        mode = self.deterministic
        if mode == "strict":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        elif mode == "warn":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        elif mode == "off":
            torch.backends.cudnn.benchmark = True
        else:
            raise ValueError(f"Unknown runtime.deterministic value: {mode}")

    def start(self, command_line):
        self.command_line = command_line
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def finish(self):
        self.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def env_info(self):
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        driver_version = None
        try:
            driver_version = torch.cuda.driver_version if hasattr(torch.cuda, "driver_version") else None
        except Exception:
            driver_version = None
        return {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "torchmetrics_version": torchmetrics.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": gpu_name,
            "driver_version": driver_version,
            "hostname": socket.gethostname(),
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
            "command_line": self.command_line,
            "seed": self.seed,
            "deterministic_mode": self.deterministic,
            "amp": self.amp,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "nondeterministic_warnings": self.nondeterministic_warnings,
        }


def make_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def get_git_dirty():
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(output)
    except Exception:
        return None
