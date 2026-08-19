import os
import socket
import traceback

from src.core.errors import LocalAssetError, OfflineViolationError

LOOPBACK_ADDRESSES = {"127.0.0.1", "::1"}

original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
original_getaddrinfo = socket.getaddrinfo
original_sendto = socket.socket.sendto
original_sendmsg = socket.socket.sendmsg
guard_enabled = False


def is_loopback(host):
    return host in LOOPBACK_ADDRESSES or host == "localhost"


def blocked(address):
    host = address[0] if isinstance(address, tuple) else address
    if is_loopback(host):
        return False
    stack = "".join(traceback.format_stack()[:-2])
    raise OfflineViolationError(
        f"Network connection to {address} is blocked by the offline guard.\n"
        f"Call stack:\n{stack}"
    )


def guarded_connect(self, address):
    if self.family == socket.AF_UNIX:
        return original_connect(self, address)
    blocked(address)
    return original_connect(self, address)


def guarded_connect_ex(self, address):
    if self.family == socket.AF_UNIX:
        return original_connect_ex(self, address)
    blocked(address)
    return original_connect_ex(self, address)


def guarded_sendto(self, *args, **kwargs):
    if self.family == socket.AF_UNIX:
        return original_sendto(self, *args, **kwargs)
    address = kwargs.get("address", args[-1] if args else None)
    if address is not None:
        blocked(address)
    return original_sendto(self, *args, **kwargs)


def guarded_sendmsg(self, *args, **kwargs):
    if self.family == socket.AF_UNIX:
        return original_sendmsg(self, *args, **kwargs)
    address = kwargs.get("address", args[3] if len(args) >= 4 else None)
    if address is not None:
        blocked(address)
    return original_sendmsg(self, *args, **kwargs)


def guarded_getaddrinfo(host, *args, **kwargs):
    if host is not None and not is_loopback(host):
        stack = "".join(traceback.format_stack()[:-1])
        raise OfflineViolationError(
            f"DNS resolution for '{host}' is blocked by the offline guard.\nCall stack:\n{stack}"
        )
    return original_getaddrinfo(host, *args, **kwargs)


def set_offline_env_vars():
    """Pure os.environ writes with no other imports, so this is safe to call before any
    third-party library (torch, torchvision, ultralytics, huggingface_hub) is imported."""
    os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "/mnt/d/backbones/.torch_home")
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/d/backbones/.hf_home")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["YOLO_OFFLINE"] = "1"  # purity-allow: PLAN-P1 SS9.1 offline env var, not model branching


def enable_offline_guard():
    global guard_enabled
    set_offline_env_vars()
    if not guard_enabled:
        socket.socket.connect = guarded_connect
        socket.socket.connect_ex = guarded_connect_ex
        socket.socket.sendto = guarded_sendto
        socket.socket.sendmsg = guarded_sendmsg
        socket.getaddrinfo = guarded_getaddrinfo
        guard_enabled = True


def disable_offline_guard():
    global guard_enabled
    socket.socket.connect = original_connect
    socket.socket.connect_ex = original_connect_ex
    socket.socket.sendto = original_sendto
    socket.socket.sendmsg = original_sendmsg
    socket.getaddrinfo = original_getaddrinfo
    guard_enabled = False


def load_local_weights(model, weights_path, strict=True, map_location="cpu", key_map=None):
    """Load a local .pth/.pt checkpoint into model. Never downloads."""
    if weights_path is None or not os.path.isfile(weights_path):
        raise LocalAssetError(
            f"Local weights file not found: {weights_path}\n"
            f"Place the corresponding .pth/.pt file under /mnt/d/backbones and set "
            f"model.params.weights_path to its path."
        )
    state_dict = torch_load(weights_path, map_location)
    if key_map is not None:
        state_dict = {key_map.get(key, key): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if strict and (missing or unexpected):
        raise LocalAssetError(
            f"Key mismatch loading {weights_path}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    if not strict and (missing or unexpected):
        print(
            f"[load_local_weights] non-strict load from {weights_path}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return model


def torch_load(weights_path, map_location):
    import torch

    checkpoint = torch.load(weights_path, map_location=map_location, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint
