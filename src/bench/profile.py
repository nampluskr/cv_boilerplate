import time

import torch


def count_params(model):
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(params_total), int(params_trainable)


def measure_flops(model, adapter, image_size, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    model.eval()
    dummy = adapter.dummy_forward_input(image_size, device)
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as flop_counter:
            model(dummy)
        total_flops = flop_counter.get_total_flops()
        return total_flops / 1e9
    except Exception:
        return None


def measure_fps(model, adapter, image_size, device, warmup=10, iters=50):
    model.eval()
    dummy = adapter.dummy_forward_input(image_size, device)
    latencies = []
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
    latencies.sort()
    median_latency = latencies[len(latencies) // 2]
    if median_latency <= 0:
        return None
    return 1.0 / median_latency


def profile_model(model, adapter, image_size, device):
    params_total, params_trainable = count_params(model)
    flops_g = measure_flops(model, adapter, image_size, device)
    fps = measure_fps(model, adapter, image_size, device)
    return {
        "params_total": params_total,
        "params_trainable": params_trainable,
        "flops_g": flops_g,
        "fps": fps,
    }
