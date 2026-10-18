"""Check local prerequisites without downloading models or calling paid APIs.

Exit 1 means a known blocker; exit 0 only means these preliminary checks passed.
It does not certify successful inference or sufficient activation memory.
"""
import argparse
import importlib.util
import platform
import shutil
import subprocess
from pathlib import Path

from cvm.devices import select_text_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-device", default="auto")
    parser.add_argument("--generation-device", default="cuda:0")
    args = parser.parse_args(argv)
    blockers = []
    print(f"OS: {platform.platform()} | Python: {platform.python_version()}")
    print(f"Free disk at checkout: {shutil.disk_usage(Path.cwd()).free / 2**30:.1f} GiB")
    for executable in ("ffmpeg", "ffprobe"):
        found = shutil.which(executable)
        print(f"{executable}: {found or 'MISSING'}")
        if not found:
            blockers.append(f"Install {executable} and put it on PATH.")
    if shutil.which("ffmpeg"):
        result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                capture_output=True, text=True, check=False)
        if result.returncode or "libx264" not in result.stdout:
            blockers.append("FFmpeg must include the libx264 encoder.")
    for package in ("diffusers", "accelerate", "transformers", "av", "PIL", "playwright"):
        if importlib.util.find_spec(package) is None:
            blockers.append(f"Missing Python module: {package}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__} | CUDA runtime: {torch.version.cuda}")
        if not torch.cuda.is_available():
            blockers.append("PyTorch cannot access CUDA; install the CUDA wheel and check the driver.")
        else:
            count = torch.cuda.device_count()
            text = select_text_device(args.text_device, args.generation_device, count)
            # Conservative weight-only lower bounds from the pinned HF snapshot.
            # Auto CPU offload moves entire components, not individual layers.
            for label, value, weight_gib in (("text encoder", text, 62.14),
                                             ("video transformer", args.generation_device, 61.73)):
                device = torch.device(value)
                index = device.index if device.index is not None else 0
                if device.type != "cuda" or not 0 <= index < count:
                    blockers.append(f"{label}: {value} is not an available CUDA device.")
                    continue
                props = torch.cuda.get_device_properties(index)
                gib = props.total_memory / 2**30
                print(f"{label}: cuda:{index}, {props.name}, {gib:.2f} GiB VRAM")
                if gib < weight_gib:
                    blockers.append(f"{label} has about {weight_gib:.2f} GiB of weights; "
                                    f"whole-model offload cannot fit it in {gib:.2f} GiB VRAM. "
                                    "Layer/group offload or quantization is required (not implemented here).")
                # Exercise kernels, rather than relying only on a driver listing.
                with torch.cuda.device(index):
                    x = torch.ones((16, 16), device=f"cuda:{index}", dtype=torch.bfloat16)
                    assert (x @ x).float().mean().item() == 16
                    torch.cuda.synchronize()
            print("CUDA kernel checks completed for valid selected devices.")
    except (ImportError, OSError, RuntimeError, ValueError, AssertionError) as exc:
        blockers.append(f"PyTorch/CUDA check failed: {exc}")
    print("Checkpoint download is approximately 196 GiB; not downloaded by this check.")
    print("Optional stages also need authenticated Copilot CLI and OPENROUTER_API_KEY.")
    for problem in blockers:
        print(f"BLOCKED: {problem}")
    if not blockers:
        print("Preliminary checks passed; model loading and full inference are still unverified.")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
