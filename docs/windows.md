# Windows / RTX 5090 assessment

## Verdict

The repository cannot run its full H3 inference pipeline unchanged on a single
32 GB RTX 5090. Windows and Blackwell-compatible PyTorch wheels are available;
the decisive blocker is the backend's whole-model CPU offloading. Installing
dependencies or using WSL2 does not solve that memory limitation.

This branch fixes automatic device selection and adds a preliminary checker.
It does **not** implement or validate a low-VRAM inference backend. No full
model download or generated-video test was performed during this assessment.

## Machine inspected on 2026-10-16

| Resource | Observed |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB, Windows WDDM |
| Driver | 610.88; nvidia-smi reports CUDA driver capability 13.3 |
| Physical RAM | 255.66 GiB, about 237.65 GiB available during inspection |
| D: free space | About 2,806 GiB |
| Python on PATH | 3.12.0, with pip only |
| Missing on PATH | ffmpeg, ffprobe, Copilot CLI |

The CUDA version displayed by nvidia-smi is driver capability, not an installed
PyTorch runtime. No separate CUDA toolkit is needed for prebuilt PyTorch wheels.

## What blocks generation

1. Upstream installs PyTorch 2.7.1 with CUDA 12.6. Use its CUDA **12.8** build for
   Blackwell. PyTorch documents both Windows wheels and Blackwell support.
2. Upstream defaults to `cuda:1` for the text encoder. This branch uses `auto`:
   `cuda:1` when multiple GPUs are visible, otherwise the generation device.
   Explicit device arguments still override this behavior.
3. `H3.load()` in `cvm/video.py` uses `ComponentsManager.enable_auto_cpu_offload()`.
   The pinned Diffusers implementation moves an entire model to the GPU before
   its forward pass. Reserving 16 GB for generation activations does not split
   model weights across CPU and GPU.
4. The pinned checkpoint contains the following on-disk component sizes. These
   are weight-storage measurements, **not measured peak RAM/VRAM requirements**.

| Component | GiB |
| --- | ---: |
| text_encoder | 62.14 |
| transformer_ref | 61.73 |
| transformer | 61.73 |
| vae | 9.70 |
| audio_vae | 0.56 |
| Total including small configs/tokenizer files | About 196 |

Sizes were queried from Hugging Face file metadata at the manifest revision
`42ed227ee7df40d41602854ae760620d6eb651fe`. The downloader includes both
transformers even though video generation uses `transformer_ref`. We preserve
the existing download manifest and its integrity checks.

The machine has substantial host RAM and sufficient disk for that download.
This makes a layer/group-offloaded implementation a plausible next engineering
step, but does not establish that loading peaks and activations will fit. Both
the Qwen text encoder and H3 transformer need finer-grained placement; VAE
encoding/decoding and the two component managers sharing one GPU also need
validation. Quantization is another option but changes numerical behavior.
An alternative H3 application is not automatically a replacement for this
repository's custom early-proxy conditioning and 20-step sampling algorithm.

## Prepare an isolated environment (PowerShell)

These commands prepare dependencies for investigation. They do not fix the
memory blocker. Run from the repository root; keep work on `try-windows-deploy`.
Python 3.12 is present locally; upstream documents Python 3.10. The dependency
combination below has not been installed or exercised on this machine.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m playwright install chromium
```

Install a Windows FFmpeg 6+ GPL build that contains both `ffmpeg.exe` and
`ffprobe.exe`, and add its `bin` directory to the current shell's PATH. Confirm:

```powershell
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders | Select-String libx264
.\.venv\Scripts\python.exe -m inference.check_environment
```

The checker exits 1 for known blockers and does not download anything. Even a
successful check is not a full inference certification. On this 5090 it should
continue to report the whole-model memory blocker after dependencies are fixed.

## Stages that can be investigated separately

The supplied Three.js scene viewer does not need H3 weights or GPU inference:

```powershell
python -m inference.viewer --scene examples/chair_placement/original --port 8000
```

Rendering supplied scenes additionally needs Pillow, Playwright Chromium and
FFmpeg. Generate a new output copy before rendering if preserving example files
is important, because the renderer writes into the scene directory.

Text-to-Three.js generation invokes **GitHub Copilot CLI** (`copilot`) using
`gpt-6-astra`, requiring its own authentication/model access. Reference image
generation and evaluation call OpenRouter and require `OPENROUTER_API_KEY`.
These stages use external services; installing Python dependencies is not their
complete setup. Bundled proxy/reference/prompt examples avoid these API stages.

Do not download the approximately 196 GiB checkpoint just to test installation.
After implementing and validating finer-grained offloading, the existing
`download_h3.py` and `python -m inference.generate_video` commands remain the
entry points. Keep the original 960x544, 124-frame contract for validation.

## Sources

- [PyTorch 2.7 Blackwell support](https://pytorch.org/blog/pytorch-2-7/)
- [Official PyTorch 2.7.1 Windows/CUDA wheel commands](https://pytorch.org/get-started/previous-versions/#v271)
- [Pinned Diffusers component manager](https://github.com/huggingface/diffusers/blob/90b4e34e79a86ec5e7f2437634fe95ecd2108796/src/diffusers/modular_pipelines/components_manager.py)
- [Pinned MiniMax-H3 checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe)
