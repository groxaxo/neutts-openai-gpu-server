# NeuTTS OpenAI GPU Server

Small OpenAI-compatible HTTP server for NeuTTS with optional llama.cpp GPU
offload. It exposes:

- `POST /v1/audio/speech`
- `GET /v1/models`
- `GET /v1/audio/voices`
- `GET /health`

The endpoint accepts OpenAI-style JSON:

```json
{
  "model": "tts-1",
  "input": "Hello from local NeuTTS",
  "voice": "jo",
  "response_format": "wav"
}
```

NeuTTS currently returns WAV audio from this server. The ONNX codec remains on
CPU; the GPU offload applies to the GGUF backbone through `llama-cpp-python`.

## Local quick start on this machine

This host already has:

- NeuTTS repo and venv: `/home/op/neutts`
- CUDA llama.cpp build: `/home/op/llama.cpp/build-cuda/bin`
- reference voices: `/home/op/neutts/samples`

Run the GPU service:

```bash
cd /home/op/neutts-openai-gpu-server
./scripts/launch_gpu_local.sh
```

Or run it in the background:

```bash
./scripts/start_gpu_daemon_local.sh
tail -f /tmp/neutts_gpu_server.log
```

Default URL:

```text
http://127.0.0.1:12435/v1/audio/speech
```

Smoke test:

```bash
curl -X POST http://127.0.0.1:12435/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data '{"model":"tts-1","input":"Hello world","voice":"jo","response_format":"wav"}' \
  --output speech.wav
```

Benchmark:

```bash
/home/op/neutts/.venv/bin/python scripts/bench_endpoint.py \
  --url http://127.0.0.1:12435/v1/audio/speech \
  --voice jo
```

## Reproduce on another Linux/NVIDIA machine

1. Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y git cmake ninja-build build-essential python3.11-venv \
  libsndfile1 ffmpeg espeak-ng
```

2. Clone NeuTTS and install it:

```bash
git clone https://github.com/neuphonic/neutts.git ~/neutts
cd ~/neutts
python3.11 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e '.[llama,onnx]'
```

3. Build llama.cpp with CUDA:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build-cuda \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build ~/llama.cpp/build-cuda --config Release -j"$(nproc)"
```

For old NVIDIA GPUs, use the matching architecture explicitly. For a GTX 750 Ti:

```bash
-DCMAKE_CUDA_ARCHITECTURES=50
```

4. Run this server:

```bash
git clone <this-repo-url> ~/neutts-openai-gpu-server
cd ~/neutts-openai-gpu-server

NEUTTS_ROOT=~/neutts \
NEUTTS_VENV=~/neutts/.venv \
NEUTTS_LLAMA_CPP_LIB_PATH=~/llama.cpp/build-cuda/bin \
NEUTTS_VOICES_DIR=~/neutts/samples \
NEUTTS_PORT=12435 \
./scripts/launch_gpu_local.sh
```

5. Confirm GPU offload:

```bash
curl -s http://127.0.0.1:12435/health | jq .
```

Look for:

```json
{
  "backbone_device": "gpu",
  "codec_device": "cpu",
  "llama_gpu_offload_supported": true,
  "cpu_affinity": [0, 1, 2, 3],
  "llama_options": {
    "n_threads": 4,
    "n_threads_batch": 4
  }
}
```

## CPU fallback

Run the same server on CPU:

```bash
./scripts/launch_cpu_local.sh
```

Default CPU fallback URL:

```text
http://127.0.0.1:12436/v1/audio/speech
```

## Useful environment variables

- `NEUTTS_HOST`: bind address, default `127.0.0.1`
- `NEUTTS_PORT`: port, default `12435`
- `NEUTTS_BACKBONE`: default `neuphonic/neutts-nano-q4-gguf`
- `NEUTTS_BACKBONE_DEVICE`: `gpu` or `cpu`
- `NEUTTS_CODEC`: default `neuphonic/neucodec-onnx-decoder`
- `NEUTTS_CODEC_DEVICE`: default `cpu`
- `NEUTTS_LLAMA_CPP_LIB_PATH`: path containing CUDA `libllama.so`
- `NEUTTS_VOICES_DIR`: directory with `<voice>.pt` and `<voice>.txt`
- `NEUTTS_DEFAULT_VOICE`: default `jo`
- `NEUTTS_CPU_AFFINITY`: Linux CPU affinity such as `0-3`
- `NEUTTS_LLAMA_THREADS`: forwarded to llama.cpp `n_threads`
- `NEUTTS_LLAMA_THREADS_BATCH`: forwarded to llama.cpp `n_threads_batch`
- `NEUTTS_LLAMA_BATCH`: forwarded to llama.cpp `n_batch`
- `NEUTTS_LLAMA_UBATCH`: forwarded to llama.cpp `n_ubatch`
- `NEUTTS_LLAMA_FLASH_ATTN`: force llama.cpp `flash_attn`, for example `0`
- `NEUTTS_LLAMA_OFFLOAD_KQV`: force llama.cpp `offload_kqv`

## Notes

- `LLAMA_CPP_LIB_PATH` must point at a llama.cpp build compatible with the
  installed `llama-cpp-python` package.
- If the process sees an incompatible `LD_LIBRARY_PATH`, the server clears it
  on startup by default. Set `NEUTTS_KEEP_LIBPATH=1` only if you know you need it.
- On small or older GPUs, GPU offload may reduce CPU load more than it improves
  wall-clock latency.
- On non-hybrid CPUs there are no P/E cores. A useful first tuning pass is to
  pin the server to one logical CPU per physical core, for example `0-3` on a
  4-core/8-thread i7-4790.
