#!/usr/bin/env python3
import argparse
import io
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def apply_cpu_affinity():
    affinity = os.environ.get("NEUTTS_CPU_AFFINITY", "").strip()
    if not affinity:
        return None

    cpus = set()
    for part in affinity.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))

    os.sched_setaffinity(0, cpus)
    return sorted(cpus)


CPU_AFFINITY = apply_cpu_affinity()

if os.environ.get("NEUTTS_LLAMA_CPP_LIB_PATH"):
    os.environ.setdefault("LLAMA_CPP_LIB_PATH", os.environ["NEUTTS_LLAMA_CPP_LIB_PATH"])

# llama.cpp wheels can accidentally load incompatible libggml/libllama builds from
# LD_LIBRARY_PATH. Keep the process clean unless the caller explicitly opts out.
if (
    os.environ.get("LD_LIBRARY_PATH")
    and os.environ.get("NEUTTS_KEEP_LIBPATH") != "1"
    and os.environ.get("NEUTTS_CLEAN_LIBPATH") != "1"
):
    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    clean_env["NEUTTS_CLEAN_LIBPATH"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], clean_env)

import soundfile as sf
import torch

from neutts import NeuTTS


SAMPLE_RATE = int(os.environ.get("NEUTTS_SAMPLE_RATE", "24000"))
DEFAULT_BACKBONE = os.environ.get("NEUTTS_BACKBONE", "neuphonic/neutts-nano-q4-gguf")
DEFAULT_CODEC = os.environ.get("NEUTTS_CODEC", "neuphonic/neucodec-onnx-decoder")
DEFAULT_VOICE = os.environ.get("NEUTTS_DEFAULT_VOICE", "jo")


def env_int(name: str):
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else None


def env_bool(name: str):
    value = os.environ.get(name)
    if value in {None, ""}:
        return None
    return value.lower() in {"1", "true", "yes", "on"}


def configure_llama_cpp_defaults():
    try:
        import llama_cpp
    except Exception:
        return {}

    if getattr(llama_cpp, "_neutts_tuned", False):
        return getattr(llama_cpp, "_neutts_tuned_options", {})

    options = {
        "n_threads": env_int("NEUTTS_LLAMA_THREADS"),
        "n_threads_batch": env_int("NEUTTS_LLAMA_THREADS_BATCH"),
        "n_batch": env_int("NEUTTS_LLAMA_BATCH"),
        "n_ubatch": env_int("NEUTTS_LLAMA_UBATCH"),
        "flash_attn": env_bool("NEUTTS_LLAMA_FLASH_ATTN"),
        "offload_kqv": env_bool("NEUTTS_LLAMA_OFFLOAD_KQV"),
    }
    options = {key: value for key, value in options.items() if value is not None}
    if not options:
        llama_cpp._neutts_tuned = True
        llama_cpp._neutts_tuned_options = {}
        return {}

    original_llama = llama_cpp.Llama

    class TunedLlama(original_llama):
        def __init__(self, *args, **kwargs):
            kwargs.update(options)
            super().__init__(*args, **kwargs)

    TunedLlama.__name__ = original_llama.__name__
    TunedLlama.__qualname__ = original_llama.__qualname__
    TunedLlama.__module__ = original_llama.__module__
    llama_cpp.Llama = TunedLlama
    llama_cpp._neutts_tuned = True
    llama_cpp._neutts_tuned_options = options
    return options


def default_voices_dir() -> Path:
    candidates = [
        os.environ.get("NEUTTS_VOICES_DIR"),
        str(Path.cwd() / "samples"),
        "/home/op/neutts/samples",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return Path(candidates[1])


VOICES_DIR = default_voices_dir()


def available_voices():
    return sorted(
        path.stem
        for path in VOICES_DIR.glob("*.pt")
        if (VOICES_DIR / f"{path.stem}.txt").exists()
    )


def llama_gpu_offload_status():
    try:
        from llama_cpp import llama_cpp as lc

        fn = getattr(lc, "llama_supports_gpu_offload", None)
        return bool(fn()) if fn else None
    except Exception:
        return None


class TTSRuntime:
    def __init__(self, backbone_device: str, codec_device: str):
        started = time.perf_counter()
        self.backbone_device = backbone_device
        self.codec_device = codec_device
        self.llama_options = configure_llama_cpp_defaults()
        self.gpu_offload_supported_before_load = llama_gpu_offload_status()
        self.tts = NeuTTS(
            backbone_repo=DEFAULT_BACKBONE,
            backbone_device=backbone_device,
            codec_repo=DEFAULT_CODEC,
            codec_device=codec_device,
        )
        self.load_seconds = time.perf_counter() - started
        self.gpu_offload_supported = llama_gpu_offload_status()
        self.lock = threading.Lock()
        self.refs = {}

    def ref(self, voice: str):
        voice = voice or DEFAULT_VOICE
        if voice in self.refs:
            return self.refs[voice]

        codes_path = VOICES_DIR / f"{voice}.pt"
        text_path = VOICES_DIR / f"{voice}.txt"
        if not codes_path.exists() or not text_path.exists():
            voices = available_voices()
            raise ValueError(
                f"Unknown voice '{voice}'. Available voices: {', '.join(voices) or 'none'}"
            )

        codes = torch.load(codes_path, map_location="cpu")
        if hasattr(codes, "detach"):
            codes = codes.detach().cpu().numpy().astype(int).tolist()
        ref_text = text_path.read_text().strip()
        self.refs[voice] = (codes, ref_text)
        return self.refs[voice]

    def synthesize(self, text: str, voice: str = DEFAULT_VOICE):
        text = (text or "").strip()
        if not text:
            raise ValueError("Missing text")

        ref_codes, ref_text = self.ref(voice)
        started = time.perf_counter()
        with self.lock:
            wav = self.tts.infer(text, ref_codes, ref_text)
        elapsed = time.perf_counter() - started
        duration = len(wav) / SAMPLE_RATE
        rtf = elapsed / duration if duration else 0.0

        buf = io.BytesIO()
        sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue(), {
            "elapsed_seconds": elapsed,
            "audio_seconds": duration,
            "rtf": rtf,
        }


runtime = None


class Handler(BaseHTTPRequestHandler):
    server_version = "NeuTTSOpenAI/1.0"

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_json(self, status: int, payload: dict):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/", "/health"}:
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "neutts-openai",
                    "model": DEFAULT_BACKBONE,
                    "codec": DEFAULT_CODEC,
                    "backbone_device": runtime.backbone_device,
                    "codec_device": runtime.codec_device,
                    "llama_cpp_lib_path": os.environ.get("LLAMA_CPP_LIB_PATH"),
                    "llama_gpu_offload_supported": runtime.gpu_offload_supported,
                    "cpu_affinity": CPU_AFFINITY,
                    "llama_options": runtime.llama_options,
                    "load_seconds": runtime.load_seconds,
                    "voices_dir": str(VOICES_DIR),
                    "voices": available_voices(),
                },
            )
            return

        if parsed.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "tts-1",
                            "object": "model",
                            "created": 1704067200,
                            "owned_by": "neuphonic",
                        },
                        {
                            "id": DEFAULT_BACKBONE,
                            "object": "model",
                            "created": 1704067200,
                            "owned_by": "neuphonic",
                        },
                    ],
                },
            )
            return

        if parsed.path == "/v1/audio/voices":
            self.send_json(
                200,
                {
                    "voices": [
                        {
                            "name": voice,
                            "language": "en",
                            "description": f"NeuTTS reference voice: {voice}",
                        }
                        for voice in available_voices()
                    ]
                },
            )
            return

        if parsed.path == "/tts":
            query = urllib.parse.parse_qs(parsed.query)
            text = query.get("text", [""])[0]
            voice = query.get("voice", [DEFAULT_VOICE])[0]
            self._send_tts(text, voice)
            return

        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in {"/tts", "/v1/audio/speech"}:
            self.send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self.send_json(400, {"error": f"invalid json: {exc}"})
            return

        text = payload.get("text") or payload.get("input") or ""
        voice = payload.get("voice") or DEFAULT_VOICE
        response_format = (payload.get("response_format") or "wav").lower()
        if response_format not in {"wav", "pcm"}:
            self.send_json(
                400,
                {
                    "error": (
                        "NeuTTS returns wav/pcm only. Requested "
                        f"response_format={response_format!r}."
                    )
                },
            )
            return
        self._send_tts(text, voice)

    def _send_tts(self, text: str, voice: str):
        try:
            wav_bytes, metrics = runtime.synthesize(text, voice)
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("X-NeuTTS-Backbone-Device", runtime.backbone_device)
        self.send_header("X-NeuTTS-Codec-Device", runtime.codec_device)
        self.send_header("X-NeuTTS-Elapsed-Seconds", f"{metrics['elapsed_seconds']:.3f}")
        self.send_header("X-NeuTTS-Audio-Seconds", f"{metrics['audio_seconds']:.3f}")
        self.send_header("X-NeuTTS-RTF", f"{metrics['rtf']:.3f}")
        self.end_headers()
        self.wfile.write(wav_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("NEUTTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NEUTTS_PORT", "12435")))
    parser.add_argument("--backbone-device", default=os.environ.get("NEUTTS_BACKBONE_DEVICE", "gpu"))
    parser.add_argument("--codec-device", default=os.environ.get("NEUTTS_CODEC_DEVICE", "cpu"))
    args = parser.parse_args()

    global runtime
    runtime = TTSRuntime(args.backbone_device, args.codec_device)
    print(
        "NeuTTS OpenAI service ready on "
        f"http://{args.host}:{args.port} "
        f"(backbone={args.backbone_device}, codec={args.codec_device}, "
        f"gpu_offload={runtime.gpu_offload_supported}, "
        f"loaded in {runtime.load_seconds:.1f}s)",
        flush=True,
    )

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
