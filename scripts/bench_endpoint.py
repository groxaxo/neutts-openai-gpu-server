#!/usr/bin/env python3
import argparse
import json
import tempfile
import time
import urllib.request
import wave
from pathlib import Path


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:12435/v1/audio/speech")
    parser.add_argument("--voice", default="jo")
    parser.add_argument("--model", default="tts-1")
    parser.add_argument("--text", default="This is a short local speed test.")
    args = parser.parse_args()

    payload = json.dumps(
        {"model": args.model, "input": args.text, "voice": args.voice, "response_format": "wav"}
    ).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        body = response.read()
        headers = dict(response.headers.items())
    elapsed = time.perf_counter() - started

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        handle.write(body)
        tmp_path = Path(handle.name)

    duration = wav_duration(tmp_path)
    tmp_path.unlink(missing_ok=True)
    print(json.dumps(
        {
            "url": args.url,
            "elapsed_seconds": round(elapsed, 3),
            "audio_seconds": round(duration, 3),
            "rtf": round(elapsed / duration, 3) if duration else None,
            "x_realtime": round(duration / elapsed, 3) if elapsed else None,
            "server_elapsed_seconds": headers.get("X-NeuTTS-Elapsed-Seconds"),
            "server_audio_seconds": headers.get("X-NeuTTS-Audio-Seconds"),
            "server_rtf": headers.get("X-NeuTTS-RTF"),
            "backbone_device": headers.get("X-NeuTTS-Backbone-Device"),
            "codec_device": headers.get("X-NeuTTS-Codec-Device"),
            "bytes": len(body),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
