#!/usr/bin/env python3
import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
import tempfile
import time

import gradio as gr


def _normalize_base_url(url: str) -> str:
    base = (url or "").strip()
    if not base:
        return "http://127.0.0.1:12435"

    for suffix in ["/v1/audio/speech", "/v1/audio/voices", "/v1/models", "/health", "/"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    return base.rstrip("/")


def _api_url(base_url: str, path: str) -> str:
    base = _normalize_base_url(base_url)
    return f"{base}{path}"


def _read_json_response(response: urllib.request.urlopen):
    payload = response.read().decode("utf-8", errors="replace")
    return json.loads(payload), dict(response.headers), response.status


def fetch_voices(base_url: str):
    try:
        request = urllib.request.Request(_api_url(base_url, "/v1/audio/voices"))
        with urllib.request.urlopen(request, timeout=10) as response:
            body, _headers, _status = _read_json_response(response)
        voices = []
        for item in body.get("voices", []):
            if isinstance(item, str):
                voices.append(item)
            elif isinstance(item, dict) and item.get("name"):
                voices.append(item["name"])
        return sorted(set(v for v in voices if v))
    except Exception:
        return []


def fetch_models(base_url: str):
    try:
        request = urllib.request.Request(_api_url(base_url, "/v1/models"))
        with urllib.request.urlopen(request, timeout=10) as response:
            body, _headers, _status = _read_json_response(response)
        models = []
        for item in body.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                models.append(item["id"])
        return models
    except Exception:
        return ["tts-1"]


def _request_speech(base_url: str, text: str, voice: str, model: str, response_format: str):
    start = time.perf_counter()
    payload = {
        "model": model or "tts-1",
        "input": text,
        "voice": voice or "jo",
        "response_format": response_format or "wav",
    }
    request = urllib.request.Request(
        _api_url(base_url, "/v1/audio/speech"),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        audio = response.read()
        headers = dict(response.headers)
        status = response.status
    elapsed = time.perf_counter() - start
    return audio, headers, status, elapsed


def generate_audio(text: str, voice: str, model: str, base_url: str):
    text = (text or "").strip()
    if not text:
        return None, "⚠️ `text` is empty."

    if not (base_url or "").strip():
        return None, "⚠️ Set a valid NeuTTS base URL first."

    try:
        audio_bytes, headers, status, elapsed = _request_speech(base_url, text, voice, model, "wav")
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        return None, f"⚠️ HTTP {exc.code}\n```json\n{error}\n```"
    except urllib.error.URLError as exc:
        return None, f"⚠️ Could not reach NeuTTS endpoint: `{exc}`"
    except Exception as exc:
        return None, f"⚠️ Synthesis failed: `{exc}`"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        handle.write(audio_bytes)
        path = handle.name
    elapsed = f"{elapsed:.3f}s"

    server_elapsed = headers.get("X-NeuTTS-Elapsed-Seconds", "n/a")
    server_audio = headers.get("X-NeuTTS-Audio-Seconds", "n/a")
    server_rtf = headers.get("X-NeuTTS-RTF", "n/a")
    backbone = headers.get("X-NeuTTS-Backbone-Device", "n/a")
    codec = headers.get("X-NeuTTS-Codec-Device", "n/a")

    text_status = (
        f"HTTP {status}\n"
        f"- `request`: {elapsed}\n"
        f"- `server_elapsed`: {server_elapsed}s\n"
        f"- `audio_seconds`: {server_audio}s\n"
        f"- `rtf`: {server_rtf}\n"
        f"- `backbone`: {backbone}\n"
        f"- `codec`: {codec}\n"
        f"- `audio_file`: `{Path(path).name}`\n"
    )

    return path, text_status


def refresh_voices(base_url: str):
    voices = fetch_voices(base_url)
    if not voices:
        return gr.Dropdown.update(choices=["jo"], value="jo")
    return gr.Dropdown.update(choices=voices, value=voices[0])


def build_ui(base_url_default: str, host: str, port: int, share: bool):
    voices = fetch_voices(base_url_default)
    if not voices:
        voices = ["jo"]
    models = fetch_models(base_url_default)
    if not models:
        models = ["tts-1"]

    with gr.Blocks(title="NeuTTS Browser UI") as app:
        gr.Markdown("## NeuTTS Browser UI")
        gr.Markdown(
            "Send text to your local NeuTTS OpenAI-compatible endpoint and play the result directly in the browser."
        )

        with gr.Row():
            base_input = gr.Textbox(
                label="NeuTTS base URL",
                value=base_url_default,
                placeholder="http://127.0.0.1:12435",
            )
            refresh_btn = gr.Button("Refresh voices", scale=0)

        with gr.Row():
            voice_select = gr.Dropdown(
                choices=voices,
                value=voices[0] if voices else "jo",
                label="Voice",
                scale=2,
            )
            model_select = gr.Dropdown(
                choices=models,
                value=models[0] if models else "tts-1",
                label="Model",
                scale=2,
                allow_custom_value=True,
            )
            response_format = gr.Dropdown(
                choices=["wav"],
                value="wav",
                label="Response Format",
                interactive=False,
            )

        text_input = gr.Textbox(
            label="Input text",
            lines=6,
            placeholder="Write what you want to hear...",
        )
        generate_button = gr.Button("Generate speech")
        audio_out = gr.Audio(label="Generated audio", type="filepath")
        status = gr.Markdown()

        def _generate(_text, voice, model, base, _fmt):
            return generate_audio(_text, voice, model, base)

        generate_button.click(
            fn=_generate,
            inputs=[text_input, voice_select, model_select, base_input, response_format],
            outputs=[audio_out, status],
        )

        refresh_btn.click(
            fn=refresh_voices,
            inputs=[base_input],
            outputs=[voice_select],
        )

    app.queue()
    app.launch(server_name=host, server_port=port, share=share)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NEUTTS_UI_BASE_URL", "http://127.0.0.1:12435"),
    )
    parser.add_argument("--host", default=os.environ.get("NEUTTS_UI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NEUTTS_UI_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_ui(args.base_url, args.host, args.port, args.share)


if __name__ == "__main__":
    main()
