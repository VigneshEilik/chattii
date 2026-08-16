import io

from fastapi.testclient import TestClient

from engine.api.api import app

AUTH_HEADERS = {"Authorization": "Bearer chatii-admin-root-key"}


def test_multimodal_vision_ocr():
    with TestClient(app) as client:
        files = {"file": ("equation.png", io.BytesIO(b"png_data"), "image/png")}
        resp = client.post(
            "/v1/multimodal/vision/analyze", files=files, headers=AUTH_HEADERS
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "equation.png"
        assert "ocr_text" in data


def test_multimodal_audio_stt_tts():
    with TestClient(app) as client:
        files = {"file": ("speech.wav", io.BytesIO(b"wav_data"), "audio/wav")}
        stt_resp = client.post(
            "/v1/multimodal/audio/transcribe", files=files, headers=AUTH_HEADERS
        )
        assert stt_resp.status_code == 200
        assert "text" in stt_resp.json()

        tts_resp = client.post(
            "/v1/multimodal/audio/speech?text=Hello", headers=AUTH_HEADERS
        )
        assert tts_resp.status_code == 200
        assert tts_resp.headers["content-type"] == "audio/wav"
