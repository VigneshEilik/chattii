from engine.multimodal.audio import AudioProcessor
from engine.multimodal.vision import VisionProcessor


def test_vision_analyze_image():
    proc = VisionProcessor()
    res = proc.analyze_image("sample.png", b"fake_bytes")
    assert res.filename == "sample.png"
    assert res.ocr_text != ""
    assert res.format == "PNG"


def test_vision_dimensions():
    proc = VisionProcessor()
    res = proc.analyze_image("test.jpg", b"123")
    assert res.width > 0
    assert res.height > 0


def test_audio_transcribe():
    proc = AudioProcessor()
    res = proc.transcribe(b"audio_payload", filename="voice.wav")
    assert res.text != ""
    assert res.language == "en"
    assert res.duration_seconds > 0


def test_audio_synthesize():
    proc = AudioProcessor()
    wav_bytes = proc.synthesize("Hello world", voice="alloy")
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0


def test_vision_caption():
    proc = VisionProcessor()
    res = proc.analyze_image("chart.png", b"data")
    assert "chart.png" in res.caption


def test_audio_voice_options():
    proc = AudioProcessor()
    wav_bytes = proc.synthesize("Testing voice", voice="echo")
    assert isinstance(wav_bytes, bytes)


def test_vision_empty_bytes_handling():
    proc = VisionProcessor()
    res = proc.analyze_image("empty.png", b"")
    assert res.filename == "empty.png"


def test_audio_transcribe_duration():
    proc = AudioProcessor()
    res = proc.transcribe(b"bytes")
    assert res.duration_seconds == 3.5
