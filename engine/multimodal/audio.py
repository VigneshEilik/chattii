from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_seconds: float


class AudioProcessor:
    """Handles speech recognition (STT) and speech synthesis (TTS)."""

    def transcribe(
        self, audio_bytes: bytes, filename: str = "audio.wav"
    ) -> TranscriptionResult:
        # Mock transcription result
        return TranscriptionResult(
            text="Hello Chatii, explain machine learning.",
            language="en",
            duration_seconds=3.5,
        )

    def synthesize(self, text: str, voice: str = "alloy") -> bytes:
        # Mock synthetic audio bytes WAV payload
        return b"RIFF....WAVEfmt ....data...."


_audio_processor = AudioProcessor()


def get_audio_processor() -> AudioProcessor:
    return _audio_processor
