from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from engine.multimodal.audio import get_audio_processor
from engine.multimodal.vision import get_vision_processor

router = APIRouter(prefix="/v1/multimodal", tags=["Multimodal"])


@router.post("/vision/analyze")
async def analyze_image(file: UploadFile = File(...)):  # noqa: B008
    try:
        content_bytes = await file.read()
        processor = get_vision_processor()
        res = processor.analyze_image(file.filename, content_bytes)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Vision processing failed: {e}")


@router.post("/audio/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):  # noqa: B008
    try:
        content_bytes = await file.read()
        processor = get_audio_processor()
        res = processor.transcribe(content_bytes, filename=file.filename)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio transcription failed: {e}")


@router.post("/audio/speech")
def synthesize_speech(text: str, voice: str = "alloy"):
    processor = get_audio_processor()
    audio_bytes = processor.synthesize(text=text, voice=voice)
    return Response(content=audio_bytes, media_type="audio/wav")
