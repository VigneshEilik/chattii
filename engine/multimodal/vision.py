from dataclasses import dataclass


@dataclass
class ImageAnalysisResult:
    filename: str
    width: int
    height: int
    format: str
    ocr_text: str
    caption: str


class VisionProcessor:
    """Processes images for OCR text extraction and image understanding."""

    def analyze_image(self, filename: str, image_bytes: bytes) -> ImageAnalysisResult:
        # Mock OCR / analysis output for input image
        ocr_text = "Sample math problem: 2x + 5 = 15"
        caption = f"An uploaded document image ({filename})"
        return ImageAnalysisResult(
            filename=filename,
            width=800,
            height=600,
            format="PNG",
            ocr_text=ocr_text,
            caption=caption,
        )


_vision_processor = VisionProcessor()


def get_vision_processor() -> VisionProcessor:
    return _vision_processor
