from engine.dataset.base.dataset import TextSample
from engine.dataset.cleaner.whitespace import WhitespaceCleaner
from engine.dataset.cleaner.unicode import UnicodeCleaner
from engine.dataset.cleaner.html import HTMLCleaner
from engine.dataset.cleaner.emoji import EmojiCleaner
from engine.dataset.cleaner.control_chars import ControlCharacterCleaner
from engine.dataset.cleaner.profanity import ProfanityCleaner
from engine.dataset.cleaner.pipeline import CleaningPipeline


def test_whitespace_cleaner() -> None:
    cleaner = WhitespaceCleaner()
    sample = TextSample(text="  Hello   World \n\n Newline  ")
    cleaned = cleaner.clean(sample)
    assert cleaned.text == "Hello World\nNewline"


def test_unicode_cleaner() -> None:
    cleaner = UnicodeCleaner(form="NFC")
    sample = TextSample(text="cafe\u0301")  # e + acute accent (decomposed)
    cleaned = cleaner.clean(sample)
    assert cleaned.text == "café"  # normalized NFC (composed)


def test_html_cleaner() -> None:
    cleaner = HTMLCleaner()
    sample = TextSample(text="<p>Hello &amp; Welcome</p>")
    cleaned = cleaner.clean(sample)
    assert cleaned.text == "Hello & Welcome"


def test_emoji_cleaner() -> None:
    cleaner = EmojiCleaner(replace_with="")
    sample = TextSample(text="Hello 😀 text! 🚀")
    cleaned = cleaner.clean(sample)
    assert cleaned.text == "Hello  text! "


def test_control_character_cleaner() -> None:
    cleaner = ControlCharacterCleaner()
    sample = TextSample(text="Hello \x00 world \x07 text \t newline \n")
    cleaned = cleaner.clean(sample)
    assert "\x00" not in cleaned.text
    assert "\x07" not in cleaned.text
    assert "\t" in cleaned.text
    assert "\n" in cleaned.text


def test_profanity_cleaner() -> None:
    cleaner = ProfanityCleaner(redact=True)
    sample = TextSample(text="This text contains bannedword1.")
    cleaned = cleaner.clean(sample)
    assert " [REDACTED] " in cleaned.text
    assert cleaned.metadata["contains_profanity"] is True


def test_cleaning_pipeline() -> None:
    pipeline = CleaningPipeline(
        cleaners=[
            HTMLCleaner(),
            UnicodeCleaner(),
            WhitespaceCleaner(),
        ]
    )
    sample = TextSample(text="<p>  Hello &amp;  Welcome  </p>")
    cleaned = pipeline.clean(sample)
    assert cleaned.text == "Hello & Welcome"
