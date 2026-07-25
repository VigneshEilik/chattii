from engine.dataset.base.dataset import TextSample
from engine.dataset.validator.language import LanguageValidator
from engine.dataset.validator.encoding import EncodingValidator
from engine.dataset.validator.duplicates import DuplicateValidator
from engine.dataset.validator.corruption import CorruptionValidator
from engine.dataset.validator.quality import QualityValidator


def test_language_validator() -> None:
    validator = LanguageValidator(target_lang="en")
    valid, _ = validator.validate(TextSample("This is an English sentence."))
    assert valid

    # Heuristic non-ASCII check for English
    invalid, reason = validator.validate(TextSample("Это русский текст, он не должен пройти проверку."))
    assert not invalid
    assert "non-ASCII" in reason


def test_encoding_validator() -> None:
    validator = EncodingValidator()
    valid, _ = validator.validate(TextSample("Clean string"))
    assert valid


def test_duplicate_validator() -> None:
    validator = DuplicateValidator(exact=True, near_dup=True, threshold=0.8)

    # First add
    v1, _ = validator.validate(TextSample("Unique text number one"))
    assert v1

    # Exact duplicate
    v2, _ = validator.validate(TextSample("Unique text number one"))
    assert not v2

    # Near duplicate
    v3, _ = validator.validate(TextSample("Unique text number two"))
    assert not v3  # Matches close enough to shingle threshold


def test_corruption_validator() -> None:
    validator = CorruptionValidator()
    valid, _ = validator.validate(TextSample("Valid text"))
    assert valid

    invalid, _ = validator.validate(TextSample("   "))
    assert not invalid


def test_quality_validator() -> None:
    validator = QualityValidator(min_length=10, max_length=100)
    valid, _ = validator.validate(TextSample("This is a quality text sample."))
    assert valid

    too_short, _ = validator.validate(TextSample("Short"))
    assert not too_short

    garbage, _ = validator.validate(TextSample("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"))
    assert not garbage
