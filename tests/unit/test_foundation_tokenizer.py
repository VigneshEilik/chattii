import shutil
import tempfile
from pathlib import Path

from engine.tokenizer.algorithms.character import CharacterTokenizer
from engine.tokenizer.algorithms.word import WordTokenizer
from engine.tokenizer.normalizer.unicode import UnicodeNormalizer
from engine.tokenizer.pretokenizer.split import RegexPretokenizer
from engine.tokenizer.vocabulary.manager import VocabularyManager


def test_unicode_normalizer() -> None:
    """
    Tests case mapping, Unicode normalization, and whitespace cleanup.
    """
    normalizer = UnicodeNormalizer(unicode_normalization="NFKC", lowercase=True)

    # Test normalization + casing + strip
    assert normalizer.normalize("  Héllò Wòrld  ") == "héllò wòrld"

    # Test collapse spaces/tabs but keep single newline
    assert normalizer.normalize("hello   \t   world\n\nnext") == "hello world\nnext"

    # Test empty string safety
    assert normalizer.normalize("") == ""


def test_regex_pretokenizer() -> None:
    """
    Tests pretokenization splits for character and regex word rules.
    """
    char_pretok = RegexPretokenizer(mode="char")
    assert char_pretok.pretokenize("hello") == ["h", "e", "l", "l", "o"]

    word_pretok = RegexPretokenizer(mode="word")
    # Test punctuation separation and space preservation
    tokens = word_pretok.pretokenize("Hello, world!")
    assert tokens == ["Hello", ",", " world", "!"]


def test_vocabulary_manager() -> None:
    """
    Tests VocabularyManager operations: add, lookup, unknown mapping,
    and special tokens.
    """
    manager = VocabularyManager(special_tokens=["<pad>", "<unk>", "<s>", "</s>"])

    assert len(manager) == 4
    assert manager.get_special_tokens()["bos"] == "<s>"

    # Add new token
    tok_id = manager.add_token("chatii")
    assert tok_id == 4
    assert manager.token_to_id("chatii") == 4
    assert manager.id_to_token(4) == "chatii"

    # Test unknown fallback
    assert manager.token_to_id("missing") == manager.token_to_id("<unk>")


def test_character_tokenizer_lifecycle() -> None:
    """
    Tests CharacterTokenizer training, encoding, decoding, and serialization.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        corpus_file = Path(temp_dir) / "corpus.txt"
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write("abc d\n")

        tokenizer = CharacterTokenizer(unicode_normalization="NFKC", lowercase=False)
        tokenizer.train(str(corpus_file))

        # Check vocab size (5 specials + 'a', 'b', 'c', ' ', 'd' = 10)
        assert tokenizer.get_vocab_size() == 10

        # Test encode with special tokens
        ids = tokenizer.encode("abc", add_special_tokens=True)
        assert ids[0] == tokenizer.vocab_manager.token_to_id("<s>")
        assert ids[-1] == tokenizer.vocab_manager.token_to_id("</s>")

        # Test decode roundtrip
        encoded_ids = tokenizer.encode("abc d", add_special_tokens=False)
        assert tokenizer.decode(encoded_ids) == "abc d"

        # Test save & load
        save_path = Path(temp_dir) / "char_tok"
        tokenizer.save(str(save_path))

        new_tokenizer = CharacterTokenizer()
        new_tokenizer.load(str(save_path))

        assert new_tokenizer.get_vocab_size() == 10
        assert new_tokenizer.decode(encoded_ids) == "abc d"

    finally:
        shutil.rmtree(temp_dir)


def test_word_tokenizer_lifecycle() -> None:
    """
    Tests WordTokenizer training, encoding, decoding, and serialization.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        corpus_file = Path(temp_dir) / "corpus.txt"
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write("Hello, world! Hello chatii.")

        tokenizer = WordTokenizer(unicode_normalization="NFKC", lowercase=False)
        tokenizer.train(str(corpus_file))

        # Test encode and decode roundtrip
        text = "Hello, chatii!"
        ids = tokenizer.encode(text, add_special_tokens=False)

        # The word "chatii!" has "chatii" in vocab, but "!" is also in vocab.
        # Check roundtrip
        assert tokenizer.decode(ids) == "Hello, chatii!"

        # Save and load
        save_path = Path(temp_dir) / "word_tok"
        tokenizer.save(str(save_path))

        new_tokenizer = WordTokenizer()
        new_tokenizer.load(str(save_path))
        assert new_tokenizer.decode(ids) == "Hello, chatii!"

    finally:
        shutil.rmtree(temp_dir)
