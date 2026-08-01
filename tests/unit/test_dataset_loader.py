import csv
import json
import os
import tempfile

from engine.dataset.loader.csv import CsvLoader
from engine.dataset.loader.json import JsonLoader
from engine.dataset.loader.jsonl import JsonlLoader
from engine.dataset.loader.text import TextLoader


def test_text_loader() -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write("Line 1\n\nLine 2\n")
        temp_name = f.name

    try:
        loader = TextLoader(temp_name)
        samples = list(loader.stream())
        assert len(samples) == 2
        assert samples[0].text == "Line 1"
        assert samples[1].text == "Line 2"
    finally:
        os.remove(temp_name)


def test_json_loader() -> None:
    data = [{"text": "Hello"}, {"text": "World"}]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        temp_name = f.name

    try:
        loader = JsonLoader(temp_name)
        samples = list(loader.stream())
        assert len(samples) == 2
        assert samples[0].text == "Hello"
        assert samples[1].text == "World"
    finally:
        os.remove(temp_name)


def test_jsonl_loader() -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write('{"text": "Line 1"}\n{"text": "Line 2"}\n')
        temp_name = f.name

    try:
        loader = JsonlLoader(temp_name)
        samples = list(loader.stream())
        assert len(samples) == 2
        assert samples[0].text == "Line 1"
        assert samples[1].text == "Line 2"
    finally:
        os.remove(temp_name)


def test_csv_loader() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", delete=False, encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerow(["Sample 1", "0"])
        writer.writerow(["Sample 2", "1"])
        temp_name = f.name

    try:
        loader = CsvLoader(temp_name)
        samples = list(loader.stream())
        assert len(samples) == 2
        assert samples[0].text == "Sample 1"
        assert samples[0].metadata["label"] == "0"
        assert samples[1].text == "Sample 2"
    finally:
        os.remove(temp_name)
