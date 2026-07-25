import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class DatasetPipelineConfig:
    name: str = "tinystories"
    version: str = "v1"
    path: str = "datasets/demo/tiny.txt"
    format: str = "txt"
    loader_type: str = "txt"
    encoding: str = "utf-8"
    streaming: bool = True
    prefetch_batches: int = 4
    tokenizer_algorithm: str = "bpe"
    vocab_dir: str = "models/tokenizer"
    packing: bool = True
    max_sequence_length: int = 2048
    add_eos: bool = True
    binary_format: str = "uint16"
    mmap: bool = True
    shard_size_mb: int = 100
    cache_enabled: bool = True
    cache_dir: str = ".cache/dataset"
    normalize_unicode: bool = True
    clean_html: bool = True
    trim_whitespace: bool = True
    remove_control_chars: bool = True
    min_length: int = 5
    max_length: int = 100000
    deduplicate: bool = True
    language: str = "en"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "configs/dataset.yaml") -> "DatasetPipelineConfig":
        if not os.path.exists(yaml_path):
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        ds = data.get("dataset", {})
        ldr = data.get("loader", {})
        strm = data.get("streaming", {})
        tok = data.get("tokenizer", {})
        pck = data.get("packing", {})
        bin_cfg = data.get("binary", {})
        cch = data.get("cache", {})
        cln = data.get("cleaning", {})
        val = data.get("validation", {})

        return cls(
            name=ds.get("name", "tinystories"),
            version=ds.get("version", "v1"),
            path=ds.get("path", "datasets/demo/tiny.txt"),
            format=ds.get("format", "txt"),
            loader_type=ldr.get("type", "txt"),
            encoding=ldr.get("encoding", "utf-8"),
            streaming=strm.get("enabled", True),
            prefetch_batches=strm.get("prefetch_batches", 4),
            tokenizer_algorithm=tok.get("algorithm", "bpe"),
            vocab_dir=tok.get("vocab_dir", "models/tokenizer"),
            packing=pck.get("enabled", True),
            max_sequence_length=pck.get("max_sequence_length", 2048),
            add_eos=pck.get("add_eos", True),
            binary_format=bin_cfg.get("format", "uint16"),
            mmap=bin_cfg.get("mmap", True),
            shard_size_mb=bin_cfg.get("shard_size_mb", 100),
            cache_enabled=cch.get("enabled", True),
            cache_dir=cch.get("dir", ".cache/dataset"),
            normalize_unicode=cln.get("normalize_unicode", True),
            clean_html=cln.get("clean_html", True),
            trim_whitespace=cln.get("trim_whitespace", True),
            remove_control_chars=cln.get("remove_control_chars", True),
            min_length=val.get("min_length", 5),
            max_length=val.get("max_length", 100000),
            deduplicate=val.get("deduplicate", True),
            language=val.get("language", "en"),
        )
