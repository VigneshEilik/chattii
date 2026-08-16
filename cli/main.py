import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from engine.tokenizer.benchmark import TokenizerBenchmark
from engine.tokenizer.health import TokenizerHealthChecker
from engine.tokenizer.inspector import TokenizerInspector
from engine.tokenizer.profiler import TokenizerProfiler
from engine.tokenizer.registry import TokenizerRegistry
from engine.tokenizer.service import TokenizerService
from engine.tokenizer.visualizer import TokenizerVisualizer
from engine.transformer.benchmark import MathBenchmark, TransformerBenchmark
from engine.transformer.config import ModelMathConfig
from engine.transformer.model import ChatiiModel
from engine.transformer.nn.normalization import LayerNorm, RMSNorm
from engine.transformer.position.rope import precompute_freqs_cis
from engine.transformer.registry import (
    ActivationRegistry,
    NormalizationRegistry,
    PositionRegistry,
)
from engine.transformer.validator import MathValidator, TransformerValidator
from engine.transformer.visualizer import TransformerVisualizer


def _handle_list_tokenizers() -> None:
    print("Registered Tokenizers:")
    for name in TokenizerRegistry._registry.keys():
        print(f"  - {name} (active)")
    print("\nFuture Integrations (stubs):")
    print("  - sentencepiece")
    print("  - wordpiece")
    print("  - unigram")
    print("  - bbpe")
    sys.exit(0)


def _handle_train_tokenizer(args: argparse.Namespace) -> None:
    service = TokenizerService(args.algorithm, max_vocab_size=args.vocab_size)
    service.train(args.corpus)
    service.save(args.vocab_dir)
    print(f"Success: Tokenizer trained and saved successfully to '{args.vocab_dir}'")
    sys.exit(0)


def _handle_encode(args: argparse.Namespace) -> None:
    service = TokenizerService(args.algorithm)
    service.load(args.vocab_dir)
    ids = service.encode(args.text)
    print(f"Encoded Token IDs: {ids}")
    sys.exit(0)


def _handle_decode(args: argparse.Namespace) -> None:
    service = TokenizerService(args.algorithm)
    service.load(args.vocab_dir)

    try:
        if "," in args.ids:
            token_ids = [int(i.strip()) for i in args.ids.split(",") if i.strip()]
        else:
            token_ids = json.loads(args.ids)
    except Exception:
        print(
            "Error: IDs parameter must be comma-separated integers "
            "or a valid JSON list."
        )
        sys.exit(1)

    text = service.decode(token_ids)
    print(f"Decoded Text string: '{text}'")
    sys.exit(0)


def _handle_validate_tokenizer(args: argparse.Namespace) -> None:
    healthy, errors = TokenizerHealthChecker.check_health(
        args.vocab_dir, args.algorithm
    )
    if healthy:
        print(f"Success: Vocabulary directory '{args.vocab_dir}' is 100% healthy.")
        sys.exit(0)
    else:
        print("Validation Failed. Found errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


def _handle_inspect_tokenizer(args: argparse.Namespace) -> None:
    TokenizerInspector.inspect_directory(args.vocab_dir)
    sys.exit(0)


def _handle_benchmark_tokenizer(args: argparse.Namespace) -> None:
    service = TokenizerService(args.algorithm)
    service.load(args.vocab_dir)
    metrics = TokenizerBenchmark.run(
        service.tokenizer, [args.sample], iterations=args.iterations
    )
    print(f"\n--- Tokenizer Benchmark Metrics ({args.algorithm.upper()}) ---")
    print(f"Iterations      : {metrics['iterations']}")
    print(f"Processed Tokens: {metrics['total_tokens_processed']}")
    print(
        f"Encode speed    : "
        f"{metrics['encode_throughput_tokens_per_sec']:.2f} tokens/sec"
    )
    print(
        f"Decode speed    : "
        f"{metrics['decode_throughput_tokens_per_sec']:.2f} tokens/sec"
    )
    sys.exit(0)


def _handle_profile_tokenizer(args: argparse.Namespace) -> None:
    service = TokenizerService(args.algorithm)
    service.load(args.vocab_dir)

    prof = TokenizerProfiler.profile_action(service.tokenizer.encode, args.text)
    print(f"\n--- Tokenizer Profiler Metrics ({args.algorithm.upper()}) ---")
    print(f"Time Taken: {prof['elapsed_seconds']} seconds")
    print(f"Peak RAM  : {prof['ram_peak_mb']} MB ({prof['ram_peak_bytes']} bytes)")
    sys.exit(0)


def _handle_visualize_tokenizer(args: argparse.Namespace) -> None:
    lengths = TokenizerVisualizer.render_length_distribution(args.vocab_dir)
    print(lengths)
    history = TokenizerVisualizer.render_merge_history(args.vocab_dir)
    print(history)
    sys.exit(0)


def _handle_export_import() -> None:
    print("Import/Export is handled directly by vocabulary serialization schemas.")
    sys.exit(0)


def _handle_inspect_math() -> None:
    config = ModelMathConfig.load_from_yaml()
    print("--- Chatii Transformer Math Engine Configuration ---")
    print(f"Activation         : {config.activation}")
    print(f"Normalization      : {config.normalization}")
    print(f"Position Embedding : {config.position_embedding}")
    print(f"Dropout Rate       : {config.dropout}")
    print(f"Initializer        : {config.initializer}")
    print(f"Precision Dtype    : {config.dtype}")
    print(f"Compute Device     : {config.device}")
    print(f"Model Dimension    : {config.d_model}")
    print(f"Attention Heads    : {config.n_heads}")

    print("\n--- Registered Math Primitives ---")
    print(f"Activations   : {list(ActivationRegistry._registry.keys())}")
    print(f"Normalizations: {list(NormalizationRegistry._registry.keys())}")
    print(f"Positionings  : {list(PositionRegistry._registry.keys())}")
    sys.exit(0)


def _handle_benchmark_math(args: argparse.Namespace) -> None:
    metrics = MathBenchmark.run_benchmark(iterations=args.iterations)
    print("\n--- Transformer Math Engine Benchmark Metrics ---")
    print(f"Iterations      : {metrics['iterations']}")
    print(f"Elapsed Time    : {metrics['elapsed_seconds']} s")
    print(f"Throughput Speed: {metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Latency per Iter: {metrics['latency_ms_per_iter']:.4f} ms")
    sys.exit(0)


def _handle_visualize_rope() -> None:
    print("\n--- Rotary Position Embedding (RoPE) Frequencies ---")
    freqs_cis = precompute_freqs_cis(dim=16, end=8)
    for pos in range(min(8, freqs_cis.shape[0])):
        line = " ".join([f"{val.real:+.2f}" for val in freqs_cis[pos][:4]])
        print(f"Pos {pos}: [{line}]")
    sys.exit(0)


def _handle_test_normalization() -> None:
    print("\n--- Testing LayerNorm & RMSNorm Sanitization ---")
    x = torch.randn(4, 16, 64)
    ln = LayerNorm(64)
    rms = RMSNorm(64)

    out_ln = ln(x)
    ok_ln, err_ln = MathValidator.validate_output(out_ln, "LayerNorm")

    out_rms = rms(x)
    ok_rms, err_rms = MathValidator.validate_output(out_rms, "RMSNorm")

    if ok_ln and ok_rms:
        print("Success: LayerNorm and RMSNorm outputs are 100% stable.")
        sys.exit(0)
    else:
        print(f"Error: Validation failed! LayerNorm: {err_ln}, RMSNorm: {err_rms}")
        sys.exit(1)


def _handle_train_forward_demo(args: argparse.Namespace) -> None:
    config = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(config)
    batch_size = args.batch_size
    seq_len = args.seq_len
    print("\nRunning Model Forward Demonstration...")
    print(f"Input Shape: [{batch_size}, {seq_len}]")
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"v\nEmbedding: [{batch_size}, {seq_len}, {config.d_model}]")
    print(f"v\nDecoder Stack Layer Count: {config.num_layers}")
    logits = model(x)
    print(f"v\nOutput Shape: {list(logits.shape)}")
    print("Success: Forward pass executed cleanly.")
    sys.exit(0)


def _handle_inspect_model() -> None:
    config = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(config)
    print(model.summary())
    sys.exit(0)


def _handle_benchmark_transformer(args: argparse.Namespace) -> None:
    config = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(config)
    print("\nBenchmarking Transformer Model Throughput...")
    metrics = TransformerBenchmark.run_benchmark(
        model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        iterations=args.iterations,
    )
    print(f"Iterations      : {metrics['iterations']}")
    print(f"Elapsed Time    : {metrics['elapsed_seconds']} s")
    print(f"Throughput Speed: {metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Latency per Iter: {metrics['latency_ms_per_iter']:.4f} ms")
    sys.exit(0)


def _handle_validate_model() -> None:
    config = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(config)
    x = torch.randint(0, config.vocab_size, (2, 128))
    logits = model(x)
    ok, errors = TransformerValidator.validate_model(model, x, logits)
    if ok:
        print("\nSuccess: Model configuration and weight structures are 100% healthy.")
        sys.exit(0)
    else:
        print("\nError: Model validation failed with errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


def _handle_count_parameters() -> None:
    config = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(config)
    print(f"\nTotal Learnable Parameters: {model.num_parameters():,}")
    sys.exit(0)


def _handle_visualize_attention(args: argparse.Namespace) -> None:
    grid = TransformerVisualizer.render_attention_mask(seq_len=args.seq_len)
    print(f"\n{grid}")
    sys.exit(0)


def _handle_train_model(args: argparse.Namespace) -> None:
    from engine.tokenizer.service import TokenizerService
    from engine.training.callbacks.checkpoint import CheckpointCallback
    from engine.training.callbacks.logging import LoggingCallback
    from engine.training.callbacks.lr_monitor import LRMonitorCallback
    from engine.training.callbacks.progress import ProgressCallback
    from engine.training.callbacks.validation import ValidationCallback
    from engine.training.service import TrainingService
    from engine.transformer.config import ModelMathConfig
    from engine.transformer.model import ChatiiModel

    service = TrainingService()
    tokenizer = TokenizerService("bpe")
    vocab_dir = "models/tokenizer"
    if not os.path.exists(vocab_dir):
        print(
            "Tokenizer vocabulary not found. "
            "Training tokenizer vocabulary automatically..."
        )
        tokenizer.train(args.train_file)
        tokenizer.save(vocab_dir)
    else:
        tokenizer.load(vocab_dir)

    model_cfg = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(model_cfg)

    callbacks = [
        ProgressCallback(),
        LRMonitorCallback(),
        CheckpointCallback(
            save_every_steps=service.config.checkpoint.save_every_steps,
            checkpoint_dir=service.config.checkpoint.dir,
            keep_top_n=service.config.checkpoint.keep_top_n,
        ),
        LoggingCallback(log_every_steps=50),
    ]
    if args.val_file:
        callbacks.append(
            ValidationCallback(eval_every_steps=service.config.validation.every_steps)
        )

    service.run_training(
        model=model,
        tokenizer=tokenizer,
        train_file=args.train_file,
        val_file=args.val_file,
        callbacks=callbacks,
        resume_dir=args.resume_dir,
    )
    sys.exit(0)


def _handle_resume_training(args: argparse.Namespace) -> None:
    from engine.tokenizer.service import TokenizerService
    from engine.training.callbacks.progress import ProgressCallback
    from engine.training.service import TrainingService
    from engine.transformer.config import ModelMathConfig
    from engine.transformer.model import ChatiiModel

    service = TrainingService()
    tokenizer = TokenizerService("bpe")
    tokenizer.load("models/tokenizer")

    model_cfg = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(model_cfg)

    service.run_training(
        model=model,
        tokenizer=tokenizer,
        train_file=args.train_file,
        val_file=args.val_file,
        callbacks=[ProgressCallback()],
        resume_dir=args.checkpoint_dir,
    )
    sys.exit(0)


def _handle_evaluate_model(args: argparse.Namespace) -> None:
    from engine.tokenizer.service import TokenizerService
    from engine.training.service import TrainingService
    from engine.transformer.config import ModelMathConfig
    from engine.transformer.model import ChatiiModel

    service = TrainingService()
    tokenizer = TokenizerService("bpe")
    tokenizer.load("models/tokenizer")

    model_cfg = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(model_cfg)

    trainer = service.run_training(
        model=model,
        tokenizer=tokenizer,
        train_file=args.val_file,
        val_file=args.val_file,
        resume_dir=args.checkpoint_dir,
    )
    metrics = trainer.evaluate()
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
    sys.exit(0)


def _handle_benchmark_training(args: argparse.Namespace) -> None:
    from engine.tokenizer.service import TokenizerService
    from engine.training.benchmark import TrainerBenchmark
    from engine.training.service import TrainingService
    from engine.transformer.config import ModelMathConfig
    from engine.transformer.model import ChatiiModel

    service = TrainingService()
    tokenizer = TokenizerService("bpe")
    vocab_dir = "models/tokenizer"
    if not os.path.exists(vocab_dir):
        tokenizer.train(args.train_file)
        tokenizer.save(vocab_dir)
    else:
        tokenizer.load(vocab_dir)

    model_cfg = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(model_cfg)

    trainer = service.run_training(
        model=model, tokenizer=tokenizer, train_file=args.train_file
    )
    metrics = TrainerBenchmark.run_benchmark(trainer, iterations=args.iterations)
    print("\nTraining Benchmark Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
    sys.exit(0)


def _handle_profile_training(args: argparse.Namespace) -> None:
    from engine.tokenizer.service import TokenizerService
    from engine.training.profiler import TrainerProfiler
    from engine.training.service import TrainingService
    from engine.transformer.config import ModelMathConfig
    from engine.transformer.model import ChatiiModel

    service = TrainingService()
    tokenizer = TokenizerService("bpe")
    vocab_dir = "models/tokenizer"
    if not os.path.exists(vocab_dir):
        tokenizer.train(args.train_file)
        tokenizer.save(vocab_dir)
    else:
        tokenizer.load(vocab_dir)

    model_cfg = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(model_cfg)

    service.run_training(model=model, tokenizer=tokenizer, train_file=args.train_file)
    metrics = TrainerProfiler.profile_memory()
    print("\nTraining Profiler Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")
    sys.exit(0)


def _handle_inspect_checkpoint(args: argparse.Namespace) -> None:
    trainer_json_path = os.path.join(args.checkpoint_dir, "trainer.json")
    manifest_json_path = os.path.join(args.checkpoint_dir, "manifest.json")

    print(f"\nInspecting Checkpoint: {args.checkpoint_dir}")
    if os.path.exists(trainer_json_path):
        with open(trainer_json_path, "r", encoding="utf-8") as f:
            trainer_data = json.load(f)
        print("\nTrainer State:")
        for k, v in trainer_data.items():
            print(f"  - {k}: {v}")

    if os.path.exists(manifest_json_path):
        with open(manifest_json_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        print("\nManifest Info:")
        for k, v in manifest_data.items():
            print(f"  - {k}: {v}")
    sys.exit(0)


def _handle_export_training_config(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_yaml = {
        "seed": 42,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_epochs": 5,
        "max_steps": None,
        "mixed_precision": "none",
        "grad_clip": 1.0,
        "device": "auto",
        "save_interval": 1000,
        "checkpoint_dir": "models/checkpoints",
        "keep_top_n": 3,
        "eval_interval": 1000,
        "early_stopping_patience": 3,
        "tensorboard": True,
        "csv": True,
        "console": True,
        "dashboard": True,
    }

    optimizer_yaml = {
        "type": "adamw",
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
    }

    scheduler_yaml = {"type": "cosine", "warmup_steps": 1000, "min_lr": 3e-5}

    with open(out_dir / "training.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(training_yaml, f, default_flow_style=False)
    with open(out_dir / "optimizer.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(optimizer_yaml, f, default_flow_style=False)
    with open(out_dir / "scheduler.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(scheduler_yaml, f, default_flow_style=False)

    print(f"Configurations successfully exported to '{args.out_dir}'")
    sys.exit(0)


def _handle_estimate_training_time(args: argparse.Namespace) -> None:
    from engine.tokenizer.service import TokenizerService
    from engine.training.benchmark import TrainerBenchmark
    from engine.training.service import TrainingService
    from engine.transformer.config import ModelMathConfig
    from engine.transformer.model import ChatiiModel

    service = TrainingService()
    tokenizer = TokenizerService("bpe")
    vocab_dir = "models/tokenizer"
    if not os.path.exists(vocab_dir):
        tokenizer.train(args.train_file)
        tokenizer.save(vocab_dir)
    else:
        tokenizer.load(vocab_dir)

    model_cfg = ModelMathConfig.load_from_yaml()
    model = ChatiiModel(model_cfg)

    trainer = service.run_training(
        model=model, tokenizer=tokenizer, train_file=args.train_file
    )

    metrics = TrainerBenchmark.run_benchmark(trainer, iterations=5)
    sec_per_step = metrics["elapsed_seconds"] / 5

    steps_per_epoch = len(trainer.train_loader)
    total_steps = steps_per_epoch * args.epochs
    total_sec = total_steps * sec_per_step

    print("\nEstimated Training Duration:")
    print(f"  - Steps per Epoch: {steps_per_epoch}")
    print(f"  - Total Steps    : {total_steps}")
    print(f"  - Sec per Step   : {sec_per_step:.4f} s")
    print(f"  - Total Time     : {total_sec:.2f} s ({total_sec / 60:.2f} min)")
    sys.exit(0)


def _setup_training_parsers(subparsers: Any) -> None:
    # Command: train-model
    train_model = subparsers.add_parser(
        "train-model", help="Starts training a model on a given corpus"
    )
    train_model.add_argument(
        "--train-file", required=True, help="Path to training corpus file"
    )
    train_model.add_argument(
        "--val-file", default=None, help="Path to validation corpus file"
    )
    train_model.add_argument(
        "--resume-dir",
        default=None,
        help="Path to a checkpoint directory to auto-resume from",
    )

    # Command: resume-training
    resume_tr = subparsers.add_parser(
        "resume-training", help="Restarts training from a saved checkpoint folder"
    )
    resume_tr.add_argument(
        "--checkpoint-dir", required=True, help="Path to saved checkpoint directory"
    )
    resume_tr.add_argument(
        "--train-file", required=True, help="Path to training corpus file"
    )
    resume_tr.add_argument(
        "--val-file", default=None, help="Path to validation corpus file"
    )

    # Command: evaluate-model
    eval_model = subparsers.add_parser(
        "evaluate-model", help="Evaluates checkpoint validation loss and perplexity"
    )
    eval_model.add_argument(
        "--checkpoint-dir", required=True, help="Path to checkpoint directory"
    )
    eval_model.add_argument(
        "--val-file", required=True, help="Path to validation corpus file"
    )

    # Command: benchmark-training
    bench_tr = subparsers.add_parser(
        "benchmark-training", help="Benchmarks step speed and training throughput"
    )
    bench_tr.add_argument(
        "--train-file", required=True, help="Path to training corpus file"
    )
    bench_tr.add_argument(
        "--iterations", type=int, default=10, help="Number of benchmark iterations"
    )

    # Command: profile-training
    prof_tr = subparsers.add_parser(
        "profile-training", help="Profiles memory usage during training"
    )
    prof_tr.add_argument(
        "--train-file", required=True, help="Path to training corpus file"
    )

    # Command: inspect-checkpoint
    inspect_chk = subparsers.add_parser(
        "inspect-checkpoint",
        help="Displays metadata stored inside a checkpoint directory",
    )
    inspect_chk.add_argument(
        "--checkpoint-dir", required=True, help="Path to checkpoint directory"
    )

    # Command: export-training-config
    exp_tr = subparsers.add_parser(
        "export-training-config",
        help="Dumps default training configurations to YAML files",
    )
    exp_tr.add_argument(
        "--out-dir", default="configs", help="Directory to save config YAMLs"
    )

    # Command: estimate-training-time
    est_time = subparsers.add_parser(
        "estimate-training-time",
        help="Estimates training completion based on a short run benchmark",
    )
    est_time.add_argument(
        "--train-file", required=True, help="Path to training corpus file"
    )
    est_time.add_argument("--epochs", type=int, default=5, help="Target total epochs")


def _setup_tokenizer_parsers(subparsers: Any) -> None:

    # Command: train-tokenizer
    train_tok = subparsers.add_parser(
        "train-tokenizer", help="Train a tokenizer from a raw text corpus"
    )
    train_tok.add_argument("--corpus", required=True, help="Path to text corpus file")
    train_tok.add_argument(
        "--vocab-dir", required=True, help="Output destination folder"
    )
    train_tok.add_argument(
        "--algorithm", default="bpe", help="Tokenizer algorithm to use"
    )
    train_tok.add_argument(
        "--vocab-size", type=int, default=1000, help="Target vocabulary size"
    )

    # Command: encode
    enc_parser = subparsers.add_parser(
        "encode", help="Encode raw text string into token IDs"
    )
    enc_parser.add_argument(
        "--vocab-dir", required=True, help="Path to loaded vocabulary folder"
    )
    enc_parser.add_argument("--text", required=True, help="Raw string to encode")
    enc_parser.add_argument("--algorithm", default="bpe", help="Tokenizer algorithm")

    # Command: decode
    dec_parser = subparsers.add_parser("decode", help="Decode token IDs back to text")
    dec_parser.add_argument(
        "--vocab-dir", required=True, help="Path to loaded vocabulary folder"
    )
    dec_parser.add_argument(
        "--ids", required=True, help="JSON array or comma-separated token IDs"
    )
    dec_parser.add_argument("--algorithm", default="bpe", help="Tokenizer algorithm")

    # Command: validate-tokenizer
    validate_parser = subparsers.add_parser(
        "validate-tokenizer",
        help="Run integrity and checksum health checks on saved models",
    )
    validate_parser.add_argument(
        "--vocab-dir", required=True, help="Path to vocabulary directory"
    )
    validate_parser.add_argument(
        "--algorithm", default="bpe", help="Tokenizer algorithm type"
    )

    # Command: inspect-tokenizer
    inspect_parser = subparsers.add_parser(
        "inspect-tokenizer", help="Audit and display vocabulary statistics"
    )
    inspect_parser.add_argument(
        "--vocab-dir", required=True, help="Path to vocabulary directory"
    )

    # Command: benchmark-tokenizer
    bench_tok = subparsers.add_parser(
        "benchmark-tokenizer", help="Measure tokenizer encoding & decoding throughput"
    )
    bench_tok.add_argument(
        "--vocab-dir", required=True, help="Path to vocabulary directory"
    )
    bench_tok.add_argument(
        "--sample",
        default="hello world chatii modular transformer",
        help="Sample text string",
    )
    bench_tok.add_argument(
        "--iterations", type=int, default=500, help="Benchmark run count"
    )
    bench_tok.add_argument("--algorithm", default="bpe", help="Tokenizer algorithm")

    # Command: profile-tokenizer
    prof_tok = subparsers.add_parser(
        "profile-tokenizer", help="Analyze tokenizer CPU & RAM resources footprint"
    )
    prof_tok.add_argument(
        "--vocab-dir", required=True, help="Path to vocabulary directory"
    )
    prof_tok.add_argument(
        "--text",
        default="hello world chatii modular transformer",
        help="Text to encode",
    )
    prof_tok.add_argument("--algorithm", default="bpe", help="Tokenizer algorithm")

    # Command: visualize-tokenizer
    vis_tok = subparsers.add_parser(
        "visualize-tokenizer", help="Print ASCII length curves and BPE merge grids"
    )
    vis_tok.add_argument(
        "--vocab-dir", required=True, help="Path to vocabulary directory"
    )

    # Command: list-tokenizers
    subparsers.add_parser(
        "list-tokenizers", help="List all registered and stubbed tokenizer algorithms"
    )

    # Command: export-tokenizer
    exp_tok = subparsers.add_parser("export-tokenizer", help="Export vocabulary (stub)")
    exp_tok.add_argument("--vocab-dir", required=True, help="Path to source directory")

    # Command: import-tokenizer
    imp_tok = subparsers.add_parser("import-tokenizer", help="Import vocabulary (stub)")
    imp_tok.add_argument("--vocab-dir", required=True, help="Path to target directory")


def _setup_math_parsers(subparsers: Any) -> None:
    # Command: inspect-math
    subparsers.add_parser(
        "inspect-math", help="Inspect math layer configurations and registries"
    )

    # Command: benchmark-math
    bench_m = subparsers.add_parser(
        "benchmark-math", help="Measure throughput and latency of math operations"
    )
    bench_m.add_argument(
        "--iterations", type=int, default=100, help="Number of benchmark iterations"
    )

    # Command: visualize-rope
    subparsers.add_parser(
        "visualize-rope", help="Visualize RoPE rotary complex frequencies"
    )

    # Command: test-normalization
    subparsers.add_parser(
        "test-normalization", help="Sanity test LayerNorm and RMSNorm layers"
    )


def _setup_model_parsers(subparsers: Any) -> None:
    # Command: train-forward-demo
    fw_demo = subparsers.add_parser(
        "train-forward-demo",
        help="Run model forward pass demonstration with dummy data",
    )
    fw_demo.add_argument("--batch-size", type=int, default=2, help="Input batch size")
    fw_demo.add_argument(
        "--seq-len", type=int, default=128, help="Input sequence length"
    )

    # Command: inspect-model
    subparsers.add_parser(
        "inspect-model",
        help="Audits and displays active model layout and configuration",
    )

    # Command: benchmark-transformer
    bench_trans = subparsers.add_parser(
        "benchmark-transformer",
        help="Measure model throughput processing tokens/second",
    )
    bench_trans.add_argument(
        "--batch-size", type=int, default=2, help="Benchmark batch size"
    )
    bench_trans.add_argument(
        "--seq-len", type=int, default=128, help="Benchmark sequence length"
    )
    bench_trans.add_argument(
        "--iterations", type=int, default=20, help="Benchmark iterations"
    )

    # Command: validate-model
    subparsers.add_parser(
        "validate-model",
        help="Audits model for NaN/Inf stability and weight-tying configs",
    )

    # Command: count-parameters
    subparsers.add_parser(
        "count-parameters", help="Audit total number of parameters in the model layers"
    )

    # Command: visualize-attention
    vis_attn = subparsers.add_parser(
        "visualize-attention",
        help="Prints ASCII representation of attention mask boundaries",
    )
    vis_attn.add_argument(
        "--seq-len", type=int, default=16, help="Sequence length for visual grid"
    )


def _setup_dataset_parsers(subparsers: Any) -> None:
    # Command: register-dataset
    reg_ds = subparsers.add_parser(
        "register-dataset", help="Register a dataset locally"
    )
    reg_ds.add_argument("--name", required=True, help="Dataset identifier name")
    reg_ds.add_argument("--path", required=True, help="Path to raw dataset file")
    reg_ds.add_argument("--format", default="txt", help="Loader format type")
    reg_ds.add_argument("--language", default="en", help="Language code")
    reg_ds.add_argument("--version", default="v1", help="Dataset version tag")

    # Command: remove-dataset
    rem_ds = subparsers.add_parser(
        "remove-dataset", help="Remove a dataset from registry"
    )
    rem_ds.add_argument("--name", required=True, help="Dataset identifier name")

    # Command: list-datasets
    subparsers.add_parser("list-datasets", help="List all registered datasets")

    # Command: dataset-info
    ds_info = subparsers.add_parser(
        "dataset-info", help="Print details about a registered dataset"
    )
    ds_info.add_argument("--name", required=True, help="Dataset identifier name")

    # Command: prepare-dataset
    prep_ds = subparsers.add_parser(
        "prepare-dataset", help="Preprocess, clean, and validate a dataset"
    )
    prep_ds.add_argument("--dataset", required=True, help="Dataset identifier name")

    # Command: inspect-dataset
    ins_ds = subparsers.add_parser(
        "inspect-dataset", help="Audit and display dataset summary"
    )
    ins_ds.add_argument("--dataset", required=True, help="Dataset identifier name")

    # Command: validate-dataset
    val_ds = subparsers.add_parser(
        "validate-dataset", help="Validate dataset schema and contents"
    )
    val_ds.add_argument("--dataset", required=True, help="Dataset identifier name")

    # Command: benchmark-dataset
    bench_ds = subparsers.add_parser(
        "benchmark-dataset", help="Measure loading and cleaning throughput"
    )
    bench_ds.add_argument("--dataset", required=True, help="Dataset identifier name")

    # Command: stream-dataset
    strm_ds = subparsers.add_parser(
        "stream-dataset", help="Stream dataset samples to stdout"
    )
    strm_ds.add_argument("--dataset", required=True, help="Dataset identifier name")

    # Command stubs
    cache_ds = subparsers.add_parser(
        "cache-dataset", help="Cache dataset preprocessing results (stub)"
    )
    cache_ds.add_argument("--dataset", required=True, help="Dataset name")
    cache_ds.add_argument("--fingerprint", default=None, help="Cache fingerprint")

    tok_ds = subparsers.add_parser("tokenize-dataset", help="Tokenize a dataset (stub)")
    tok_ds.add_argument("--dataset", required=True, help="Dataset name")

    pack_ds = subparsers.add_parser(
        "pack-dataset", help="Pack sequences for a dataset (stub)"
    )
    pack_ds.add_argument("--dataset", required=True, help="Dataset name")

    stat_ds = subparsers.add_parser(
        "dataset-statistics", help="Calculate dataset statistics (stub)"
    )
    stat_ds.add_argument("--dataset", required=True, help="Dataset name")


def _handle_register_dataset(args: Any) -> None:
    from engine.dataset.service import DatasetService

    service = DatasetService()
    service.register_dataset(
        args.name, args.path, args.format, args.language, args.version
    )


def _handle_remove_dataset(args: Any) -> None:
    from engine.dataset.service import DatasetService

    service = DatasetService()
    service.remove_dataset(args.name)


def _handle_list_datasets() -> None:
    from engine.dataset.service import DatasetService

    service = DatasetService()
    service.list_datasets()


def _handle_dataset_info(args: Any) -> None:
    from engine.dataset.service import DatasetService

    service = DatasetService()
    service.dataset_info(args.name)


def _handle_prepare_dataset(args: Any) -> None:
    from engine.dataset.service import DatasetService

    service = DatasetService()
    service.prepare(args.dataset)


def _handle_inspect_dataset(args: Any) -> None:
    from engine.dataset.service import DatasetService

    service = DatasetService()
    service.inspect(args.dataset)


def _handle_validate_dataset(args: Any) -> None:
    from engine.dataset.service import DatasetService

    service = DatasetService()
    service.validate(args.dataset)


def _handle_benchmark_dataset(args: Any) -> None:
    from engine.dataset.service import DatasetService

    service = DatasetService()
    service.benchmark(args.dataset)


def _handle_stream_dataset(args: Any) -> None:
    from engine.dataset.service import DatasetService

    service = DatasetService()
    service.stream(args.dataset)


def _handle_cache_dataset(args: Any) -> None:
    print(f"Caching dataset '{args.dataset}' with fingerprint '{args.fingerprint}'")


def _handle_tokenize_dataset(args: Any) -> None:
    print(f"Tokenizing dataset '{args.dataset}'")


def _handle_pack_dataset(args: Any) -> None:
    print(f"Packing sequences for dataset '{args.dataset}'")


def _handle_dataset_statistics(args: Any) -> None:
    print(f"Calculating dataset statistics for '{args.dataset}'")


def _setup_distributed_parsers(subparsers: Any) -> None:
    # train-distributed
    tr_d = subparsers.add_parser(
        "train-distributed",
        help="Run distributed training across multi-GPU/node systems",
    )
    tr_d.add_argument("--script", required=True, help="Path to execution script")
    tr_d.add_argument(
        "--world-size", type=int, default=1, help="Total target processes"
    )
    tr_d.add_argument(
        "--master-addr", default="127.0.0.1", help="Master node IP address"
    )

    # resume-distributed
    res_d = subparsers.add_parser(
        "resume-distributed",
        help="Resume distributed training from a directory checkpoint",
    )
    res_d.add_argument(
        "--checkpoint-dir", required=True, help="Path to checkpoint folder"
    )

    # benchmark-distributed
    subparsers.add_parser(
        "benchmark-distributed",
        help="Measures distributed communication latency and scaling throughput",
    )

    # profile-distributed
    subparsers.add_parser(
        "profile-distributed",
        help="Profiles distributed memory and device resource timelines",
    )

    # inspect-cluster
    subparsers.add_parser(
        "inspect-cluster",
        help="Discovers active cluster node name, host IP, and topology",
    )

    # inspect-gpu
    subparsers.add_parser(
        "inspect-gpu",
        help="Audits available GPU capacity, memory layout, and temperatures",
    )

    # validate-distributed
    subparsers.add_parser(
        "validate-distributed",
        help="Validates hardware parameters and master node access ports",
    )

    # export-distributed-config
    exp_c = subparsers.add_parser(
        "export-distributed-config",
        help="Dumps default distributed config properties to YAML",
    )
    exp_c.add_argument("--out-dir", default="configs", help="Output directory path")

    # monitor-training
    subparsers.add_parser(
        "monitor-training",
        help="Streams live Loss, LR, VRAM %%, GPU %%, TFLOPS, power, and temps",
    )


def _handle_train_distributed(args: Any) -> None:
    from engine.distributed.launcher.launcher import DistributedLauncher

    print(
        f"Launching distributed training with world size {args.world_size} on script '{args.script}'..."
    )
    processes = DistributedLauncher.launch(
        script_path=args.script,
        world_size=args.world_size,
        master_addr=args.master_addr,
    )
    for p in processes:
        p.wait()
    print("Distributed training session complete.")


def _handle_resume_distributed(args: Any) -> None:
    from engine.distributed.fault_tolerance.recovery import DistributedRecoveryManager

    latest = DistributedRecoveryManager.find_latest_checkpoint(args.checkpoint_dir)
    if latest:
        print(f"Latest checkpoint located: {latest}")
    else:
        print(f"No checkpoint located under: {args.checkpoint_dir}")


def _handle_benchmark_distributed() -> None:
    from engine.distributed.benchmark import DistributedBenchmark

    metrics = DistributedBenchmark.run_benchmark()
    print("\n--- Distributed Training Benchmark ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def _handle_profile_distributed() -> None:
    from engine.distributed.profiler.gpu import GPUProfiler
    from engine.distributed.profiler.memory import MemoryProfiler

    gpu_stats = GPUProfiler.profile_gpus()
    mem_stats = MemoryProfiler.get_summary()
    print("\n--- Distributed Profiles ---")
    print(f"  System RAM: {mem_stats}")
    print(f"  GPU stats:  {gpu_stats}")


def _handle_inspect_cluster() -> None:
    from engine.distributed.visualizer import DistributedTopologyVisualizer

    layout = DistributedTopologyVisualizer.draw_ascii_topology()
    print(layout)


def _handle_inspect_gpu() -> None:
    from engine.distributed.profiler.gpu import GPUProfiler

    stats = GPUProfiler.profile_gpus()
    print("\n--- GPU Profiler Audit ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def _handle_validate_distributed() -> None:
    from engine.distributed.validator import DistributedValidator

    audit = DistributedValidator.validate_setup()
    print("\n--- Distributed Setup Validation ---")
    for k, v in audit.items():
        print(f"  {k}: {v}")


def _handle_export_distributed_config(args: Any) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    yaml_path = os.path.join(args.out_dir, "distributed.yaml")
    default_cfg = {
        "backend": "gloo",
        "strategy": "ddp",
        "world_size": 1,
        "mixed_precision": "bf16",
        "compile": False,
        "gradient_checkpointing": False,
        "flash_attention": "auto",
        "checkpoint_interval": 1000,
        "tensorboard": True,
        "csv_logging": True,
        "master_addr": "127.0.0.1",
        "master_port": 29500,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(default_cfg, f, default_flow_style=False)
    print(f"Distributed config exported successfully to: {yaml_path}")


def _handle_monitor_training() -> None:
    from engine.distributed.monitoring.metrics import TrainingMetricsTracker

    tracker = TrainingMetricsTracker()
    # Mock single step monitor update
    stats = tracker.step(loss=1.45, lr=1e-4, tokens=1024, examples=2)
    print("\n--- Monitor Training Status ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def _handle_generate(args: Any) -> None:
    from engine.inference.config import GenerationConfig
    from engine.inference.service import InferenceService

    service = InferenceService()
    service.load_model(args.checkpoint_dir)
    cfg = GenerationConfig(
        config_dict={"temperature": args.temperature, "max_new_tokens": args.max_tokens}
    )
    resp = service.generate(args.prompt, config=cfg)
    print(f"\nResponse:\n{resp.text}")
    sys.exit(0)


def _handle_chat(args: Any) -> None:
    from engine.inference.config import GenerationConfig
    from engine.inference.service import InferenceService

    service = InferenceService()
    service.load_model(args.checkpoint_dir)
    cfg = GenerationConfig(
        config_dict={"temperature": args.temperature, "max_new_tokens": args.max_tokens}
    )
    session = service.new_session(is_chat=True)
    print("Interactive Chat Mode initialized. Type 'exit' to end chat.")
    while True:
        try:
            user_text = input("User: ")
            if user_text.strip().lower() in ["exit", "quit"]:
                break
            session.add_message("user", user_text)
            resp = service.chat(session, config=cfg)
            print(f"Assistant: {resp.text}")
        except (KeyboardInterrupt, EOFError):
            break
    sys.exit(0)


def _handle_stream(args: Any) -> None:
    from engine.inference.config import GenerationConfig
    from engine.inference.service import InferenceService

    service = InferenceService()
    service.load_model(args.checkpoint_dir)
    cfg = GenerationConfig(
        config_dict={"temperature": args.temperature, "max_new_tokens": args.max_tokens}
    )
    print("Streaming response:")
    for token in service.stream(args.prompt, config=cfg):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()
    sys.exit(0)


def _handle_benchmark_inference(args: Any) -> None:
    from engine.inference.benchmark import InferenceBenchmarkSuite
    from engine.inference.service import InferenceService
    from engine.inference.visualizer import InferenceVisualizer

    service = InferenceService()
    service.load_model(args.checkpoint_dir)
    suite = InferenceBenchmarkSuite(service)
    results = suite.run_benchmark(max_new_tokens=args.max_tokens)
    print(InferenceVisualizer.render_results(results))
    sys.exit(0)


def _handle_inspect_cache(args: Any) -> None:
    from engine.inference.service import InferenceService

    service = InferenceService()
    session = service.new_session(session_id="default_inspect", is_chat=False)
    print("\n--- Inspect Active Caches ---")
    print(f"Max Cached Sessions: {service.cache_manager.max_cached_sessions}")
    print(f"Active Cache IDs: {list(service.cache_manager.cached_sessions.keys())}")
    if session.cache:
        print(f"KV Cache layers allocation: {getattr(session.cache, 'num_layers', 12)}")
    sys.exit(0)


def _handle_validate_inference(args: Any) -> None:
    from engine.inference.health import InferenceHealthCheck
    from engine.inference.service import InferenceService
    from engine.inference.validator import InferenceValidator

    service = InferenceService()
    service.load_model(args.checkpoint_dir)
    checker = InferenceHealthCheck(service)
    health = checker.check_health()
    print("\n--- Inference Engine Health Check ---")
    for k, v in health.items():
        print(f"  {k}: {v}")
    if args.checkpoint_dir:
        validation = InferenceValidator.validate_checkpoint(args.checkpoint_dir)
        print("\n--- Checkpoint Validation ---")
        for k, v in validation.items():
            print(f"  {k}: {v}")
    sys.exit(0)


def _handle_profile_inference(args: Any) -> None:
    from engine.inference.profiler import InferenceProfiler
    from engine.inference.service import InferenceService

    service = InferenceService()
    service.load_model(args.checkpoint_dir)
    profiler = InferenceProfiler(service)
    profile = profiler.profile_request(args.prompt, max_new_tokens=args.max_tokens)
    print("\n--- Inference Profiler Output ---")
    for k, v in profile.items():
        print(f"  {k}: {v}")
    sys.exit(0)


def _handle_export_generation_config(args: Any) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    yaml_path = os.path.join(args.out_dir, "generation.yaml")
    default_cfg = {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_new_tokens": 256,
        "do_sample": True,
        "use_cache": True,
        "seed": 42,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(default_cfg, f, default_flow_style=False)
    print(f"Generation config exported successfully to: {yaml_path}")
    sys.exit(0)


def _setup_inference_parsers(subparsers: Any) -> None:
    # generate
    gen = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen.add_argument("--prompt", required=True, help="Input prompt text")
    gen.add_argument("--checkpoint-dir", default=None, help="Path to checkpoint folder")
    gen.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    gen.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens")

    # chat
    chat = subparsers.add_parser("chat", help="Run interactive console chat")
    chat.add_argument(
        "--checkpoint-dir", default=None, help="Path to checkpoint folder"
    )
    chat.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    chat.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens")

    # stream
    stream = subparsers.add_parser("stream", help="Stream generated tokens to stdout")
    stream.add_argument("--prompt", required=True, help="Input prompt text")
    stream.add_argument(
        "--checkpoint-dir", default=None, help="Path to checkpoint folder"
    )
    stream.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    stream.add_argument(
        "--max-tokens", type=int, default=128, help="Maximum new tokens"
    )

    # benchmark-inference
    bench = subparsers.add_parser(
        "benchmark-inference", help="Benchmark generation speed and TTFT"
    )
    bench.add_argument(
        "--checkpoint-dir", default=None, help="Path to checkpoint folder"
    )
    bench.add_argument(
        "--max-tokens", type=int, default=64, help="Maximum new tokens per prompt"
    )

    # inspect-cache
    subparsers.add_parser(
        "inspect-cache", help="Inspect active KV cache slots and memory layout"
    )

    # validate-inference
    val = subparsers.add_parser(
        "validate-inference", help="Validate inference health and loaded checkpoints"
    )
    val.add_argument("--checkpoint-dir", default=None, help="Path to checkpoint folder")

    # profile-inference
    prof = subparsers.add_parser(
        "profile-inference",
        help="Profile inference latencies and peak memory allocations",
    )
    prof.add_argument(
        "--prompt",
        default="Explain transformer self-attention.",
        help="Profiling prompt",
    )
    prof.add_argument(
        "--checkpoint-dir", default=None, help="Path to checkpoint folder"
    )
    prof.add_argument("--max-tokens", type=int, default=64, help="Maximum new tokens")

    # export-generation-config
    exp = subparsers.add_parser(
        "export-generation-config", help="Export default generation configuration YAML"
    )
    exp.add_argument("--out-dir", default="configs", help="Output directory path")


def _handle_serve(args: Any) -> None:
    import uvicorn
    import yaml

    with open("configs/api.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    host = args.host or cfg.get("server", {}).get("host", "0.0.0.0")
    port = args.port or cfg.get("server", {}).get("port", 8000)
    print(f"Launching Chatii API server on {host}:{port}...")
    uvicorn.run("engine.api.api:app", host=host, port=port, reload=False)
    sys.exit(0)


def _handle_warmup(args: Any) -> None:
    print("Evaluating model warmup...")
    sys.exit(0)


def _handle_shutdown_cli(args: Any) -> None:
    print("Initiating cluster shutdown...")
    sys.exit(0)


def _handle_config_cli(args: Any) -> None:
    import yaml

    with open("configs/api.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(yaml.dump(cfg, default_flow_style=False))
    sys.exit(0)


def _handle_version(args: Any) -> None:
    print("Chatii Production Platform v1.0.0")
    sys.exit(0)


def _handle_metrics_cli(args: Any) -> None:
    import requests

    try:
        resp = requests.get("http://localhost:8000/metrics", timeout=5)
        print(resp.text)
    except Exception as e:
        print(f"Error fetching metrics: {e}")
    sys.exit(0)


def _handle_health_cli(args: Any) -> None:
    import requests

    try:
        resp = requests.get("http://localhost:8000/health/ready", timeout=5)
        print(resp.json())
    except Exception as e:
        print(f"Error checking health: {e}")
    sys.exit(0)


def _handle_benchmark_api(args: Any) -> None:
    from engine.api.benchmark.api_benchmark import run_api_benchmark

    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Authorization": "Bearer chatii-admin-root-key"}
    payload = {
        "model": "chatii-v1",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 16,
    }
    print("Running API Benchmark...")
    res = run_api_benchmark(url, headers, payload, count=5)
    print(f"Average Latency: {res['avg_latency']:.3f}s")
    print(f"Throughput: {res['tps']:.2f} requests/sec")
    sys.exit(0)


def _handle_load_model_cli(args: Any) -> None:
    import requests

    try:
        resp = requests.post(
            "http://localhost:8000/v1/admin/load",
            json={"model": args.model},
            headers={"Authorization": "Bearer chatii-admin-root-key"},
            timeout=10,
        )
        print(resp.json())
    except Exception as e:
        print(f"Error loading model: {e}")
    sys.exit(0)


def _handle_unload_model_cli(args: Any) -> None:
    import requests

    try:
        resp = requests.post(
            "http://localhost:8000/v1/admin/unload",
            json={"model": args.model},
            headers={"Authorization": "Bearer chatii-admin-root-key"},
            timeout=10,
        )
        print(resp.json())
    except Exception as e:
        print(f"Error unloading model: {e}")
    sys.exit(0)


def _handle_reload_model_cli(args: Any) -> None:
    import requests

    try:
        headers = {"Authorization": "Bearer chatii-admin-root-key"}
        requests.post(
            "http://localhost:8000/v1/admin/unload",
            json={"model": args.model},
            headers=headers,
            timeout=10,
        )
        resp = requests.post(
            "http://localhost:8000/v1/admin/load",
            json={"model": args.model},
            headers=headers,
            timeout=10,
        )
        print(resp.json())
    except Exception as e:
        print(f"Error reloading model: {e}")
    sys.exit(0)


def _handle_list_models_cli(args: Any) -> None:
    import requests

    try:
        resp = requests.get(
            "http://localhost:8000/v1/models",
            headers={"Authorization": "Bearer chatii-admin-root-key"},
            timeout=5,
        )
        print(resp.json())
    except Exception as e:
        print(f"Error listing models: {e}")
    sys.exit(0)


def _setup_api_parsers(subparsers: Any) -> None:
    for cmd in ["serve", "serve-api", "serve-websocket", "serve-sse"]:
        p = subparsers.add_parser(cmd, help=f"Serve the model using {cmd}")
        p.add_argument("--host", default=None, help="Server host IP")
        p.add_argument("--port", type=int, default=None, help="Server port number")

    subparsers.add_parser("warmup", help="Warm up active models")
    subparsers.add_parser("shutdown", help="Initiate graceful shutdown")
    subparsers.add_parser("config", help="Print active API configuration")
    subparsers.add_parser("version", help="Print Chatii platform version")
    subparsers.add_parser("metrics", help="Fetch remote API server metrics")
    subparsers.add_parser("health", help="Check remote API server health")
    subparsers.add_parser("benchmark-api", help="Benchmark Remote API server speed")

    for cmd in ["load-model", "unload-model", "reload-model"]:
        p = subparsers.add_parser(cmd, help=f"{cmd} command")
        p.add_argument("--model", required=True, help="Target model identifier")

    subparsers.add_parser("list-models", help="List registered models")


def main() -> None:
    """
    Unified CLI entrypoint for Chatii execution commands.
    """
    parser = argparse.ArgumentParser(
        prog="chatii",
        description="Chatii: A Production-Grade Modular Transformer Framework.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Operation subcommands")

    # Command: train
    subparsers.add_parser(
        "train", help="Train a model configured in configs/training.yaml"
    )

    # Command: benchmark
    subparsers.add_parser(
        "benchmark", help="Run latency, throughput, and memory benchmarks"
    )

    # Command: evaluate
    subparsers.add_parser(
        "evaluate", help="Evaluate model performance against benchmark datasets"
    )

    _setup_tokenizer_parsers(subparsers)
    _setup_math_parsers(subparsers)
    _setup_model_parsers(subparsers)
    _setup_training_parsers(subparsers)
    _setup_dataset_parsers(subparsers)
    _setup_distributed_parsers(subparsers)
    _setup_inference_parsers(subparsers)
    _setup_api_parsers(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        handlers = {
            "list-tokenizers": lambda: _handle_list_tokenizers(),
            "train-tokenizer": lambda: _handle_train_tokenizer(args),
            "encode": lambda: _handle_encode(args),
            "decode": lambda: _handle_decode(args),
            "validate-tokenizer": lambda: _handle_validate_tokenizer(args),
            "inspect-tokenizer": lambda: _handle_inspect_tokenizer(args),
            "benchmark-tokenizer": lambda: _handle_benchmark_tokenizer(args),
            "profile-tokenizer": lambda: _handle_profile_tokenizer(args),
            "visualize-tokenizer": lambda: _handle_visualize_tokenizer(args),
            "export-tokenizer": lambda: _handle_export_import(),
            "import-tokenizer": lambda: _handle_export_import(),
            "inspect-math": lambda: _handle_inspect_math(),
            "benchmark-math": lambda: _handle_benchmark_math(args),
            "visualize-rope": lambda: _handle_visualize_rope(),
            "test-normalization": lambda: _handle_test_normalization(),
            "train-forward-demo": lambda: _handle_train_forward_demo(args),
            "inspect-model": lambda: _handle_inspect_model(),
            "benchmark-transformer": lambda: _handle_benchmark_transformer(args),
            "validate-model": lambda: _handle_validate_model(),
            "count-parameters": lambda: _handle_count_parameters(),
            "visualize-attention": lambda: _handle_visualize_attention(args),
            "train-model": lambda: _handle_train_model(args),
            "resume-training": lambda: _handle_resume_training(args),
            "evaluate-model": lambda: _handle_evaluate_model(args),
            "benchmark-training": lambda: _handle_benchmark_training(args),
            "profile-training": lambda: _handle_profile_training(args),
            "inspect-checkpoint": lambda: _handle_inspect_checkpoint(args),
            "export-training-config": lambda: _handle_export_training_config(args),
            "estimate-training-time": lambda: _handle_estimate_training_time(args),
            "prepare-dataset": lambda: _handle_prepare_dataset(args),
            "inspect-dataset": lambda: _handle_inspect_dataset(args),
            "validate-dataset": lambda: _handle_validate_dataset(args),
            "benchmark-dataset": lambda: _handle_benchmark_dataset(args),
            "cache-dataset": lambda: _handle_cache_dataset(args),
            "tokenize-dataset": lambda: _handle_tokenize_dataset(args),
            "pack-dataset": lambda: _handle_pack_dataset(args),
            "stream-dataset": lambda: _handle_stream_dataset(args),
            "dataset-statistics": lambda: _handle_dataset_statistics(args),
            "list-datasets": lambda: _handle_list_datasets(),
            "register-dataset": lambda: _handle_register_dataset(args),
            "remove-dataset": lambda: _handle_remove_dataset(args),
            "dataset-info": lambda: _handle_dataset_info(args),
            "train-distributed": lambda: _handle_train_distributed(args),
            "resume-distributed": lambda: _handle_resume_distributed(args),
            "benchmark-distributed": lambda: _handle_benchmark_distributed(),
            "profile-distributed": lambda: _handle_profile_distributed(),
            "inspect-cluster": lambda: _handle_inspect_cluster(),
            "inspect-gpu": lambda: _handle_inspect_gpu(),
            "validate-distributed": lambda: _handle_validate_distributed(),
            "export-distributed-config": lambda: _handle_export_distributed_config(
                args
            ),
            "monitor-training": lambda: _handle_monitor_training(),
            "generate": lambda: _handle_generate(args),
            "chat": lambda: _handle_chat(args),
            "stream": lambda: _handle_stream(args),
            "benchmark-inference": lambda: _handle_benchmark_inference(args),
            "inspect-cache": lambda: _handle_inspect_cache(args),
            "validate-inference": lambda: _handle_validate_inference(args),
            "profile-inference": lambda: _handle_profile_inference(args),
            "export-generation-config": lambda: _handle_export_generation_config(args),
            "serve": lambda: _handle_serve(args),
            "serve-api": lambda: _handle_serve(args),
            "serve-websocket": lambda: _handle_serve(args),
            "serve-sse": lambda: _handle_serve(args),
            "warmup": lambda: _handle_warmup(args),
            "shutdown": lambda: _handle_shutdown_cli(args),
            "config": lambda: _handle_config_cli(args),
            "version": lambda: _handle_version(args),
            "metrics": lambda: _handle_metrics_cli(args),
            "health": lambda: _handle_health_cli(args),
            "benchmark-api": lambda: _handle_benchmark_api(args),
            "load-model": lambda: _handle_load_model_cli(args),
            "unload-model": lambda: _handle_unload_model_cli(args),
            "reload-model": lambda: _handle_reload_model_cli(args),
            "list-models": lambda: _handle_list_models_cli(args),
        }
        if args.command in handlers:
            handlers[args.command]()

    except Exception as e:
        print(f"Execution Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
