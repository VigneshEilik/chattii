import os
import subprocess
import sys

def main() -> None:
    print("--- Chatii Training Engine Live Demonstration ---")

    # 1. Export configuration files
    print("\nStep 1: Exporting default training configurations...")
    subprocess.run(
        [sys.executable, "cli/main.py", "export-training-config", "--out-dir", "configs"],
        check=True
    )

    # 2. Setup demo dataset file
    print("\nStep 2: Preparing raw training dataset...")
    train_file = "datasets/demo/tiny.txt"
    os.makedirs("datasets/demo", exist_ok=True)
    if not os.path.exists(train_file):
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("Chatii is a production-grade modular Transformer framework.\n" * 50)

    # 3. Launch training with a tiny config override for speed
    print("\nStep 3: Starting model training (10 steps)...")
    with open("configs/training.yaml", "w", encoding="utf-8") as f:
        f.write("""seed: 42
batch_size: 4
gradient_accumulation_steps: 1
max_epochs: 2
max_steps: 10
mixed_precision: none
grad_clip: 1.0
device: cpu
save_interval: 5
checkpoint_dir: models/checkpoints
keep_top_n: 2
eval_interval: 5
early_stopping_patience: 3
tensorboard: false
csv: true
console: true
dashboard: false
""")

    subprocess.run(
        [sys.executable, "cli/main.py", "train-model", "--train-file", train_file],
        check=True
    )

    # 4. Check if checkpoints exist
    print("\nStep 4: Verifying saved checkpoints...")
    checkpoint_dir = "models/checkpoints"
    if os.path.exists(checkpoint_dir):
        print(f"Checkpoints saved: {os.listdir(checkpoint_dir)}")

    # 5. Inspect saved checkpoint
    print("\nStep 5: Inspecting checkpoint...")
    subprocess.run(
        [sys.executable, "cli/main.py", "inspect-checkpoint", "--checkpoint-dir", "models/checkpoints/latest"],
        check=True
    )

    # 6. Estimate training completion time
    print("\nStep 6: Estimating training time...")
    subprocess.run(
        [sys.executable, "cli/main.py", "estimate-training-time", "--train-file", train_file, "--epochs", "5"],
        check=True
    )

    print("\nDemo Completed Successfully!")

if __name__ == "__main__":
    main()
