class TrainingEvents:
    """Standardized event keys fired across the training lifecycle."""

    TRAIN_START = "train_start"
    TRAIN_END = "train_end"

    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"

    STEP_START = "step_start"
    STEP_END = "step_end"

    VAL_START = "val_start"
    VAL_END = "val_end"

    CHECKPOINT_SAVE = "checkpoint_save"
