from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    HEIGHT: int = 224
    WIDTH: int = 224
    CHANNELS: int = 3
    NUM_CLASSES: int = 4
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 16
    EPOCHS: int = 5
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    NUM_WORKERS: int = 0
    BREAK_AFTER_IT = None
    ROOT_DIR: str = "../"
