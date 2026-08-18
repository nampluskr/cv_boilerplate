import logging
import os

from src.utils.io import append_csv_row

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(run_dir, log_level="INFO", name="src"):
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(run_dir, "train.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class MetricsCsvWriter:
    """Appends one row per (epoch, split) to metrics_epoch.csv."""

    def __init__(self, run_dir, metric_names):
        self.path = os.path.join(run_dir, "metrics_epoch.csv")
        self.fieldnames = ["epoch", "split", "loss"] + list(metric_names) + ["lr", "elapsed_sec"]

    def write(self, epoch, split, loss, metrics, lr, elapsed_sec):
        row = {"epoch": epoch, "split": split, "loss": loss, "lr": lr, "elapsed_sec": elapsed_sec}
        for name in self.fieldnames:
            if name in metrics:
                row[name] = metrics[name]
        for name in self.fieldnames:
            row.setdefault(name, "")
        append_csv_row(self.path, row, self.fieldnames)
