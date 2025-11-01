"""
Training Logger Utility
Handles TensorBoard and console logging for RL training
"""

import os
from typing import Dict, Any, Optional
import time
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainingLogger:
    """
    Unified logger for RL training metrics

    Supports:
    - TensorBoard logging
    - Weights & Biases logging
    - Console output
    - CSV file logging
    """

    def __init__(
        self,
        log_dir: str = "data/logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize logger

        Args:
            log_dir: Directory for logs
            use_tensorboard: Enable TensorBoard
            use_wandb: Enable Weights & Biases
            wandb_project: W&B project name
            wandb_config: W&B config dict
            experiment_name: Name for this experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_name = f"run_{timestamp}"

        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard setup
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.tb_writer = None
        if self.use_tensorboard:
            tb_dir = self.experiment_dir / "tensorboard"
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard logging to: {tb_dir}")
            print(f"  View with: tensorboard --logdir {tb_dir}")

        # Weights & Biases setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if wandb_project is None:
                wandb_project = "drone-rl"

            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=wandb_config or {},
                dir=str(self.experiment_dir)
            )
            print(f"W&B logging to project: {wandb_project}")

        # CSV logging
        self.csv_path = self.experiment_dir / "metrics.csv"
        self.csv_header_written = False

        # Metrics buffer
        self.current_step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value

        Args:
            tag: Metric name
            value: Metric value
            step: Training step (uses internal counter if None)
        """
        if step is None:
            step = self.current_step

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)

        # Weights & Biases
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple scalars at once

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step
        """
        if step is None:
            step = self.current_step

        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

        # Also log to CSV
        self._log_to_csv(metrics, step)

    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log a histogram of values"""
        if step is None:
            step = self.current_step

        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, step)

        if self.use_wandb:
            wandb.log({tag: wandb.Histogram(values)}, step=step)

    def _log_to_csv(self, metrics: Dict[str, Any], step: int):
        """Log metrics to CSV file"""
        # Write header if first time
        if not self.csv_header_written:
            with open(self.csv_path, 'w') as f:
                f.write("step," + ",".join(metrics.keys()) + "\n")
            self.csv_header_written = True

        # Write data
        with open(self.csv_path, 'a') as f:
            values = [str(metrics.get(k, '')) for k in metrics.keys()]
            f.write(f"{step}," + ",".join(values) + "\n")

    def log_config(self, config: Dict[str, Any]):
        """Log configuration dictionary"""
        if self.tb_writer is not None:
            # Log as text
            config_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
            self.tb_writer.add_text("config", config_str)

        if self.use_wandb:
            wandb.config.update(config)

        # Save to file
        import yaml
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def print_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Print metrics to console"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"{prefix}{metrics_str}")

    def increment_step(self):
        """Increment internal step counter"""
        self.current_step += 1

    def close(self):
        """Close all loggers"""
        if self.tb_writer is not None:
            self.tb_writer.close()

        if self.use_wandb:
            wandb.finish()

        print(f"Logs saved to: {self.experiment_dir}")


if __name__ == "__main__":
    # Test logger
    print("Testing TrainingLogger...")

    logger = TrainingLogger(
        log_dir="data/logs",
        use_tensorboard=True,
        use_wandb=False,
        experiment_name="test_run"
    )

    # Log some test metrics
    for step in range(100):
        metrics = {
            'episode_reward': 10.0 + step * 0.5,
            'policy_loss': 1.0 / (step + 1),
            'value_loss': 0.5 / (step + 1),
        }

        logger.log_scalars(metrics, step=step)

        if step % 10 == 0:
            logger.print_metrics(metrics, prefix=f"Step {step}: ")

    logger.close()
    print("Test complete!")
