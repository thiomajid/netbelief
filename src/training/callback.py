import io
import random
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from huggingface_hub import create_repo, repo_exists, upload_folder
from PIL import Image

from src.modules.lstm import LSTMForecaster
from src.training.arguments import TrainingConfig
from src.training.module import checkpoint_post_eval
from src.training.state import TrainerState
from src.training.tensorboard import TensorBoardLogger
from src.utils.types import LoguruLogger
from src.utils.viz import plot_forecast


class Callback:
    def __init__(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self, state: TrainerState, metrics: dict):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass


class CheckpointCallback(Callback):
    def __init__(
        self,
        model: nnx.Module,
        options: ocp.CheckpointManagerOptions,
        checkpoint_dir: Path,
        logger: LoguruLogger,
    ):
        super().__init__()

        self.model = model
        self.logger = logger
        self.options = options
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, state, metrics):
        checkpoint_post_eval(
            logger=self.logger,
            model=self.model,
            checkpoint_dir=self.checkpoint_dir,
            options=self.options,
            metrics=metrics,
            global_step=state.current_step,
            epoch=state.epoch,
        )


class PushToHubCallback(Callback):
    def __init__(self, logger: LoguruLogger, args: TrainingConfig):
        super().__init__()

        self.logger = logger
        self.args = args

    def on_train_end(self):
        folder_path = Path(self.args.output_dir)
        self.logger.info(
            f"Pushing artifacts from {str(folder_path)} to Hugging Face Hub repository: {self.args.hub_model_id}..."
        )

        if not repo_exists(self.args.hub_model_id, token=self.args.hub_token):
            self.logger.info(f"Creating repository {self.args.hub_model_id}...")
            create_repo(
                repo_id=self.args.hub_model_id,
                token=self.args.hub_token,
                private=self.args.hub_private_repo,
                exist_ok=True,
            )

        upload_folder(
            repo_id=self.args.hub_model_id,
            folder_path=folder_path,
            token=self.args.hub_token,
            commit_message=self.args.upload_message,
        )
        self.logger.info(
            f"Push to Hub completed. Results can be viewed at https://huggingface.co/{self.args.hub_model_id}"
        )


class PlotForecastCallback(Callback):
    def __init__(
        self,
        model: LSTMForecaster,
        series: jax.Array,
        targets: jax.Array,
        reporter: TensorBoardLogger,
    ):
        super().__init__()
        self.model = model
        self.series = series
        self.targets = targets
        self.reporter = reporter

    def on_epoch_end(self, state: TrainerState, metrics: dict):
        output = self.model(self.series)
        batch, devices, num_metrics, _ = output.point_predictions.shape

        batch_idx = random.randint(0, batch - 1)
        device_idx = random.randint(0, devices - 1)
        metric_idx = random.randint(0, num_metrics - 1)

        context = self.series[batch_idx, device_idx, metric_idx, :]
        predicted = output.point_predictions[batch_idx, device_idx, metric_idx, :]
        q_forecast = output.quantile_predictions[
            :, batch_idx, device_idx, metric_idx, :
        ]
        gt_series = self.targets[batch_idx, device_idx, metric_idx, :]

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_forecast(
            context=context,
            forecasts=predicted,
            ground_truth=gt_series,
            ax=ax,
            title=f"Forecast - Device {device_idx}, Metric {metric_idx}",
            quantiles=self.model.quantiles,
            quantile_forecasts=q_forecast,
        )

        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            img = np.array(Image.open(buffer).convert("RGB"))

            self.reporter.log_figure(
                tag=f"Forecast/Device_{device_idx}_Metric_{metric_idx}",
                figure=img,
                step=state.current_step,
            )

        plt.close(fig)

        del fig
        del ax
