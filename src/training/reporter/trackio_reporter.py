"""Trackio metrics reporter for Hugging Face tracking."""

import trackio

from src.training.arguments import TrainingConfig
from src.training.reporter.base_reporter import MetricsReporter
from src.utils.types import LoguruLogger


class TrackioReporter(MetricsReporter):
    """Metrics reporter using Hugging Face trackio."""

    def __init__(
        self,
        log_dir: str,
        logger: LoguruLogger,
        args: TrainingConfig,
    ):
        super().__init__(log_dir, logger, args.run_name)

        trackio.init(
            project=args.trackio.project,
            name=args.trackio.run_name,
            group=args.trackio.group,
        )

    def log_scalar(self, tag, value, step):
        trackio.log({tag: self.maybe_scalar(value)})

    def log_scalars(self, tag_scalar_dict, step):
        data = {k: self.maybe_scalar(v) for k, v in tag_scalar_dict.items()}
        trackio.log(data)

    def log_figure(self, tag, figure, step):
        trackio.log({tag: trackio.Image(figure)})

    def log_learning_rate(self, lr, step):
        self.log_scalar("train/learning_rate", lr, step)

    def close(self):
        trackio.finish()
