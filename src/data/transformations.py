import grain.python as grain
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.types import ForecasterInput


class StandardScalingTransform(grain.MapTransform):
    def __init__(self, feature_scaler: StandardScaler, target_scaler: StandardScaler):
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    def map(self, element):
        context, target = element
        x_scaled = self.feature_scaler.transform(context)
        y_scaled = self.target_scaler.transform(target)
        return (x_scaled, y_scaled)


class DeviceMaskingTransform(grain.RandomMapTransform):
    def __init__(self, mask_prob: float):
        assert 0.0 <= mask_prob <= 1.0, (
            f"The mask's probability {mask_prob} must be in (0, 1)"
        )
        self.mask_prob = mask_prob

    def random_map(self, element: tuple[np.ndarray, np.ndarray], rng):
        series, targets = element
        B, D, M, T = series.shape

        # 1 = mask this device, 0 = keep it
        mask = rng.binomial(n=1, p=self.mask_prob, size=(B, D))
        mask = mask[:, :, None, None]  # (B, D, 1, 1) -> broadcast over metrics and time

        # Zero out masked devices
        masked_series = series * (1 - mask)

        return ForecasterInput(
            series=masked_series,
            targets=targets,
            mask=mask.astype(np.bool_),
        )


class FeatureMaskingTransform(grain.RandomMapTransform):
    def __init__(self, mask_prob: float):
        assert 0.0 <= mask_prob <= 1.0, (
            f"The mask's probability {mask_prob} must be in (0, 1)"
        )
        self.mask_prob = mask_prob

    def random_map(self, element: tuple[np.ndarray, np.ndarray], rng):
        series, target = element
        B, D, M, T = series.shape

        # mask individual metrics independently across devices and time
        mask = rng.binomial(n=1, p=self.mask_prob, size=(B, D, M))
        mask = mask[:, :, :, None]  # (B, D, M, 1) -> broadcast over time

        # zero out masked features
        masked_series = series * (1 - mask)

        return ForecasterInput(
            series=masked_series,
            targets=target,
            mask=mask.astype(np.bool_),
        )
