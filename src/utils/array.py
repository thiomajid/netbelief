import jax

from src.utils.types import LoguruLogger


def log_node_devices_stats(logger: LoguruLogger):
    devices = jax.devices()
    logger.info(f"Found {len(devices)} devices.")

    for device in devices:
        try:
            stats = device.memory_stats()
            # Collect memory stats in a compact format
            stat_items = []
            for key, value in stats.items():
                if "bytes" in key:
                    value_gb = value / (1024**3)
                    stat_items.append(f"{key}: {value_gb:.2f} GB")
                else:
                    stat_items.append(f"{key}: {value}")
            logger.info(f"{device}: " + "; ".join(stat_items))
        except Exception as e:
            logger.warning(f"Could not get memory stats for {device}: {e}")
