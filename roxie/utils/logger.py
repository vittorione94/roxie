"""Logging system, metric aggregation, and output backends."""

import datetime
import os
import subprocess
import time

import numpy as np
import termcolor

current_logger = None

_gpu_stats_enabled = True
_GPU_QUERY_FIELDS = (
    "utilization.gpu",
    "temperature.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
)


def gpu_stats(prefix="sys/gpu"):
    """Samples per-GPU telemetry via nvidia-smi as a metric dictionary.

    Args:
        prefix: Namespace prefix for the returned metric keys.

    Returns:
        A dictionary mapping GPU metric keys to float values, or {} if no GPU is available.
    """
    global _gpu_stats_enabled
    if not _gpu_stats_enabled:
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        _gpu_stats_enabled = False
        return {}
    if out.returncode != 0:
        _gpu_stats_enabled = False
        return {}

    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    multi = len(lines) > 1
    stats = {}
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_GPU_QUERY_FIELDS):
            continue

        def _f(x):
            try:
                return float(x)
            except ValueError:
                return None

        util, temp, mem_used, mem_total, power = (_f(p) for p in parts)
        base = f"{prefix}/{i}" if multi else prefix
        fields = {
            "util_pct": util,
            "temp_c": temp,
            "mem_used_mb": mem_used,
            "mem_pct": (100.0 * mem_used / mem_total
                        if mem_used is not None and mem_total else None),
            "power_w": power,
        }
        for name, val in fields.items():
            if val is not None:
                stats[f"{base}/{name}"] = val
    return stats


def log(msg, color="green"):
    """Prints a formatted info message to stdout."""
    print(termcolor.colored(msg, color, attrs=["bold"]))


def warning(msg, color="yellow"):
    """Prints a formatted warning message to stdout."""
    print(termcolor.colored("Warning: " + msg, color, attrs=["bold"]))


def error(msg, color="red"):
    """Prints a formatted error message to stdout."""
    print(termcolor.colored("Error: " + msg, color, attrs=["bold"]))


class Backend:
    """Abstract base class for metric sinks."""

    def log(self, data, step):
        """Logs aggregated metric data for a given step."""
        raise NotImplementedError

    def close(self):
        """Closes resources associated with the backend."""
        pass


class ConsoleBackend(Backend):
    """Pretty-prints metrics to stdout as an indented, hierarchical table."""

    INDENT = "  "
    TOP_KEYS = ("epoch", "steps")

    ORDER = (
        "epoch",
        "steps",
        "train",
        "train/score",
        "train/length",
        "train/episodes",
        "train/gradient_steps",
        "train/loss",
        "train/reward",
        "train/root_dist",
        "train/noise",
        "test",
        "test/score",
        "test/length",
        "test/distinct_starts",
        "sys",
        "sys/sps",
        "sys/time",
        "sys/gpu",
        "sys/mem",
    )

    def __init__(self, width=60):
        self.width = width
        self.known_keys = set()
        self.rows = []

    def _display_path(self, key):
        return key

    def _sort_key(self, path):
        segs = path.split("/")
        key = []
        for i in range(len(segs)):
            prefix = "/".join(segs[: i + 1])
            rank = self.ORDER.index(prefix) if prefix in self.ORDER else len(self.ORDER)
            key.append((rank, segs[i]))
        return key

    def _rebuild_layout(self, keys):
        keys = set(keys)
        std_of = {k[:-4]: k for k in keys if k.endswith("/std") and k[:-4] in keys}
        merged = set(std_of.values())

        leaves = [k for k in keys if k not in merged]
        display = {k: self._display_path(k) for k in leaves}
        leaves.sort(key=lambda k: self._sort_key(display[k]))

        self.rows = []
        seen = set()
        for k in leaves:
            *parents, leaf = display[k].split("/")
            for i, name in enumerate(parents):
                prefix = "/".join(parents[: i + 1])
                if prefix not in seen:
                    seen.add(prefix)
                    self.rows.append((self.INDENT * i + name.replace("_", " "), None, None))
            indent = self.INDENT * len(parents)
            self.rows.append((indent + leaf.replace("_", " "), k, std_of.get(k)))

    @staticmethod
    def _fmt(val, pad=False):
        if val is None:
            return f"{'-':>8}" if pad else "-"
        if np.issubdtype(type(val), np.floating):
            return f"{val:8.3g}" if pad else f"{val:.3g}"
        if np.issubdtype(type(val), np.integer):
            return f"{val:,}"
        return str(val)

    def log(self, data, step):
        """Logs formatted table rows to stdout."""
        new_keys = [key for key in data if key not in self.known_keys]
        if new_keys:
            first_row = len(self.known_keys) == 0
            if not first_row:
                print()
                warning(f"Logging new keys {new_keys}")
            self.known_keys.update(data.keys())
            self._rebuild_layout(self.known_keys)

        print()
        for label, key, std_key in self.rows:
            if key is None:
                print(label + " " * max(self.width - len(label), 0))
                continue
            val = data.get(key)
            str_type = str(type(val))
            if "tensorflow" in str_type:
                warning(f"Logging TensorFlow tensor {key}")
            elif "torch" in str_type:
                warning(f"Logging Torch tensor {key}")
            std_val = data.get(std_key) if std_key is not None else None
            if std_val is not None:
                right = f"{self._fmt(val)} +- {self._fmt(std_val)}"
            else:
                right = self._fmt(val, pad=True)
            print(label + " " * max(self.width - len(label) - len(right), 1) + right)
        print()


class CSVBackend(Backend):
    """Appends metric rows to a CSV file, adjusting columns dynamically."""

    def __init__(self, path):
        self.log_file_path = path
        self.known_keys = set()
        self.final_keys = []

    def log(self, data, step):
        """Logs a row of metrics to the CSV file."""
        new_keys = [key for key in data if key not in self.known_keys]
        first_row = len(self.known_keys) == 0
        if new_keys:
            for key in new_keys:
                self.known_keys.add(key)
            self.final_keys = list(sorted(self.known_keys))

        vals = [data.get(key) for key in self.final_keys]
        if new_keys:
            if first_row:
                log(f"Logging data to {self.log_file_path}")
                try:
                    os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
                except Exception:
                    pass
                with open(self.log_file_path, "w") as file:
                    file.write(",".join(self.final_keys) + "\n")
                    file.write(",".join(map(str, vals)) + "\n")
            else:
                with open(self.log_file_path, "r") as file:
                    lines = file.read().splitlines()
                old_keys = lines[0].split(",")
                old_lines = [line.split(",") for line in lines[1:]]
                new_indices = []
                j = 0
                for i, key in enumerate(self.final_keys):
                    if j < len(old_keys) and key == old_keys[j]:
                        j += 1
                    else:
                        new_indices.append(i)
                assert j == len(old_keys)
                for line in old_lines:
                    for i in new_indices:
                        line.insert(i, "None")
                with open(self.log_file_path, "w") as file:
                    file.write(",".join(self.final_keys) + "\n")
                    for line in old_lines:
                        file.write(",".join(line) + "\n")
                    file.write(",".join(map(str, vals)) + "\n")
        else:
            with open(self.log_file_path, "a") as file:
                file.write(",".join(map(str, vals)) + "\n")


class WandbBackend(Backend):
    """Streams logged metrics to Weights & Biases."""

    def __init__(self, project=None, entity=None, name=None, group=None,
                 job_type=None, tags=None, mode="online", config=None, dir=None,
                 relogin=True, **kwargs):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled but the 'wandb' package is not "
                "installed. Install it with `uv add wandb` (or disable "
                "logging.wandb.enabled)."
            ) from e
        self._wandb = wandb
        needs_auth = relogin and mode not in ("offline", "disabled") \
            and not os.environ.get("WANDB_API_KEY")
        if needs_auth:
            wandb.login(relogin=True)
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            job_type=job_type,
            tags=list(tags) if tags else None,
            mode=mode,
            config=config,
            dir=dir,
            **kwargs,
        )

    def log(self, data, step):
        """Logs metrics to the active W&B run, filtering out None values."""
        self.run.log({k: v for k, v in data.items() if v is not None}, step=int(step))

    def close(self):
        """Finishes the W&B run session."""
        self.run.finish()


def default_backends(path, width=60):
    """Constructs default Console and CSV logging backends."""
    return [ConsoleBackend(width=width), CSVBackend(os.path.join(path, "log.csv"))]


class Logger:
    """Accumulates and dispatches training metrics to active backends."""

    def __init__(self, path=None, width=60, script_path=None, backends=None):
        self.path = path or str(time.time())

        if script_path:
            with open(script_path, "r") as script_file:
                script = script_file.read()
                try:
                    os.makedirs(self.path, exist_ok=True)
                except Exception:
                    pass
                script_path = os.path.join(self.path, "script.py")
                with open(script_path, "w") as config_file:
                    config_file.write(script)
                log(f"Script file saved to {script_path}")

        self.stat_keys = set()
        self.epoch_dict = {}
        self.width = width
        self.epoch = 0
        self.last_epoch_progress = None
        self.start_time = time.time()
        self.last_epoch_time = self.start_time

        self.backends = default_backends(self.path, width=width) if backends is None \
            else list(backends)

    def add_backend(self, backend):
        """Attaches an additional backend sink."""
        self.backends.append(backend)

    def store(self, key, value, stats=False):
        """Buffers a metric value for the current epoch."""
        if key not in self.epoch_dict:
            self.epoch_dict[key] = [value]
            if stats:
                self.stat_keys.add(key)
        else:
            self.epoch_dict[key].append(value)

    def dump(self, step=None):
        """Aggregates buffered epoch metrics and dispatches them to backends."""
        keys = list(self.epoch_dict.keys())
        for key in keys:
            values = self.epoch_dict[key]
            present = [v for v in values if v is not None]
            if not present:
                self.epoch_dict[key] = None
                continue
            values = present
            if key in self.stat_keys:
                self.epoch_dict[key + "/mean"] = np.mean(values)
                self.epoch_dict[key + "/std"] = np.std(values)
                self.epoch_dict[key + "/min"] = np.min(values)
                self.epoch_dict[key + "/max"] = np.max(values)
                self.epoch_dict[key + "/size"] = len(values)
                del self.epoch_dict[key]
            else:
                self.epoch_dict[key] = np.mean(values)

        data = dict(self.epoch_dict)
        if step is None:
            step = self.epoch

        for backend in self.backends:
            try:
                backend.log(data, step)
            except Exception as exc:
                error(f"Backend {type(backend).__name__} failed to log: {exc}")

        self.epoch += 1
        self.epoch_dict.clear()
        self.last_epoch_progress = None
        self.last_epoch_time = time.time()

    def close(self):
        """Flushes and closes all registered backends."""
        for backend in self.backends:
            try:
                backend.close()
            except Exception as exc:
                error(f"Backend {type(backend).__name__} failed to close: {exc}")

    def show_progress(
        self, steps, num_epoch_steps, num_steps, color="white", on_color="on_blue"
    ):
        """Renders an inline progress bar for the current epoch and total steps."""
        epoch_steps = (steps - 1) % num_epoch_steps + 1
        epoch_progress = int(self.width * epoch_steps / num_epoch_steps)
        if epoch_progress != self.last_epoch_progress:
            current_time = time.time()
            seconds = current_time - self.start_time
            seconds_per_step = seconds / steps
            epoch_rem_steps = num_epoch_steps - epoch_steps
            epoch_rem_secs = max(epoch_rem_steps * seconds_per_step, 0)
            epoch_rem_secs = datetime.timedelta(seconds=epoch_rem_secs + 1e-6)
            epoch_rem_secs = str(epoch_rem_secs)[:-7]
            total_rem_steps = num_steps - steps
            total_rem_secs = max(total_rem_steps * seconds_per_step, 0)
            total_rem_secs = datetime.timedelta(seconds=total_rem_secs)
            total_rem_secs = str(total_rem_secs)[:-7]
            msg = f"Time left:  epoch {epoch_rem_secs}  total {total_rem_secs}"
            msg = msg.center(self.width)
            print(
                termcolor.colored("\r" + msg[:epoch_progress], color, on_color), end=""
            )
            print(msg[epoch_progress:], sep="", end="")
            self.last_epoch_progress = epoch_progress


def initialize(*args, **kwargs):
    """Initializes the global Logger instance."""
    global current_logger
    current_logger = Logger(*args, **kwargs)
    return current_logger


def get_current_logger():
    """Retrieves or creates the global Logger instance."""
    global current_logger
    if current_logger is None:
        current_logger = Logger()
    return current_logger


def store(*args, **kwargs):
    """Stores a metric in the global Logger."""
    logger = get_current_logger()
    return logger.store(*args, **kwargs)


def dump(*args, **kwargs):
    """Dumps buffered metrics from the global Logger."""
    logger = get_current_logger()
    return logger.dump(*args, **kwargs)


def show_progress(*args, **kwargs):
    """Renders progress on the global Logger."""
    logger = get_current_logger()
    return logger.show_progress(*args, **kwargs)


def close(*args, **kwargs):
    """Closes all backends on the global Logger."""
    logger = get_current_logger()
    return logger.close(*args, **kwargs)


def get_path():
    """Retrieves the active run directory path from the global Logger."""
    logger = get_current_logger()
    return logger.path