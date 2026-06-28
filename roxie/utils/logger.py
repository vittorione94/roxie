import datetime
import os
import subprocess
import time

import numpy as np
import termcolor
import yaml

current_logger = None

# nvidia-smi GPU telemetry. Probed lazily and disabled after the first failure
# (no GPU / no nvidia-smi / driver error) so a CPU run or a missing binary
# doesn't pay a subprocess cost — or spam errors — every epoch.
_gpu_stats_enabled = True
_GPU_QUERY_FIELDS = (
    "utilization.gpu",
    "temperature.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
)


def gpu_stats(prefix="gpu"):
    """Sample per-GPU telemetry via ``nvidia-smi`` as a flat metric dict.

    Returns ``{}`` when no NVIDIA GPU is queryable (and disables itself so later
    calls are free). Cheap enough to call once per epoch. Keys are nested under
    ``prefix`` (and the GPU index when more than one device is present) so the
    logger backends group them: e.g. ``gpu/util_pct``, ``gpu/temp_c``, or
    ``gpu/0/util_pct`` on multi-GPU hosts.
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
            # nvidia-smi reports e.g. "[N/A]" for power on some cards.
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
    print(termcolor.colored(msg, color, attrs=["bold"]))


def warning(msg, color="yellow"):
    print(termcolor.colored("Warning: " + msg, color, attrs=["bold"]))


def error(msg, color="red"):
    print(termcolor.colored("Error: " + msg, color, attrs=["bold"]))


class Backend:
    """A sink for aggregated metrics.

    At the end of every epoch the :class:`Logger` calls ``log`` with the flat
    metric dict (keys grouped with ``/``) and the step it belongs to. Concrete
    backends display, persist, or upload them. Keep ``log`` cheap and never let
    it raise — the Logger isolates failures, but a slow backend stalls training.
    """

    def log(self, data, step):
        raise NotImplementedError

    def close(self):
        pass


class ConsoleBackend(Backend):
    """Pretty-prints metrics to stdout as an indented, ``/``-nested table."""

    def __init__(self, width=60):
        self.width = width
        self.known_keys = set()
        self.final_keys = []
        self.console_formats = []

    def _rebuild_layout(self, new_keys):
        first_row = len(self.known_keys) == 0
        if not first_row:
            print()
            warning(f"Logging new keys {new_keys}")
        for key in new_keys:
            self.known_keys.add(key)
        self.final_keys = list(sorted(self.known_keys))
        self.console_formats = []
        seen = set()
        for key in self.final_keys:
            *left_keys, right_key = key.split("/")
            for i, k in enumerate(left_keys):
                left_key = "/".join(left_keys[: i + 1])
                if left_key not in seen:
                    left = "  " * i + k.replace("_", " ")
                    self.console_formats.append((left, None))
                    seen.add(left_key)
            indent = "  " * len(left_keys)
            right_key = right_key.replace("_", " ")
            self.console_formats.append((indent + right_key, key))

    def log(self, data, step):
        new_keys = [key for key in data if key not in self.known_keys]
        if new_keys:
            self._rebuild_layout(new_keys)

        print()
        for left, key in self.console_formats:
            if key:
                val = data.get(key)
                str_type = str(type(val))
                if "tensorflow" in str_type:
                    warning(f"Logging TensorFlow tensor {key}")
                elif "torch" in str_type:
                    warning(f"Logging Torch tensor {key}")
                if np.issubdtype(type(val), np.floating):
                    right = f"{val:8.3g}"
                elif np.issubdtype(type(val), np.integer):
                    right = f"{val:,}"
                else:
                    right = str(val)
                spaces = " " * (self.width - len(left) - len(right))
                print(left + spaces + right)
            else:
                spaces = " " * (self.width - len(left))
                print(left + spaces)
        print()


class CSVBackend(Backend):
    """Appends metrics to a CSV file, rewriting columns when new keys appear."""

    def __init__(self, path):
        self.log_file_path = path
        self.known_keys = set()
        self.final_keys = []

    def log(self, data, step):
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
                # New columns appeared mid-run: rewrite the file, back-filling
                # the previous rows with "None" for the freshly added keys.
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
    """Streams metrics to Weights & Biases.

    ``wandb`` is imported lazily so the dependency is only required when this
    backend is actually enabled. Construct it with the same ``config`` dict you
    save to disk so the run page mirrors the experiment configuration.
    """

    def __init__(self, project=None, entity=None, name=None, group=None,
                 tags=None, mode="online", config=None, dir=None,
                 relogin=True, **kwargs):
        try:
            import wandb
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "wandb logging is enabled but the 'wandb' package is not "
                "installed. Install it with `uv add wandb` (or disable "
                "logging.wandb.enabled)."
            ) from e
        self._wandb = wandb
        # Force an interactive re-authentication before the run starts. On a
        # desktop with several wandb accounts the cached netrc credentials would
        # otherwise silently pick whichever was last used, sending the run to the
        # wrong account; ``relogin`` re-prompts for the API key every time so the
        # account is chosen deliberately. Skipped when WANDB_API_KEY is set (CI)
        # or offline/disabled modes, where no auth is needed. Set relogin=False
        # to trust the cached login instead.
        needs_auth = relogin and mode not in ("offline", "disabled") \
            and not os.environ.get("WANDB_API_KEY")
        if needs_auth:
            wandb.login(relogin=True)
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=list(tags) if tags else None,
            mode=mode,
            config=config,
            dir=dir,
            **kwargs,
        )

    def log(self, data, step):
        self.run.log(data, step=int(step))

    def close(self):
        self.run.finish()


def default_backends(path, width=60):
    """Console + CSV — the always-on backends written to the run directory."""
    return [ConsoleBackend(width=width), CSVBackend(os.path.join(path, "log.csv"))]


class Logger:
    """Collects per-epoch metrics and fans them out to one or more backends.

    ``store`` accumulates named values across an epoch; ``dump`` aggregates them
    (mean, or full stats for keys stored with ``stats=True``) into a flat dict
    and hands it to every backend. Display and persistence live entirely in the
    backends — by default a :class:`ConsoleBackend` and a :class:`CSVBackend`.
    """

    def __init__(self, path=None, width=60, script_path=None, config=None,
                 backends=None):
        self.path = path or str(time.time())

        # Save the launch script.
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

        # Save the configuration.
        if config:
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:
                pass
            config_path = os.path.join(self.path, "config.yaml")
            with open(config_path, "w") as config_file:
                yaml.dump(config, config_file)
            log(f"Config file saved to {config_path}")

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
        """Attach another backend (e.g. wandb) after construction."""
        self.backends.append(backend)

    def store(self, key, value, stats=False):
        """Keeps named values during an epoch."""

        if key not in self.epoch_dict:
            self.epoch_dict[key] = [value]
            if stats:
                self.stat_keys.add(key)
        else:
            self.epoch_dict[key].append(value)

    def dump(self, step=None):
        """Aggregates the epoch's values and forwards them to the backends.

        ``step`` is the x-axis the metrics belong to (e.g. environment steps);
        when omitted it falls back to a monotonic epoch counter.
        """

        # Compute statistics if needed.
        keys = list(self.epoch_dict.keys())
        for key in keys:
            values = self.epoch_dict[key]
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
        for backend in self.backends:
            try:
                backend.close()
            except Exception as exc:
                error(f"Backend {type(backend).__name__} failed to close: {exc}")

    def show_progress(
        self, steps, num_epoch_steps, num_steps, color="white", on_color="on_blue"
    ):
        """Shows a progress bar for the current epoch and total training."""

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
    global current_logger
    current_logger = Logger(*args, **kwargs)
    return current_logger


def get_current_logger():
    global current_logger
    if current_logger is None:
        current_logger = Logger()
    return current_logger


def store(*args, **kwargs):
    logger = get_current_logger()
    return logger.store(*args, **kwargs)


def dump(*args, **kwargs):
    logger = get_current_logger()
    return logger.dump(*args, **kwargs)


def show_progress(*args, **kwargs):
    logger = get_current_logger()
    return logger.show_progress(*args, **kwargs)


def close(*args, **kwargs):
    logger = get_current_logger()
    return logger.close(*args, **kwargs)


def get_path():
    logger = get_current_logger()
    return logger.path
