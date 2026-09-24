"""The metric namespace is a contract, not an implementation detail.

Panel definitions in `roxie/report.py`, the curve loader in `roxie/plot.py`, the
throughput read-back in `scripts/run_release_benchmark.sh` and every saved wandb
report are all written against these key names. Renaming one silently turns a
published panel into an empty axis, so the scheme is pinned here.

Every metric belongs to exactly one section:

    epoch / steps   the run axes, ungrouped
    train/*         the behaviour policy and the learner
    test/*          the held-out eval
    sys/*           throughput, wall-clock, host and device health
"""

import pytest

from roxie.utils.logger import ConsoleBackend, CSVBackend, Logger


TOP_LEVEL = {"epoch", "steps"}
SECTIONS = ("train/", "test/", "sys/")


class _CaptureBackend:
    """Records what the Logger hands to a backend."""

    def __init__(self):
        self.calls = []

    def log(self, data, step):
        self.calls.append((dict(data), step))

    def close(self):
        pass


class TestNamespaceScheme:
    def test_every_key_the_repo_writes_is_sectioned(self):
        """The names `roxie/report.py`, `roxie/plot.py` and every saved wandb
        report are written against. One test over the list: a failure names the
        key, and 26 cells that all assert one prefix rule are 26 rows of noise.
        """
        for key in (
            "train/score", "train/score/std", "train/length", "train/length/std",
            "train/episodes/epoch", "train/episodes/total", "train/gradient_steps",
            "train/loss/actor", "train/loss/critic", "train/reward/upright",
            "train/noise/per_joint_abs", "train/td3/tanh_grad", "train/ppo/approx_kl",
            "train/mining/effective_bins",
            "test/score", "test/score/std", "test/length", "test/length/std",
            "test/distinct_starts", "test/score_per_step",
            "sys/sps", "sys/time/total_s", "sys/time/epoch_s",
            "sys/gpu/util_pct", "sys/mem/rss_gb", "sys/mem/live_arrays",
        ):
            assert key.startswith(SECTIONS), (
                f"{key} is not under train/, test/ or sys/"
            )

    def test_trainer_stores_only_sectioned_keys(self):
        """Read the trainer's store() calls out of the source.

        Driving the real `_store_epoch_metrics` would need an env, an agent and
        a card. The thing worth protecting is the literal key names, and those
        are visible statically.
        """
        import ast
        import inspect
        import textwrap

        from roxie.utils.trainer import Trainer

        # The only two places the trainer calls store(), so every key name is
        # decided here.
        literals = []
        for fn in (Trainer._store_epoch_metrics, Trainer._store_test_metrics):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "store"):
                    continue
                if not node.args:
                    continue
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    literals.append(arg.value)
                elif isinstance(arg, ast.JoinedStr):
                    # f"train/{k}" — check the literal prefix only.
                    head = arg.values[0]
                    if isinstance(head, ast.Constant):
                        literals.append(head.value)

        assert literals, "found no store() literals — did the trainer move?"
        for key in literals:
            assert key in TOP_LEVEL or key.startswith(SECTIONS), (
                f"Trainer logs unsectioned key {key!r}; put it under "
                f"train/, test/ or sys/ (or add it to the run axes)"
            )

    def test_backends_do_not_log_for_themselves(self):
        """The namespace is the trainer's alone.

        The rollout runs the eval and carries the env's params, so it could
        quietly start logging its own keys — and a backend-specific key would be
        exactly the drift that splitting the loop was meant to prevent. It
        returns values instead; `_store_*` names them.
        """
        import inspect

        from roxie.utils import rollout

        src = inspect.getsource(rollout)
        assert "logger.store" not in src, (
            "roxie.utils.rollout logs its own metrics; hand the values back "
            "to Trainer._store_* instead, so one place owns the namespace"
        )

    def test_every_metric_name_is_namespaced(self):
        """Every metric goes out under `train/`, `test/` or `sys/`: a bare name
        collides across emitters and sorts into the wrong group in a plot."""
        import inspect

        from roxie.utils.trainer import Trainer

        src = inspect.getsource(Trainer._store_epoch_metrics)
        for bare in ('"score"', '"length"', '"sps"', '"loss/actor"',
                     '"loss/critic"', '"gradient_steps"', '"episodes/epoch"',
                     '"time/total_s"', '"mem/rss_gb"'):
            assert bare not in src, f"metric {bare} is emitted un-namespaced"


class TestAbsentMetrics:
    """A metric that did not happen must read as absent, never as zero.

    `train/loss/*` logged as 0.0 during epochs with no learning pass is what
    made the v1 release grid's zero-gradient-step bug invisible — 50 epochs of
    a flat, plausible-looking zero loss. See tests/test_update_schedule.py.
    """

    def test_none_survives_to_the_backend(self):
        cap = _CaptureBackend()
        log = Logger(path="/tmp", backends=[cap])
        log.store("train/loss/actor", None)
        log.store("train/score", 1.0)
        log.dump(step=10)

        data, step = cap.calls[0]
        assert step == 10
        assert data["train/loss/actor"] is None
        assert data["train/score"] == 1.0

    def test_console_renders_none_as_a_dash(self):
        assert ConsoleBackend._fmt(None) == "-"
        assert ConsoleBackend._fmt(None, pad=True).strip() == "-"

    def test_console_dump_with_none_does_not_raise(self, capsys):
        backend = ConsoleBackend()
        backend.log({"epoch": 1, "train/loss/actor": None, "train/score": 2.5}, step=1)
        assert "train" in capsys.readouterr().out

    def test_wandb_backend_drops_none(self):
        """wandb gets a gap in the series, not a null."""
        from roxie.utils.logger import WandbBackend

        sent = {}

        class _FakeRun:
            def log(self, data, step):
                sent.update({"data": data, "step": step})

        backend = WandbBackend.__new__(WandbBackend)  # bypass wandb.init
        backend.run = _FakeRun()
        backend.log({"train/loss/actor": None, "train/score": 3.0}, step=7)

        assert "train/loss/actor" not in sent["data"]
        assert sent["data"] == {"train/score": 3.0}
        assert sent["step"] == 7

    def test_csv_writes_none_for_absent(self, tmp_path):
        path = tmp_path / "log.csv"
        backend = CSVBackend(str(path))
        backend.log({"train/score": 1.0, "train/loss/actor": None}, step=1)
        lines = path.read_text().splitlines()
        assert lines[0] == "train/loss/actor,train/score"
        assert lines[1] == "None,1.0"


class TestConsoleLayout:
    def test_sections_group_and_order(self, capsys):
        backend = ConsoleBackend()
        backend.log({
            "epoch": 3, "steps": 1024,
            "sys/sps": 5000.0, "train/score": 12.0, "test/score": 15.0,
            "train/loss/actor": -0.5,
        }, step=1024)
        out = capsys.readouterr().out
        # Section headers are present and train precedes test precedes sys.
        assert out.index("train") < out.index("test") < out.index("sys")

    def test_mean_std_pair_collapses(self, capsys):
        backend = ConsoleBackend()
        backend.log({"train/score": 10.0, "train/score/std": 2.0}, step=1)
        out = capsys.readouterr().out
        assert "+-" in out
        # `std` is folded into the score row, not printed as its own leaf.
        assert out.count("std") == 0


class TestAggregation:
    def test_dump_means_repeated_stores(self):
        cap = _CaptureBackend()
        log = Logger(path="/tmp", backends=[cap])
        for v in (1.0, 2.0, 3.0):
            log.store("train/score", v)
        log.dump(step=1)
        assert cap.calls[0][0]["train/score"] == pytest.approx(2.0)

    def test_dump_clears_between_epochs(self):
        cap = _CaptureBackend()
        log = Logger(path="/tmp", backends=[cap])
        log.store("train/score", 1.0)
        log.dump(step=1)
        log.store("train/score", 5.0)
        log.dump(step=2)
        assert cap.calls[1][0]["train/score"] == pytest.approx(5.0)

    def test_backend_failure_is_isolated(self):
        class _Boom:
            def log(self, data, step):
                raise RuntimeError("backend down")

            def close(self):
                pass

        cap = _CaptureBackend()
        log = Logger(path="/tmp", backends=[_Boom(), cap])
        log.store("train/score", 1.0)
        log.dump(step=1)  # must not raise
        assert cap.calls, "a failing backend blocked a healthy one"
