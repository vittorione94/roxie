#!/usr/bin/env python
"""The mjbatch_cpu cell: dm_control humanoid PPO on mjbatch, against release_v1.

mjbatch (https://github.com/kevinzakka/mjbatch) runs MuJoCo simulations in parallel on a
C++ thread pool, the same job EnvPool's pool does for the `envpool_cpu` cell. It is not a
roxie backend and cannot be one as it stands: it ships no dm_control tasks and no learner,
only the stepper. So this file carries the whole cell -- the suite humanoid built on
`Batch`, a PPO matching `experiments/dmc/agent/ppo_bench.yaml`, and the runner that pins
and packs the jobs -- and writes a `log.csv` whose columns match the release_v1 ones so the
curves overlay directly.

    scripts/run_release_benchmark_mjbatch_cpu.py --dry-run     # show the pinning plan
    scripts/run_release_benchmark_mjbatch_cpu.py               # the three humanoid tasks
    scripts/run_release_benchmark_mjbatch_cpu.py --task HumanoidWalk --out DIR   # one job

It needs mjbatch and torch, which live in mjbatch's own venv rather than roxie's, so the
script re-execs itself there; point MJBATCH_ROOT at the checkout if it is not in the
default place.

TWO RUNS AT ONCE, because that is the condition the baseline was measured in: every
envpool_cpu PPO run in release_v1 shared the box with another envpool_cpu run for most of
its life (ddpg for 6.8h of HumanoidStand's 7.3h, sac for 4.0h of HumanoidRun's 4.5h), so an
uncontended mjbatch run would beat it on scheduling alone. Slots come from the same
physical-core grouping `run_release_benchmark.sh` uses: CORES_PER_RUN is LOGICAL cores, so
10 is five physical cores with both SMT siblings, and no two slots share a physical core.
Letting them share is not a small effect -- an SMT-overlapping split measured 15k steps/s
against 22k for a disjoint one.

Only the physics backend is meant to differ, so do not retune the learner here. Note that
release_v1's own PPO baselines do NOT all agree with `ppo_bench.yaml`: HumanoidStand and
HumanoidWalk predate a retune and ran 8 epochs x 8 minibatches against the current 5 x 4,
which is most of why they log ~19k steps/s against HumanoidRun's ~31k. HumanoidRun is the
only task whose baseline is throughput-comparable to this cell as it stands.
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time


def _reexec_under_mjbatch():
    """Re-runs this script under mjbatch's venv, which is the one with mjbatch and torch."""
    try:
        import mjbatch  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    root = os.environ.get("MJBATCH_ROOT", os.path.expanduser("~/Documents/mjbatch"))
    python = os.path.join(root, ".venv", "bin", "python")
    if not os.path.exists(python):
        raise SystemExit(
            f"mjbatch is not importable and {python} does not exist. Set MJBATCH_ROOT to the "
            "checkout and run `uv sync --group examples` in it."
        )
    if os.path.realpath(python) == os.path.realpath(sys.executable):
        raise SystemExit(
            f"{python} cannot import mjbatch; run `uv sync --group examples` in {root}"
        )
    os.execv(python, [python, os.path.abspath(__file__), *sys.argv[1:]])


_reexec_under_mjbatch()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mjbatch import Batch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "outputs", "mjbatch_v1")
CORES_PER_RUN = 10  # LOGICAL cores, as in run_release_benchmark.sh
TASKS = ("HumanoidWalk", "HumanoidRun", "HumanoidStand")


def find_humanoid_xml():
    """The suite humanoid MJCF, preferring the copy EnvPool itself runs."""
    if "HUMANOID_XML" in os.environ:
        return os.environ["HUMANOID_XML"]
    patterns = (
        "~/.cache/uv/archive-v0/*/envpool_assets/mujoco/assets_dmc/humanoid.xml",
        "~/.cache/uv/archive-v0/*/mujoco_playground/_src/dm_control_suite/xmls/humanoid.xml",
    )
    for pattern in patterns:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            return hits[0]
    raise SystemExit("no suite humanoid.xml found; set HUMANOID_XML to one")


XML = find_humanoid_xml()
DEVICE = "cpu"  # the comparison is a CPU one; the learner stays off the GPU


# --- dm_control suite humanoid ---------------------------------------------------------
STAND_HEIGHT = 1.4
WALK_SPEED, RUN_SPEED = 1.0, 10.0
MOVE_SPEED = {"HumanoidStand": 0.0, "HumanoidWalk": WALK_SPEED, "HumanoidRun": RUN_SPEED}
SIM_DT, DECIMATION, EPISODE = 0.005, 5, 1000  # 0.025 control step, 1000 of them
OBS_DIM, ACT_DIM = 67, 21
EXTREMITIES = ("left_hand", "left_foot", "right_hand", "right_foot")

# --- roxie ppo_bench parity ------------------------------------------------------------
NUM_ENVS, HORIZON = 1024, 32  # sample_sequence_length; GAE spends the last on the bootstrap
TOTAL_STEPS, EPOCH_STEPS = 500_000_000, 500_000
GAMMA, LAMBDA, CLIP = 0.99, 0.95, 0.2
ENT_COEF, MAX_GRAD_NORM = 0.002, 1.0
EPOCHS, MINIBATCHES = 5, 4
LR, LR_END, LR_STEPS = 3e-4, 1e-4, 16000  # optax linear_schedule over optimizer steps
HIDDEN, INIT_STD, OUTPUT_INIT_SCALE = 256, 0.4, 0.01
STD_MIN, STD_MAX, OBS_EPS = 1e-2, 5.0, 1e-8
TEST_EPISODES = 10
SEED = 0
DEVICE = "cpu"  # the comparison is a CPU one; the learner stays off the GPU


def sigmoid(x, value_at_1, kind):
    if kind == "gaussian":
        return np.exp(-0.5 * (x * np.sqrt(-2 * np.log(value_at_1))) ** 2)
    if kind == "linear":
        scaled = x * (1 - value_at_1)
        return np.where(np.abs(scaled) < 1, 1 - scaled, 0.0)
    if kind == "quadratic":
        scaled = x * np.sqrt(1 - value_at_1)
        return np.where(np.abs(scaled) < 1, 1 - scaled**2, 0.0)
    raise ValueError(kind)


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, kind="gaussian", value_at_margin=0.1):
    """A numpy port of dm_control.utils.rewards.tolerance."""
    lower, upper = bounds
    in_bounds = (lower <= x) & (x <= upper)
    if margin == 0.0:
        return np.where(in_bounds, 1.0, 0.0)
    d = np.where(x < lower, lower - x, x - upper) / margin
    return np.where(in_bounds, 1.0, sigmoid(d, value_at_margin, kind))


def build_model():
    spec = mujoco.MjSpec.from_file(XML)
    # dm_control reads the COM velocity off this sensor. The suite XML declares it; the
    # playground copy of the same model does not, so add it when it is missing.
    if not any(s.name == "torso_subtreelinvel" for s in spec.sensors):
        spec.add_sensor(
            name="torso_subtreelinvel",
            type=mujoco.mjtSensor.mjSENS_SUBTREELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname="torso",
        )
    spec.option.timestep = SIM_DT
    return spec.compile()


class Humanoid:
    def __init__(self, task, num_envs, num_threads=0, seed=SEED):
        self.move_speed = MOVE_SPEED[task]
        model = build_model()
        # forward=True so xpos/xmat/sensordata are current with qpos after step, which is
        # what dm_control's mj_step2/mj_step1 split gives the suite tasks.
        self.batch = Batch(model, num_envs, num_threads, forward=True)
        self.qpos, self.qvel, self.ctrl = (self.batch.bind(f) for f in ("qpos", "qvel", "ctrl"))
        self.xpos, self.xmat = self.batch.bind("xpos"), self.batch.bind("xmat")
        self.com_vel = self.batch.sensor("torso_subtreelinvel")
        self.head = model.body("head").id
        self.torso = model.body("torso").id
        self.limbs = np.array([model.body(b).id for b in EXTREMITIES])
        jnt = model.jnt_type[1:], model.jnt_limited[1:], model.jnt_range[1:]
        self.jnt_type, self.jnt_limited, self.jnt_range = jnt
        self.rng = np.random.default_rng(seed)
        self.steps = np.zeros(num_envs, np.int64)
        self.num_envs = num_envs
        self.reset(np.arange(num_envs))

    def reset(self, ids):
        """Randomizes the pose the way dm_control's randomize_limited_and_rotational_joints does.

    Every hinge here is limited, so each takes a uniform draw from its own range, and the
    free root takes a normalized uniform quaternion -- `random.rand`, not `randn`, which
    dm_control keeps deliberately so its benchmark numbers stay comparable. Its retry loop
    for a pose that starts interpenetrating is NOT reproduced: mjbatch exposes no `ncon`.
    """
        n = ids.size
        self.batch.reset(ids)
        lo, hi = self.jnt_range[:, 0], self.jnt_range[:, 1]
        lo = np.where(self.jnt_limited, lo, -np.pi)
        hi = np.where(self.jnt_limited, hi, np.pi)
        quat = self.rng.random((n, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        self.qpos[ids[:, None], np.arange(3, 7)] = quat
        self.qpos[ids[:, None], np.arange(7, 7 + ACT_DIM)] = self.rng.uniform(lo, hi, (n, ACT_DIM))
        self.batch.forward(ids)
        self.steps[ids] = 0

    def head_height(self):
        return self.xpos[:, self.head, 2]

    def torso_frame(self):
        return self.xmat[:, self.torso].reshape(-1, 3, 3)

    def obs(self):
        frame = self.torso_frame()
        limb = self.xpos[:, self.limbs] - self.xpos[:, self.torso][:, None]
        cols = (
            self.qpos[:, 7:],  # joint angles
            self.head_height()[:, None],
            np.einsum("nlj,njk->nlk", limb, frame).reshape(-1, 12),  # extremities, torso frame
            frame[:, 2],  # torso vertical orientation
            self.com_vel,
            self.qvel,
        )
        return np.concatenate(cols, 1, dtype=np.float32)

    def reward(self, action):
        standing = tolerance(self.head_height(), (STAND_HEIGHT, np.inf), STAND_HEIGHT / 4)
        upright = tolerance(self.torso_frame()[:, 2, 2], (0.9, np.inf), 1.9, "linear", 0.0)
        small = tolerance(action, margin=1.0, kind="quadratic", value_at_margin=0.0).mean(-1)
        small = (4 + small) / 5
        if self.move_speed == 0.0:
            move = tolerance(self.com_vel[:, :2], margin=2.0).mean(-1)
        else:
            speed = np.linalg.norm(self.com_vel[:, :2], axis=-1)
            move = tolerance(speed, (self.move_speed, np.inf), self.move_speed, "linear", 0.0)
            move = (5 * move + 1) / 6
        return (standing * upright * small * move).astype(np.float32)

    def step(self, action):
        self.ctrl[:] = np.clip(action, -1.0, 1.0)
        self.batch.step(nstep=DECIMATION)
        self.steps += 1
        # The suite humanoid never terminates early, so every end is a timeout and bootstraps.
        return self.reward(self.ctrl), self.steps >= EPISODE


def log_density(z, log_std):
    return -0.5 * (z * z).sum(-1) - log_std.sum() - 0.5 * ACT_DIM * np.log(2 * np.pi)


def mlp(out_dim, scale=None):
    layers = [nn.Linear(OBS_DIM, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU()]
    head = nn.Linear(HIDDEN, out_dim)
    if scale is not None:  # roxie's output_init_scale, shrinking the initial policy
        with torch.no_grad():
            head.weight.mul_(scale)
            head.bias.mul_(scale)
    return nn.Sequential(*layers, head)


class ActorCritic(nn.Module):
    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self):
        super().__init__()
        self.actor = mlp(ACT_DIM, OUTPUT_INIT_SCALE)
        self.critic = mlp(1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), float(np.log(INIT_STD))))
        self.register_buffer("mean", torch.zeros(OBS_DIM))
        self.register_buffer("var", torch.ones(OBS_DIM))
        self.register_buffer("count", torch.full((), 1e-4))

    def std(self):
        return self.log_std.clamp(np.log(STD_MIN), np.log(STD_MAX)).exp()

    @torch.no_grad()
    def absorb(self, obs):  # Chan's parallel update of the running statistics
        n, delta = obs.shape[0], obs.mean(0) - self.mean
        total = self.count + n
        var = self.var * self.count + obs.var(0, correction=0) * n
        self.var.copy_((var + delta**2 * self.count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.count.add_(n)

    def forward(self, obs):
        obs = (obs - self.mean) / (self.var.sqrt() + OBS_EPS)
        return self.actor(obs), self.critic(obs).squeeze(-1)


@torch.no_grad()
def rollout(net, env, buf):
    def policy(obs):
        mean, val = net(torch.as_tensor(obs))
        return mean.numpy(), val.numpy()

    obs = env.obs()
    episodes = 0
    for t in range(HORIZON):
        mean, val = policy(obs)
        log_std = net.std().log().numpy()
        noise = env.rng.standard_normal(mean.shape, np.float32)
        act, logp = mean + np.exp(log_std) * noise, log_density(noise, log_std)
        reward, done = env.step(act)
        next_obs = env.obs()
        if done.any():  # a timeout, never a failure: bootstrap off the final observation
            reward = reward.copy()
            reward[done] += GAMMA * policy(next_obs[done])[1]
        for k, v in dict(obs=obs, act=act, logp=logp, val=val, rew=reward, alive=~done).items():
            buf[k][t] = v
        ids = np.flatnonzero(done)
        if ids.size:
            episodes += ids.size
            env.reset(ids)
            next_obs = env.obs()
        obs = next_obs
    batch = {k: torch.as_tensor(v) for k, v in buf.items()}
    batch["last_val"] = torch.as_tensor(policy(obs)[1])
    return batch, episodes


def gae(batch):
    vals = torch.cat([batch["val"], batch["last_val"][None]])
    adv, carry = torch.zeros_like(batch["rew"]), 0.0
    for t in reversed(range(HORIZON)):
        alive = batch["alive"][t]
        delta = batch["rew"][t] + GAMMA * alive * vals[t + 1] - vals[t]
        adv[t] = carry = delta + GAMMA * LAMBDA * alive * carry
    return adv, adv + batch["val"]


def update(net, actor_opt, critic_opt, batch, adv, ret):
    # GAE spends the last step on the bootstrap, so the usable rollout is HORIZON - 1.
    keep = slice(0, HORIZON - 1)
    obs = batch["obs"][keep].reshape(-1, OBS_DIM)
    act = batch["act"][keep].reshape(-1, ACT_DIM)
    logp_old = batch["logp"][keep].reshape(-1)
    adv, ret = adv[keep].reshape(-1), ret[keep].reshape(-1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    actor_params = list(net.actor.parameters()) + [net.log_std]
    critic_params = list(net.critic.parameters())
    stats = {"kl": 0.0, "clip": 0.0, "grads": 0}
    for _ in range(EPOCHS):
        for i in torch.randperm(obs.shape[0]).chunk(MINIBATCHES):
            mean, val = net(obs[i])
            std = net.std()
            logp = log_density((act[i] - mean) / std, std.log())
            ratio = (logp - logp_old[i]).exp()
            surrogate = torch.min(ratio * adv[i], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[i])
            entropy = (std.log() + 0.5 * np.log(2 * np.pi * np.e)).sum()
            actor_loss = -surrogate.mean() - ENT_COEF * entropy
            critic_loss = 0.5 * (val - ret[i]).pow(2).mean()
            actor_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)
            (actor_loss + critic_loss).backward()
            nn.utils.clip_grad_norm_(actor_params, MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(critic_params, MAX_GRAD_NORM)
            actor_opt.step()
            critic_opt.step()
            with torch.no_grad():
                stats["kl"] += float((logp_old[i] - logp).mean())
                stats["clip"] += float(((ratio - 1).abs() > CLIP).float().mean())
                stats["grads"] += 1
    net.absorb(obs)  # after the epochs: the batch was collected under the old statistics
    return stats


@torch.no_grad()
def evaluate(net, env):
    """Mean return over TEST_EPISODES full episodes from a fixed seed, acting at the mean."""
    env.rng = np.random.default_rng(12345)
    env.reset(np.arange(env.num_envs))
    total = np.zeros(env.num_envs, np.float64)
    for _ in range(EPISODE):
        mean, _ = net(torch.as_tensor(env.obs()))
        reward, _ = env.step(mean.numpy())
        total += reward
    return float(total.mean()), float(total.std())


def lr_at(step):
    """optax.linear_schedule(3e-4, 1e-4, 16000), which holds at end_value afterwards."""
    return LR + (LR_END - LR) * min(step / LR_STEPS, 1.0)


def train(task, out_dir, num_threads, total_steps, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    os.makedirs(out_dir, exist_ok=True)
    env = Humanoid(task, NUM_ENVS, num_threads, seed)
    test_env = Humanoid(task, TEST_EPISODES, num_threads, seed)
    net = ActorCritic().to(DEVICE)
    actor_opt = torch.optim.Adam(list(net.actor.parameters()) + [net.log_std], LR)
    critic_opt = torch.optim.Adam(net.critic.parameters(), LR)

    shapes = dict(obs=(OBS_DIM,), act=(ACT_DIM,), logp=(), val=(), rew=(), alive=())
    buf = {k: np.empty((HORIZON, NUM_ENVS, *v), np.float32) for k, v in shapes.items()}
    per_rollout = NUM_ENVS * HORIZON
    # An epoch ends at the first rollout boundary past a multiple of EPOCH_STEPS, which is
    # what roxie's trainer does -- so the two runs log at the same env-step marks.
    epochs = -(-total_steps // EPOCH_STEPS)

    columns = [
        "epoch",
        "steps",
        "sys/sps",
        "sys/time/epoch_s",
        "sys/time/total_s",
        "test/score",
        "test/score/std",
        "train/score",
        "train/episodes/total",
        "train/gradient_steps",
        "train/ppo/approx_kl",
        "train/ppo/clip_frac",
        "train/ppo/policy_std",
        "train/lr",
    ]
    csv_path = os.path.join(out_dir, "log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(columns)

    steps, grads, total_episodes, start = 0, 0, 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_score, kl, clip, rollouts = 0.0, 0.0, 0.0, 0
        while steps < min(epoch * EPOCH_STEPS, total_steps):
            for opt in (actor_opt, critic_opt):
                opt.param_groups[0]["lr"] = lr_at(grads)
            batch, episodes = rollout(net, env, buf)
            stats = update(net, actor_opt, critic_opt, batch, *gae(batch))
            steps += per_rollout
            grads += stats["grads"]
            total_episodes += episodes
            rollouts += 1
            train_score += float(batch["rew"].mean()) * EPISODE
            kl += stats["kl"] / stats["grads"]
            clip += stats["clip"] / stats["grads"]
        train_score, kl, clip = (x / rollouts for x in (train_score, kl, clip))
        score, score_std = evaluate(net, test_env)
        now = time.perf_counter()
        row = [
            epoch,
            steps,
            steps / (now - start),
            now - epoch_start,
            now - start,
            score,
            score_std,
            train_score,
            total_episodes,
            grads,
            kl,
            clip,
            float(net.std().mean().detach()),
            lr_at(grads),
        ]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        torch.save(net.state_dict(), os.path.join(out_dir, "policy.pt"))
        print(
            f"{task} epoch {epoch:4d}/{epochs}  {steps / 1e6:7.1f}M steps  "
            f"{steps / (now - start) / 1e3:6.1f}k sps  test {score:7.2f}  "
            f"train {train_score:7.2f}  elapsed {(now - start) / 3600:5.2f}h",
            flush=True,
        )


def physical_core_groups():
    """Each physical core's logical cpus, e.g. [[0, 12], [1, 13], ...] on an SMT box."""
    groups, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        if cpu in seen:
            continue
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        try:
            with open(path) as f:
                group = [int(c) for c in f.read().strip().split(",")]
        except OSError:
            group = [cpu]
        groups.append(group)
        seen.update(group)
    return groups


def plan_slots(cores_per_run=CORES_PER_RUN):
    """Splits the box into slots of whole physical cores, sharing none between them."""
    groups = physical_core_groups()
    per_slot = max(1, cores_per_run // len(groups[0]))
    slots = [
        sorted(c for group in groups[i : i + per_slot] for c in group)
        for i in range(0, len(groups) - per_slot + 1, per_slot)
    ]
    return slots


def run_cell(tasks, out_root, steps, cores_per_run, dry_run):
    """Runs each task as a pinned child of this script, keeping every slot busy."""
    slots = plan_slots(cores_per_run)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"mjbatch_cpu: {len(slots)} slots x {len(slots[0])} cpus, {steps} steps -> {out_root}")
    if dry_run:
        for i, cores in enumerate(slots):
            print(f"  slot {i} -> cpus {','.join(map(str, cores))}")
        print(f"  queue: {' '.join(tasks)}")
        return 0

    pending, running, failed = list(tasks), {}, []
    while pending or running:
        for i, cores in enumerate(slots):
            if i in running or not pending:
                continue
            task = pending.pop(0)
            out = os.path.join(out_root, task, "mjbatch_cpu", "ppo", stamp)
            os.makedirs(out, exist_ok=True)
            log = open(os.path.join(out, "train.log"), "w")
            argv = [sys.executable, os.path.abspath(__file__)]
            argv += ["--task", task, "--out", out, "--threads", str(cores_per_run)]
            argv += ["--steps", str(steps)]
            # The child pins itself, so the affinity holds however it was spawned.
            env = {**os.environ, "PIN_CPUS": ",".join(map(str, cores))}
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[i] = (task, child, log)
            print(f"[{time.strftime('%T')}] {task} -> cpus {','.join(map(str, cores))} -> {out}")
        for i, (task, child, log) in list(running.items()):
            if child.poll() is None:
                continue
            log.close()
            print(f"[{time.strftime('%T')}] {task} finished rc={child.returncode}")
            if child.returncode != 0:
                failed.append(task)
            del running[i]
        if running:
            time.sleep(5)
    print(f"[{time.strftime('%T')}] mjbatch_cpu done" + (f", failed: {failed}" if failed else ""))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(MOVE_SPEED), help="run this one task here")
    parser.add_argument("--out", help="output directory, required with --task")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--threads", type=int, default=CORES_PER_RUN)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="print the pinning plan and stop")
    args = parser.parse_args()
    # This runs for most of a day under nohup; block-buffered progress is no progress.
    sys.stdout.reconfigure(line_buffering=True)

    if args.task:
        if not args.out:
            parser.error("--out is required with --task")
        if "PIN_CPUS" in os.environ:
            os.sched_setaffinity(0, [int(c) for c in os.environ["PIN_CPUS"].split(",")])
        train(args.task, args.out, args.threads, args.steps, args.seed)
        return 0
    return run_cell(TASKS, args.out_root, args.steps, args.threads, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
