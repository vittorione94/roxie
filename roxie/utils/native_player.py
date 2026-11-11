"""Native MuJoCo CPU execution engine for interactive playback."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

# What a step reads and no earlier step can recompute.
_STATE_FIELDS = (
    "time", "qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied",
    "xfrc_applied", "mocap_pos", "mocap_quat",
)
# Same size in both, different meaning: C's `M` is sparse in its own order.
_LAID_OUT_DIFFERENTLY = frozenset({"M"})
# Pytree metadata in an MJX `Contact`, which no step can change.
_STATIC_CONTACT_FIELDS = frozenset({"dim", "efc_address"})
# What MJX gives a contact slot nothing occupies.
_NO_CONTACT_DIST = 1e10


class _CPhysics:
    """One model's C MuJoCo, called from inside a jitted task as a pure function."""

    def __init__(self, model: mujoco.MjModel, mjx_model: Any):
        self._model = model
        # Kept alive: `_C_PHYSICS` is keyed by its id, reused once freed.
        self._mjx_model = mjx_model
        self._data = mujoco.MjData(model)
        self._warned_contacts = False

    def _shared_fields(self, struct: Any) -> dict[str, Any]:
        """Float fields of an MJX struct that this `MjData` holds in the same layout."""
        fields = {}
        for field in dataclasses.fields(struct):
            value = getattr(struct, field.name)
            held = getattr(self._data, field.name, None)
            if (
                held is not None
                and field.name not in _LAID_OUT_DIFFERENTLY
                and not field.name.startswith("efc_")
                and hasattr(value, "shape")
                and jnp.issubdtype(value.dtype, jnp.inexact)
                and np.size(held) == np.prod(value.shape, dtype=int)
            ):
                fields[field.name] = value
        return fields

    def _read_contacts(self, specs, efc_spec, dims, addresses):
        """This step's contacts and their forces, in MJX's fixed contact layout.

        The way `mjx.put_data` places them: `dim` and `efc_address` are static in
        an MJX `Contact`, so each contact takes a free slot of its own dimension
        and its force rows go to that slot's address.

        Args:
            specs: shape and dtype of each `Contact` array that can change.
            efc_spec: shape and dtype of `efc_force`.
            dims: the static `Contact.dim`.
            addresses: the static `Contact.efc_address`.

        Returns:
            `(contacts, efc_force)`. Only contact rows of `efc_force` are filled;
            equality and limit rows sit elsewhere in C, by this step's count.
        """
        data = self._data
        contacts = {
            name: np.zeros(spec.shape, dtype=spec.dtype) for name, spec in specs.items()
        }
        efc_force = np.zeros(efc_spec.shape, dtype=efc_spec.dtype)
        contacts["dist"][:] = _NO_CONTACT_DIST
        for name in ("geom", "geom1", "geom2"):
            contacts[name][:] = -1

        pyramidal = self._model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        free = {dim: list(np.flatnonzero(dims == dim)) for dim in np.unique(dims)}
        found = data.contact
        dropped = 0
        # Deepest first: box on box overflows MJX's slots; drop the shallowest.
        for index in np.argsort(found.dist[: data.ncon]):
            dim = found.dim[index]
            if not free.get(dim):
                dropped += 1
                continue
            slot = free[dim].pop(0)
            for name in contacts:
                held = np.asarray(getattr(found, name)[index])
                contacts[name][slot] = held.reshape(specs[name].shape[1:])
            row = found.efc_address[index]
            if row >= 0:
                rows = 2 * (dim - 1) if pyramidal and dim > 1 else dim
                efc_force[addresses[slot] : addresses[slot] + rows] = (
                    data.efc_force[row : row + rows]
                )
        if dropped and not self._warned_contacts:
            self._warned_contacts = True
            print(
                f"C MuJoCo found {data.ncon} contacts and {dropped} did not fit "
                f"MJX's layout; the deepest are kept in data.contact.",
                flush=True,
            )
        return contacts, efc_force

    def _advance(self, state, ctrl, substeps, specs, dims, addresses):
        data = self._data
        for name, value in state.items():
            if name == "time":
                data.time = float(value)
            else:
                getattr(data, name)[:] = np.asarray(value).reshape(getattr(data, name).shape)
        if ctrl is not None:
            data.ctrl[:] = np.asarray(ctrl)
        if substeps:
            mujoco.mj_step(self._model, data, nstep=substeps)
        else:
            mujoco.mj_forward(self._model, data)

        def read(group):
            return {
                name: np.asarray(getattr(data, name), dtype=spec.dtype).reshape(spec.shape)
                for name, spec in group.items()
            }

        contacts, efc_force = self._read_contacts(
            specs["contact"], specs["efc_force"], dims, addresses,
        )
        return {
            "data": read(specs["data"]),
            "impl": read(specs["impl"]),
            "contact": contacts,
            "efc_force": efc_force,
        }

    def call(self, data: Any, ctrl: Any, substeps: int) -> Any:
        """Returns `data` after `substeps` steps, or after a forward pass if zero."""
        impl = data._impl

        def spec_of(value):
            return jax.ShapeDtypeStruct(value.shape, value.dtype)

        results = self._shared_fields(data)
        specs = {
            "data": jax.tree.map(spec_of, results),
            "impl": jax.tree.map(spec_of, self._shared_fields(impl)),
            "contact": {
                field.name: spec_of(getattr(impl.contact, field.name))
                for field in dataclasses.fields(impl.contact)
                if field.name not in _STATIC_CONTACT_FIELDS
            },
            "efc_force": spec_of(impl.efc_force),
        }
        state = {name: getattr(data, name) for name in _STATE_FIELDS if name in results}
        # Pure, not `io_callback`: the whole state goes in and out each call.
        updated = jax.pure_callback(
            functools.partial(
                self._advance, substeps=substeps, specs=specs,
                dims=np.asarray(impl.contact.dim),
                addresses=np.asarray(impl.contact.efc_address),
            ),
            specs, state, ctrl,
        )
        return data.replace(
            _impl=impl.replace(
                contact=impl.contact.replace(**updated["contact"]),
                efc_force=updated["efc_force"],
                **updated["impl"],
            ),
            **updated["data"],
        )


_C_PHYSICS: dict[int, _CPhysics] = {}


def _route_mjx_to_c() -> None:
    """Sends MJX's physics entry points to C MuJoCo for registered models.

    At these calls and not above them, because a task can use physics anywhere:
    finger's reset reads forward kinematics, and cheetah's settles for 200 steps.
    Any other model passes straight through to MJX.
    """
    from mujoco import mjx
    from mujoco_playground._src import mjx_env

    if getattr(mjx.forward, "routed_to_c", False):
        return
    mjx_forward, mjx_step, env_step = mjx.forward, mjx.step, mjx_env.step

    def forward(model, data):
        physics = _C_PHYSICS.get(id(model))
        return physics.call(data, None, 0) if physics else mjx_forward(model, data)

    def step(model, data):
        physics = _C_PHYSICS.get(id(model))
        return physics.call(data, None, 1) if physics else mjx_step(model, data)

    def substeps(model, data, action, n_substeps=1):
        physics = _C_PHYSICS.get(id(model))
        if physics is None:
            return env_step(model, data, action, n_substeps)
        return physics.call(data, action, n_substeps)

    forward.routed_to_c = True
    mjx.forward, mjx.step, mjx_env.step = forward, step, substeps


class CPhysicsPlayer:
    """Steps a `mujoco_playground` task with its physics in C MuJoCo.

    The task's reset, observation and reward stay its own JAX code, jitted, and
    only its physics calls leave for C, which leaves 0.1 s to compile where MJX's
    physics took 8 s on the humanoid. Nothing in the task has to know.

    Attributes:
        reset: starts an episode from a key.
        step: advances the state by one action.
    """

    def __init__(self, func_env: Any):
        env = func_env.env
        _route_mjx_to_c()
        _C_PHYSICS[id(env.mjx_model)] = _CPhysics(env.mj_model, env.mjx_model)
        self.reset = jax.jit(env.reset)
        self.step = jax.jit(env.step)


def make_native_player(func_env: Any) -> CPhysicsPlayer | None:
    """Returns the player of a `mujoco_playground` task, or None for anything else."""
    if hasattr(getattr(func_env, "env", None), "mjx_model"):
        return CPhysicsPlayer(func_env)
    return None
