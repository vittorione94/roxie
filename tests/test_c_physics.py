"""`mujoco_playground` tasks replayed with their physics in C MuJoCo."""

import dataclasses

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx
from mujoco.mjx._src import support

from roxie.environment.loader import build_playground_env, load_playground_env
from roxie.environment.suites import DMC_TASKS
from roxie.utils import native_player
from roxie.utils.native_player import CPhysicsPlayer, make_native_player

KEY = jax.random.PRNGKey(0)


def _func_env(task):
    func_env, _ = load_playground_env(task, impl="jax")
    return func_env


@pytest.fixture
def c_calls(monkeypatch):
    calls = []
    advance = native_player._CPhysics._advance

    def counted(self, *args, **kwargs):
        calls.append(1)
        return advance(self, *args, **kwargs)

    monkeypatch.setattr(native_player._CPhysics, "_advance", counted)
    return calls


@pytest.mark.parametrize("task", DMC_TASKS)
def test_every_benchmark_task_runs_with_its_physics_in_c(task, c_calls):
    func_env = _func_env(task)
    player = CPhysicsPlayer(func_env)
    state = player.reset(KEY)
    action = jnp.zeros(func_env.env.action_size, dtype=jnp.float32)
    state = player.step(state, action)
    jax.block_until_ready(state.obs)
    assert c_calls
    assert np.isfinite(np.asarray(state.obs)).all()
    assert float(state.data.time) > 0.0


def test_c_tracks_mjx_on_the_first_steps():
    func_env = _func_env("CartpoleBalance")
    env = func_env.env
    # Distinct functions: `jax.jit` caches on the bound method itself, so jitting
    # `env.step` here would hand back whichever program was traced first.
    mjx_reset = jax.jit(lambda key: env.reset(key))
    mjx_step = jax.jit(lambda state, action: env.step(state, action))
    native_player._C_PHYSICS.pop(id(env.mjx_model), None)
    mjx_state = mjx_step(mjx_reset(KEY), jnp.zeros(env.action_size, dtype=jnp.float32))

    player = CPhysicsPlayer(func_env)
    c_state = player.step(player.reset(KEY), jnp.zeros(env.action_size, dtype=jnp.float32))
    np.testing.assert_allclose(mjx_state.obs, c_state.obs, atol=1e-3)


def test_a_model_nothing_registered_stays_on_mjx(c_calls):
    registered = _func_env("CartpoleBalance")
    CPhysicsPlayer(registered)
    other = _func_env("PendulumSwingup").env
    state = jax.jit(lambda key: other.reset(key))(KEY)
    jax.block_until_ready(state.obs)
    assert not c_calls


def test_a_registered_model_outlives_its_player():
    """Its id keys the routing, so a freed one could route a stranger to C."""
    func_env = _func_env("CartpoleBalance")
    model = func_env.env.mjx_model
    CPhysicsPlayer(func_env)
    assert native_player._C_PHYSICS[id(model)]._mjx_model is model


def test_a_push_reaches_the_c_physics():
    func_env = _func_env("CartpoleBalance")
    player = CPhysicsPlayer(func_env)
    still = player.reset(KEY)
    action = jnp.zeros(func_env.env.action_size, dtype=jnp.float32)
    force = np.zeros((func_env.env.mj_model.nbody, 6), dtype=np.float32)
    force[1, 0] = 50.0
    pushed = still.replace(data=still.data.replace(xfrc_applied=jnp.asarray(force)))
    assert not np.allclose(
        player.step(still, action).data.qvel, player.step(pushed, action).data.qvel,
    )


def test_a_world_built_to_play_reports_native_physics():
    bundle = build_playground_env("CartpoleBalance", mode="play")
    assert bundle.env.func_env.metadata["impl"] == "native"
    assert isinstance(make_native_player(bundle.env.func_env), CPhysicsPlayer)


def test_an_env_with_no_mujoco_task_has_no_native_player():
    assert make_native_player(object()) is None


CONTACTS = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="ball" pos="0 0 .3"><freejoint/><geom name="ball" size=".1"/></body>
    <body name="box" pos="2 0 .099"><freejoint/><geom name="box" type="box" size=".1 .1 .1"/></body>
  </worldbody>
  <sensor>
    <touch name="under_ball" site="ball_bottom"/>
    <subtreelinvel name="ball_velocity" body="ball"/>
  </sensor>
</mujoco>""".replace(
    '<geom name="ball" size=".1"/>',
    '<geom name="ball" size=".1"/><site name="ball_bottom" size=".12"/>',
)


def _contact_world():
    model = mujoco.MjModel.from_xml_string(CONTACTS)
    mjx_model = mjx.put_model(model)
    physics = native_player._CPhysics(model, mjx_model)
    native_player._route_mjx_to_c()
    return model, mjx_model, physics, mjx.make_data(model)


def _c_step(physics, data, substeps=1):
    return jax.jit(lambda d: physics.call(d, None, substeps))(data)


def _active(data):
    """`{(geom, geom): deepest dist}` for the contacts MJX would call touching."""
    contact = data._impl.contact
    found = {}
    for geom, dist in zip(np.asarray(contact.geom), np.asarray(contact.dist)):
        if dist < 0:
            pair = tuple(sorted(int(g) for g in geom))
            found[pair] = min(found.get(pair, 0.0), float(dist))
    return found


def test_data_contact_reports_what_c_found_as_mjx_would():
    model, mjx_model, physics, data = _contact_world()
    c_data = _c_step(physics, data, 0)
    # This `mjx_model` is registered with nothing, so the route hands it to MJX.
    mjx_data = jax.jit(lambda d: mjx.forward(mjx_model, d))(data)
    assert set(_active(c_data)) == set(_active(mjx_data)) == {(0, 2)}
    assert _active(c_data)[(0, 2)] == pytest.approx(_active(mjx_data)[(0, 2)], abs=1e-5)


def test_contacts_appear_when_a_body_lands_and_empty_slots_match_nothing():
    model, _, physics, data = _contact_world()
    before = _c_step(physics, data, 0)
    assert (0, 1) not in _active(before)
    spare = np.asarray(before._impl.contact.dist) == 1e10
    assert spare.any()
    assert (np.asarray(before._impl.contact.geom)[spare] == -1).all()

    after = _c_step(physics, data, 200)
    assert (0, 1) in _active(after)
    assert float(after.sensordata[0]) > 0.0


def test_mjx_contact_force_works_on_what_c_returns():
    model, mjx_model, physics, data = _contact_world()
    settled = _c_step(physics, data, 500)
    contact = settled._impl.contact
    slot = int(np.argmin(np.where(
        (np.asarray(contact.geom) == [0, 1]).all(axis=1), np.asarray(contact.dist), 1e10,
    )))
    force = support.contact_force(mjx_model, settled, slot)
    weight = float(model.body_mass[1]) * 9.81
    assert float(force[0]) == pytest.approx(weight, rel=0.05)


def test_too_many_contacts_keep_the_deepest_and_say_so(capsys):
    _, _, physics, data = _contact_world()
    physics._data.qpos[:] = np.asarray(data.qpos)
    physics._data.qpos[2] = 0.05
    mujoco.mj_forward(physics._model, physics._data)
    assert physics._data.ncon > 1
    contact = data._impl.contact
    specs = {
        field.name: jax.ShapeDtypeStruct(
            (1, *getattr(contact, field.name).shape[1:]),
            getattr(contact, field.name).dtype,
        )
        for field in dataclasses.fields(contact)
        if field.name not in native_player._STATIC_CONTACT_FIELDS
    }
    efc = jax.ShapeDtypeStruct((4,), np.float32)
    dims, addresses = np.array([3]), np.array([0])
    kept, efc_force = physics._read_contacts(specs, efc, dims, addresses)
    assert kept["dist"][0] == pytest.approx(physics._data.contact.dist.min())
    assert np.abs(efc_force).sum() > 0.0
    assert capsys.readouterr().out.count("the deepest are kept") == 1
    physics._read_contacts(specs, efc, dims, addresses)
    assert capsys.readouterr().out == ""


def test_the_static_contact_layout_is_left_as_mjx_built_it():
    _, _, physics, data = _contact_world()
    stepped = _c_step(physics, data, 5)
    np.testing.assert_array_equal(stepped._impl.contact.dim, data._impl.contact.dim)
    np.testing.assert_array_equal(
        stepped._impl.contact.efc_address, data._impl.contact.efc_address,
    )


def test_derived_fields_inside_the_mjx_impl_come_back_from_c():
    _, _, physics, data = _contact_world()
    stepped = _c_step(physics, data, 10)
    assert np.abs(np.asarray(stepped._impl.cinert)).sum() > 0.0
    copied = physics._shared_fields(data._impl)
    assert {"cinert", "crb", "cacc", "subtree_linvel"} <= set(copied)
    assert not any(name.startswith("efc_") for name in copied)
    assert "M" not in copied
