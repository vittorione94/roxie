from mujoco_playground import registry

def load_playground_env(env_name: str):
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    return env, env_cfg