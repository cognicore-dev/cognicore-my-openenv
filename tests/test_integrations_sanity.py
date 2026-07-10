import pytest
import warnings

try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


@pytest.mark.skipif(not GYM_AVAILABLE, reason="Gymnasium not installed")
@pytest.mark.parametrize("env_id", [
    'cognicore/MazeRunner-v0',
    'cognicore/GridWorld-v0',
    'cognicore/Trading-v0',
    'cognicore/Survival-v0'
])
def test_gymnasium_envs(env_id):
    """Sanity check that Gymnasium environments comply with the API."""
    import cognicore.gym  # noqa: F401
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = gym.make(env_id)
        check_env(env.unwrapped, skip_render_check=True)
        env.close()


@pytest.mark.skipif(not GYM_AVAILABLE or not SB3_AVAILABLE, reason="Gymnasium or SB3 not installed")
def test_sb3_compatibility():
    """Sanity check that Stable Baselines 3 can wrap and train on our environments."""
    import cognicore.gym  # noqa: F401
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = Monitor(gym.make('cognicore/GridWorld-v0'))
        model = PPO('MlpPolicy', env, verbose=0, n_steps=64, batch_size=32)
        model.learn(total_timesteps=256)
        env.close()
