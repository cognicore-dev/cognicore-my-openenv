"""
CogniCore x HuggingFace Hub — Upload and download trained models.

Upload:
    from cognicore.integrations.huggingface import upload_model
    upload_model(model, "cognicore/ppo-mazerunner-v0", env_id="cognicore/MazeRunner-v0")

Download:
    from cognicore.integrations.huggingface import download_model
    model = download_model("cognicore/ppo-mazerunner-v0")
"""
from __future__ import annotations
import os
import json
import logging
from typing import Optional

logger = logging.getLogger("cognicore.hf")

try:
    from huggingface_hub import HfApi, hf_hub_download, create_repo
    HAS_HF = True
except ImportError:
    HAS_HF = False

try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.evaluation import evaluate_policy
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def upload_model(
    model,
    repo_id: str,
    env_id: str,
    algo: str = "PPO",
    train_steps: int = 0,
    eval_reward: float = 0.0,
    eval_std: float = 0.0,
    tags: Optional[list] = None,
    token: Optional[str] = None,
) -> str:
    """Upload a trained SB3 model to HuggingFace Hub.

    Parameters
    ----------
    model : SB3 model
        Trained model to upload.
    repo_id : str
        HF repo ID (e.g., "username/ppo-mazerunner").
    env_id : str
        CogniCore environment ID used for training.
    algo : str
        Algorithm name.
    train_steps : int
        Number of training steps.
    eval_reward : float
        Mean evaluation reward.
    token : str or None
        HF token. Uses HF_TOKEN env var if not provided.

    Returns
    -------
    str
        URL of uploaded model.
    """
    if not HAS_HF:
        raise ImportError("pip install huggingface-hub")
    if not HAS_SB3:
        raise ImportError("pip install stable-baselines3")

    token = token or os.environ.get("HF_TOKEN")

    # Save model locally
    local_dir = f"_hf_upload_{repo_id.replace('/', '_')}"
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, "model")
    model.save(model_path)

    # Create model card
    card = f"""---
library_name: stable-baselines3
tags:
- reinforcement-learning
- cognicore
- {algo.lower()}
- {env_id.replace('/', '-')}
{('- ' + chr(10) + '- ').join(tags) if tags else ''}
model-index:
- name: {algo} on {env_id}
  results:
  - task:
      type: reinforcement-learning
      name: {env_id}
    metrics:
    - type: mean_reward
      value: {eval_reward:.1f} +/- {eval_std:.1f}
---

# {algo} on {env_id}

Trained with [CogniCore](https://github.com/Kaushalt2004/cognicore-my-openenv) environments.

## Training

- **Algorithm:** {algo}
- **Environment:** `{env_id}`
- **Training Steps:** {train_steps:,}
- **Mean Reward:** {eval_reward:+.1f} +/- {eval_std:.1f}

## Usage

```python
import cognicore.gym
import gymnasium as gym
from stable_baselines3 import {algo}

# Load model
model = {algo}.load("model")

# Run
env = gym.make("{env_id}")
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Environment

CogniCore provides cognitive RL environments with:
- Memory middleware (embedding-based retrieval)
- Procedural generation
- Auto-curriculum
- SB3/Gymnasium compatible
"""

    card_path = os.path.join(local_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card)

    # Metadata
    meta = {
        "env_id": env_id,
        "algo": algo,
        "train_steps": train_steps,
        "eval_reward": eval_reward,
        "eval_std": eval_std,
        "framework": "cognicore",
    }
    with open(os.path.join(local_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Upload
    api = HfApi(token=token)
    try:
        create_repo(repo_id, token=token, exist_ok=True)
    except Exception:
        pass

    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        token=token,
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Model uploaded to {url}")

    # Cleanup
    import shutil
    shutil.rmtree(local_dir, ignore_errors=True)

    return url


def download_model(
    repo_id: str,
    algo: str = "PPO",
    revision: str = "main",
    token: Optional[str] = None,
):
    """Download a trained model from HuggingFace Hub.

    Returns
    -------
    SB3 model
        Loaded model ready for inference.
    """
    if not HAS_HF:
        raise ImportError("pip install huggingface-hub")
    if not HAS_SB3:
        raise ImportError("pip install stable-baselines3")

    algos = {"PPO": PPO, "DQN": DQN, "A2C": A2C}
    algo_cls = algos.get(algo.upper(), PPO)

    # Download model file from an explicit revision for reproducibility.
    model_path = hf_hub_download(repo_id, "model.zip", revision=revision, token=token)
    model = algo_cls.load(model_path)

    logger.info(f"Model loaded from {repo_id}")
    return model


def list_cognicore_models(token: Optional[str] = None) -> list:
    """List all CogniCore models on HuggingFace Hub."""
    if not HAS_HF:
        raise ImportError("pip install huggingface-hub")

    api = HfApi(token=token)
    models = api.list_models(tags="cognicore", sort="downloads", direction=-1)
    return [{"id": m.id, "downloads": m.downloads, "likes": m.likes} for m in models]
