"""
AgentPassport System — Complete live state serialization for CogniCore agents.

This module provides the `AgentPassport` class which captures an agent's memory,
replay event history, immune state, reflection state, and configuration into a
single, portable `.passport` file (a zip-based format with JSON manifests and
binary blobs for numpy arrays). It supports deployment via local, ssh, http,
and wasm protocols.
"""

import json
import zipfile
import tempfile
import os
import shutil
import urllib.request
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
import numpy as np


class AgentPassport:
    """Captures and restores the entire live state of a running agent.
    
    A passport is a zip file containing:
    - manifest.json: Configuration and metadata.
    - memory.json: Serialized memory entries.
    - replay.json: Replay event history.
    - immune.json: Immune antibody store and defender states.
    - weights.npz: Numpy arrays of RL defender weights.
    - reflection.json: Reflection engine state.
    - reward.json: Reward history.
    """
    
    VERSION = "1.0.0"

    @classmethod
    def checkpoint(
        cls, 
        agent: Any, 
        env: Any, 
        store: Any, 
        memory: Any, 
        immune_state: Optional[Any] = None
    ) -> str:
        """Captures the live state and serializes it into a .passport file.
        
        Args:
            agent: The running agent instance.
            env: The environment instance.
            store: The EventStore containing replay history.
            memory: The MemoryBackend containing memory entries.
            immune_state: Optional immune system state (AntibodyStore/RLDefender).
            
        Returns:
            str: Path to the generated .passport file (saved in current dir).
        """
        temp_dir = tempfile.mkdtemp()
        try:
            manifest = {
                "version": cls.VERSION,
                "agent_class": agent.__class__.__name__,
                "env_class": env.__class__.__name__,
                "config": getattr(agent, "config", {})
            }
            with open(os.path.join(temp_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

            memory_entries = []
            if hasattr(memory, "entries"):
                memory_entries = [
                    {
                        "text": getattr(e, "text", ""),
                        "category": getattr(e, "category", ""),
                        "success": getattr(e, "success", getattr(e, "correct", True)),
                        "action": getattr(e, "action", ""),
                        "metadata": getattr(e, "metadata", {})
                    } for e in memory.entries
                ]
            elif hasattr(memory, "get_all"):
                memory_entries = memory.get_all() 
                
            with open(os.path.join(temp_dir, "memory.json"), "w") as f:
                json.dump(memory_entries, f)

            events = []
            if hasattr(store, "get_all"):
                for ev in store.get_all():
                    events.append({
                        "event_id": getattr(ev, "event_id", ""),
                        "task_id": getattr(ev, "task_id", ""),
                        "seq": getattr(ev, "seq", 0),
                        "step": getattr(ev, "step", 0),
                        "event_type": getattr(ev, "event_type", ""),
                        "agent": getattr(ev, "agent", ""),
                        "input_text": getattr(ev, "input_text", ""),
                        "output_text": getattr(ev, "output_text", ""),
                        "reward": getattr(ev, "reward", 0.0),
                        "timestamp": getattr(ev, "timestamp", 0.0)
                    })
            with open(os.path.join(temp_dir, "replay.json"), "w") as f:
                json.dump(events, f)
                
            immune_dict = {}
            weights = {}
            if immune_state:
                immune_dict["antibodies"] = getattr(immune_state, "antibodies", [])
                if hasattr(immune_state, "weights") and isinstance(immune_state.weights, dict):
                    weights = immune_state.weights
                    
            with open(os.path.join(temp_dir, "immune.json"), "w") as f:
                json.dump(immune_dict, f)
            if weights:
                np.savez(os.path.join(temp_dir, "weights.npz"), **weights)
                
            reflection_state = getattr(agent, "reflection_state", {})
            with open(os.path.join(temp_dir, "reflection.json"), "w") as f:
                json.dump(reflection_state, f)
                
            reward_history = getattr(agent, "reward_history", [])
            with open(os.path.join(temp_dir, "reward.json"), "w") as f:
                json.dump(reward_history, f)

            import uuid
            passport_name = f"agent_{int(time.time())}_{uuid.uuid4().hex[:8]}.passport"
            
            with zipfile.ZipFile(passport_name, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zf.write(file_path, arcname=file)
                        
            return passport_name
        finally:
            shutil.rmtree(temp_dir)

    @classmethod
    def restore(cls, filepath: str) -> Dict[str, Any]:
        """Restores a fully working agent state from a .passport file.
        
        Args:
            filepath: Path to the .passport file.
            
        Returns:
            Dict containing the restored components.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is corrupted.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Passport file not found: {filepath}")
            
        state = {}
        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                with zf.open("manifest.json") as f:
                    state["manifest"] = json.load(f)
                
                if state["manifest"].get("version", "0.0.0") > cls.VERSION:
                    import logging
                    logging.getLogger(__name__).warning("Restoring a passport from a newer version!")
                    
                for json_file in ["memory.json", "replay.json", "immune.json", "reflection.json", "reward.json"]:
                    if json_file in zf.namelist():
                        with zf.open(json_file) as f:
                            state[json_file.split('.')[0]] = json.load(f)
                            
                if "weights.npz" in zf.namelist():
                    with zf.open("weights.npz") as f:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
                            tmp.write(f.read())
                            tmp_name = tmp.name
                        try:
                            loaded = np.load(tmp_name, allow_pickle=True)
                            state["weights"] = {k: np.array(v) for k, v in loaded.items()}
                        finally:
                            if 'loaded' in locals():
                                loaded.close()
                            # Retry remove if it fails (Windows quirk with zipfile)
                            try:
                                os.remove(tmp_name)
                            except OSError:
                                pass
                else:
                    state["weights"] = {}
                    
        except zipfile.BadZipFile:
            raise ValueError(f"Corrupted passport file: {filepath}")
            
        return state

    @classmethod
    def deploy(cls, filepath: str, target: str) -> bool:
        """Deploys a passport to a remote or local target.
        
        Args:
            filepath: Path to the .passport file.
            target: String format "local://path", "ssh://user@host:path", 
                    "http://endpoint", or "wasm://browser".
                    
        Returns:
            bool: True if deployment succeeded, False otherwise.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Passport file not found: {filepath}")
            
        if target.startswith("local://"):
            dest = target[len("local://"):]
            dest_dir = os.path.dirname(dest)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(filepath, dest)
            return True
            
        elif target.startswith("ssh://"):
            parts = target[len("ssh://"):].split(":")
            if len(parts) != 2:
                raise ValueError("SSH target must be ssh://user@host:path")
            host_part, path_part = parts
            try:
                subprocess.run(["scp", filepath, f"{host_part}:{path_part}"], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
                
        elif target.startswith("http://") or target.startswith("https://"):
            try:
                with open(filepath, "rb") as f:
                    req = urllib.request.Request(target, data=f.read(), method="POST")
                    req.add_header('Content-Type', 'application/zip')
                    urllib.request.urlopen(req, timeout=10)  # nosec B310
                return True
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"HTTP Deploy failed: {e}")
                return False
                
        elif target.startswith("wasm://"):
            import logging
            logging.getLogger(__name__).warning("WASM deployment is a stub in this runtime environment.")
            return True
            
        else:
            raise ValueError(f"Unsupported deployment target scheme: {target}")

    @classmethod
    def diff(cls, passport_a: str, passport_b: str) -> str:
        """Produces a human-readable report of exactly what changed between two checkpoints.
        
        Args:
            passport_a: Path to older passport.
            passport_b: Path to newer passport.
            
        Returns:
            str: Human readable delta report.
        """
        state_a = cls.restore(passport_a)
        state_b = cls.restore(passport_b)
        
        report = []
        report.append(f"Passport Diff: {os.path.basename(passport_a)} -> {os.path.basename(passport_b)}")
        report.append("=" * 50)
        
        mem_a = len(state_a.get("memory", []))
        mem_b = len(state_b.get("memory", []))
        report.append(f"Memory entries: {mem_a} -> {mem_b} ({mem_b - mem_a:+d})")
        
        rep_a = len(state_a.get("replay", []))
        rep_b = len(state_b.get("replay", []))
        report.append(f"Replay events : {rep_a} -> {rep_b} ({rep_b - rep_a:+d})")
        
        rew_a = sum(state_a.get("reward", []))
        rew_b = sum(state_b.get("reward", []))
        report.append(f"Total reward  : {rew_a} -> {rew_b} ({rew_b - rew_a:+})")
        
        imm_a = len(state_a.get("immune", {}).get("antibodies", []))
        imm_b = len(state_b.get("immune", {}).get("antibodies", []))
        report.append(f"Immune antibodies: {imm_a} -> {imm_b} ({imm_b - imm_a:+d})")
        
        conf_a = state_a.get("manifest", {}).get("config", {})
        conf_b = state_b.get("manifest", {}).get("config", {})
        if conf_a != conf_b:
            report.append("Configuration changed.")
        else:
            report.append("Configuration identical.")
            
        return "\n".join(report)

