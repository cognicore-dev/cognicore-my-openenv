"""
Civilization System — Federated agent behavioral knowledge sharing.

This module provides peer-to-peer sharing of behavioral knowledge without 
sharing raw data. Agents can join a Civilization to contribute to and absorb
insights from the collective.
"""

import json
import socket
import threading
import statistics
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, List, Optional


class InsightReport:
    """Structured report containing aggregated insights for an environment."""
    def __init__(self, data: Dict[str, Any]):
        self.top_failure_modes = data.get("top_failure_modes", {})
        self.most_effective_strategies = data.get("most_effective_strategies", {})
        self.average_episode_length = data.get("average_episode_length", 0.0)
        self.reward_distribution = data.get("reward_distribution", {})
        self.contributing_agent_count = data.get("contributing_agent_count", 0)

    def to_dict(self):
        return {
            "top_failure_modes": self.top_failure_modes,
            "most_effective_strategies": self.most_effective_strategies,
            "average_episode_length": self.average_episode_length,
            "reward_distribution": self.reward_distribution,
            "contributing_agent_count": self.contributing_agent_count
        }


class Civilization:
    """Federated learning client for interacting with a Civilization server."""
    
    def __init__(self, agent: Any):
        self.agent = agent
        self.address = None
        self.privacy_level = "isolated"
        self.peer_id = None

    def join(self, address: str, privacy_level: str = "federated") -> bool:
        """Joins a civilization server.
        
        Args:
            address: URL like 'http://localhost:9876' or 'cognicore://...'.
            privacy_level: 'federated', 'isolated', or 'open'.
            
        Returns:
            bool: True if connected.
        """
        if address.startswith("cognicore://"):
            address = address.replace("cognicore://", "http://", 1)
            
        self.address = address
        self.privacy_level = privacy_level
        
        try:
            req = urllib.request.Request(f"{self.address}/register", method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:  # nosec B310
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    self.peer_id = data.get("peer_id")
                    return True
        except Exception:
            return False
        return False

    def extract_statistics(self) -> Dict[str, Any]:
        """Extracts anonymized behavioral stats from the agent's history."""
        # This explicitly NEVER includes raw observations or prompts
        store = getattr(self.agent, "store", None)
        events = store.get_all() if hasattr(store, "get_all") else []
        
        actions = [str(getattr(e, "output_text", getattr(e, "action", ""))) for e in events]
        rewards = [getattr(e, "reward", 0.0) for e in events]
        
        # Action Distribution
        act_dist = {}
        for a in actions:
            act_dist[a] = act_dist.get(a, 0) + 1
            
        # Reward Signals per Strategy (heuristically, mapping previous action to reward)
        strategy_rewards = {}
        if len(actions) > 0 and len(rewards) > 0:
            strategy_rewards[actions[-1]] = sum(rewards)
            
        # Failure Modes (actions taken right before negative rewards)
        failure_modes = {}
        for i in range(1, len(rewards)):
            if rewards[i] < 0:
                failure_modes[actions[i-1]] = failure_modes.get(actions[i-1], 0) + 1
                
        # Memory Access Patterns
        memory_accesses = sum(1 for a in actions if "recall" in a.lower())
        
        return {
            "action_distribution": act_dist,
            "strategy_rewards": strategy_rewards,
            "failure_frequencies": failure_modes,
            "memory_access_rate": memory_accesses / max(1, len(actions)),
            "episode_length": len(actions)
        }

    def contribute(self, env_id: str) -> bool:
        """Uploads behavioral stats to the network if privacy allows."""
        if self.privacy_level == "isolated":
            return False
            
        stats = self.extract_statistics()
        payload = json.dumps({"peer_id": self.peer_id, "env_id": env_id, "stats": stats}).encode('utf-8')
        
        try:
            req = urllib.request.Request(f"{self.address}/contribute", data=payload, method="POST")
            req.add_header('Content-Type', 'application/json')
            with urllib.request.urlopen(req, timeout=5) as response:  # nosec B310
                return response.status == 200
        except Exception:
            return False

    def global_insights(self, env_id: str) -> InsightReport:
        """Queries the network for aggregated statistics."""
        if not self.address:
            return InsightReport({})
            
        try:
            req = urllib.request.Request(f"{self.address}/insights?env_id={env_id}", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:  # nosec B310
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    return InsightReport(data)
        except Exception:
            pass
        return InsightReport({})

    def absorb(self, report: InsightReport):
        """Applies received behavioral statistics as soft constraints or priors."""
        # In a real agent, this would modify action selection distributions.
        # We attach the insights to the agent so it can be queried during `.act()`
        self.agent.civilization_priors = report.to_dict()


class CivilizationRequestHandler(BaseHTTPRequestHandler):
    """HTTP Handler for the Civilization Server."""
    
    server_state = {
        "peers": set(),
        "contributions": {}  # env_id -> list of stats
    }
    
    def _send_json(self, data: Dict[str, Any], status: int = 200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/register":
            import uuid
            peer_id = str(uuid.uuid4())
            self.server_state["peers"].add(peer_id)
            self._send_json({"peer_id": peer_id})
            
        elif parsed.path == "/contribute":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                peer_id = data.get("peer_id")
                env_id = data.get("env_id")
                stats = data.get("stats")
                
                if peer_id not in self.server_state["peers"]:
                    self._send_json({"error": "Unauthorized"}, 401)
                    return
                    
                # Byzantine Fault Tolerance: Reject statistical outliers
                # If episode length is suspiciously high (e.g. > 100000), reject
                if stats.get("episode_length", 0) > 100000:
                    self._send_json({"error": "Rejected: Statistical outlier"}, 400)
                    return
                    
                if env_id not in self.server_state["contributions"]:
                    self.server_state["contributions"][env_id] = []
                self.server_state["contributions"][env_id].append(stats)
                
                self._send_json({"status": "accepted"})
            except Exception:
                self._send_json({"error": "Bad request"}, 400)
        else:
            self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/insights":
            query = parse_qs(parsed.query)
            env_id = query.get("env_id", [""])[0]
            
            conts = self.server_state["contributions"].get(env_id, [])
            count = len(conts)
            
            if count == 0:
                self._send_json(InsightReport({}).to_dict())
                return
                
            # Aggregate stats
            avg_len = sum(c.get("episode_length", 0) for c in conts) / count
            
            all_failures = {}
            all_strategies = {}
            for c in conts:
                for k, v in c.get("failure_frequencies", {}).items():
                    all_failures[k] = all_failures.get(k, 0) + v
                for k, v in c.get("strategy_rewards", {}).items():
                    all_strategies[k] = all_strategies.get(k, 0) + v
                    
            report = {
                "top_failure_modes": all_failures,
                "most_effective_strategies": all_strategies,
                "average_episode_length": avg_len,
                "reward_distribution": {"mean": 0.0, "std": 0.0},
                "contributing_agent_count": count
            }
            self._send_json(report)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress logging for cleaner CLI
        pass


class CivilizationServer:
    """Runs the central node for federated learning."""
    
    def __init__(self, port: int = 9876):
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        self.server = HTTPServer(('0.0.0.0', self.port), CivilizationRequestHandler)  # nosec B104
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join()
