"""
Antibody Store — known threat patterns stored as embeddings.
Provides instant O(1) lookup for previously-seen attacks.
Antibodies decay over time unless reinforced.
"""
import numpy as np
import json, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

STORE_DIR = Path.home() / ".cognicore" / "immune"
STORE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Antibody:
    antibody_id: str
    pattern_embedding: np.ndarray
    threat_type: str
    response_action: int  # DefenseAction value
    confidence: float
    created_at: float
    last_seen: float
    reinforcement_count: int = 1
    decay_rate: float = 0.001  # per day

    @property
    def potency(self) -> float:
        """Antibody strength — decays over time unless reinforced."""
        days_old = (time.time() - self.last_seen) / 86400.0
        decay = max(0.0, 1.0 - self.decay_rate * days_old)
        boost = min(2.0, 1.0 + 0.1 * self.reinforcement_count)
        return min(1.0, self.confidence * decay * boost)


@dataclass
class AntibodyMatch:
    matched: bool
    confidence: float = 0.0
    threat_type: str = ""
    response_action: int = 0
    antibody_id: str = ""
    similarity: float = 0.0


class AntibodyStore:
    """Fast lookup store for known threat patterns."""

    def __init__(self, store_path=None):
        self.store_path = store_path or str(STORE_DIR / "antibodies.json")
        self.antibodies: List[Antibody] = []
        self._load()

    def create_antibody(self, features: np.ndarray, threat_type: str,
                       response_action: int, confidence: float):
        """Create a new antibody from a confirmed threat."""
        ab_id = f"ab_{int(time.time())}_{len(self.antibodies)}"
        ab = Antibody(
            antibody_id=ab_id,
            pattern_embedding=features.copy(),
            threat_type=threat_type,
            response_action=response_action,
            confidence=confidence,
            created_at=time.time(),
            last_seen=time.time())
        self.antibodies.append(ab)
        self._save()
        return ab_id

    def match(self, features: np.ndarray,
             threshold: float = 0.90) -> AntibodyMatch:
        """Check if input matches a known threat antibody."""
        if not self.antibodies or features is None:
            return AntibodyMatch(matched=False)

        features = np.array(features, dtype=np.float32)
        best_sim = 0.0
        best_ab = None

        for ab in self.antibodies:
            if ab.potency < 0.1:
                continue  # Decayed antibody
            emb = ab.pattern_embedding
            if len(emb) != len(features):
                continue
            norm_a = np.linalg.norm(features)
            norm_b = np.linalg.norm(emb)
            if norm_a < 1e-8 or norm_b < 1e-8:
                continue
            sim = float(np.dot(features, emb) / (norm_a * norm_b))
            effective_sim = sim * ab.potency
            if effective_sim > best_sim:
                best_sim = effective_sim
                best_ab = ab

        if best_ab and best_sim >= threshold:
            return AntibodyMatch(
                matched=True,
                confidence=best_sim,
                threat_type=best_ab.threat_type,
                response_action=best_ab.response_action,
                antibody_id=best_ab.antibody_id,
                similarity=best_sim)

        return AntibodyMatch(matched=False, similarity=best_sim)

    def reinforce(self, antibody_id: str):
        """Strengthen an antibody when the same threat is seen again."""
        for ab in self.antibodies:
            if ab.antibody_id == antibody_id:
                ab.reinforcement_count += 1
                ab.last_seen = time.time()
                self._save()
                return True
        return False

    def prune_decayed(self, min_potency: float = 0.05):
        """Remove antibodies that have decayed below threshold."""
        before = len(self.antibodies)
        self.antibodies = [ab for ab in self.antibodies
                          if ab.potency >= min_potency]
        pruned = before - len(self.antibodies)
        if pruned > 0:
            self._save()
        return pruned

    def get_all(self) -> List[dict]:
        return [{
            "id": ab.antibody_id,
            "type": ab.threat_type,
            "potency": round(ab.potency, 3),
            "reinforced": ab.reinforcement_count,
            "age_days": round((time.time() - ab.created_at) / 86400, 1),
        } for ab in self.antibodies]

    def _save(self):
        data = []
        for ab in self.antibodies:
            data.append({
                "antibody_id": ab.antibody_id,
                "pattern_embedding": ab.pattern_embedding.tolist(),
                "threat_type": ab.threat_type,
                "response_action": ab.response_action,
                "confidence": ab.confidence,
                "created_at": ab.created_at,
                "last_seen": ab.last_seen,
                "reinforcement_count": ab.reinforcement_count,
                "decay_rate": ab.decay_rate,
            })
        Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.store_path).write_text(json.dumps(data, indent=2))

    def _load(self):
        p = Path(self.store_path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            self.antibodies = []
            for d in data:
                self.antibodies.append(Antibody(
                    antibody_id=d["antibody_id"],
                    pattern_embedding=np.array(d["pattern_embedding"], dtype=np.float32),
                    threat_type=d["threat_type"],
                    response_action=d["response_action"],
                    confidence=d["confidence"],
                    created_at=d["created_at"],
                    last_seen=d["last_seen"],
                    reinforcement_count=d.get("reinforcement_count", 1),
                    decay_rate=d.get("decay_rate", 0.001)))
        except Exception:
            self.antibodies = []

    def clear(self):
        self.antibodies = []
        p = Path(self.store_path)
        if p.exists():
            p.unlink()
