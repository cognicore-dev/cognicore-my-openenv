"""
Threat Reporter — generates security reports and dashboard data.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ThreatReport:
    generated_at: float = 0.0
    total_inputs: int = 0
    threats_blocked: int = 0
    threats_allowed: int = 0
    false_positives: int = 0
    accuracy: float = 0.0
    category_breakdown: Dict[str, int] = field(default_factory=dict)
    recent_threats: List[dict] = field(default_factory=list)
    antibody_count: int = 0
    defender_stats: dict = field(default_factory=dict)
    risk_trend: List[dict] = field(default_factory=list)


class ThreatReporter:
    """Generates security reports from immune system components."""

    def __init__(self, memory=None, antibodies=None, defender=None):
        self.memory = memory
        self.antibodies = antibodies
        self.defender = defender
        self._history = []

    def record_decision(self, input_text: str, decision_action: int,
                       was_threat: bool, confidence: float,
                       category: str = "unknown"):
        """Record a decision for reporting."""
        self._history.append({
            "timestamp": time.time(),
            "action": decision_action,
            "was_threat": was_threat,
            "confidence": confidence,
            "category": category,
            "input_preview": input_text[:50] + "..." if len(input_text) > 50 else input_text,
        })
        # Keep last 1000
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

    def generate_report(self) -> ThreatReport:
        """Generate a full security report."""
        report = ThreatReport(generated_at=time.time())

        report.total_inputs = len(self._history)

        # Count outcomes
        for h in self._history:
            if h["action"] == 1:  # BLOCK
                report.threats_blocked += 1
                if not h["was_threat"]:
                    report.false_positives += 1
            elif h["action"] == 0 and h["was_threat"]:  # ALLOW but was threat
                report.threats_allowed += 1

            cat = h.get("category", "unknown")
            report.category_breakdown[cat] = \
                report.category_breakdown.get(cat, 0) + 1

        # Accuracy
        if report.total_inputs > 0:
            correct = sum(1 for h in self._history
                         if (h["action"] != 0) == h["was_threat"])
            report.accuracy = correct / report.total_inputs

        # Recent threats
        threats = [h for h in self._history if h["was_threat"]]
        report.recent_threats = threats[-10:]

        # Antibody count
        if self.antibodies:
            report.antibody_count = len(self.antibodies.antibodies)

        # Defender stats
        if self.defender:
            report.defender_stats = self.defender.get_stats()

        # Memory stats
        if self.memory:
            try:
                mem_stats = self.memory.get_stats()
                report.defender_stats["memory"] = mem_stats
            except Exception:
                pass

        # Risk trend (hourly buckets for last 24h)
        now = time.time()
        for hour in range(24):
            t_start = now - (hour + 1) * 3600
            t_end = now - hour * 3600
            bucket = [h for h in self._history
                     if t_start <= h["timestamp"] < t_end]
            threats_in_bucket = sum(1 for h in bucket if h["was_threat"])
            report.risk_trend.append({
                "hour": -hour,
                "total": len(bucket),
                "threats": threats_in_bucket,
            })

        return report

    def to_dashboard_json(self) -> dict:
        """Return JSON-serializable data for the dashboard /security page."""
        r = self.generate_report()
        return {
            "generated_at": r.generated_at,
            "total_inputs": r.total_inputs,
            "threats_blocked": r.threats_blocked,
            "false_positives": r.false_positives,
            "accuracy": round(r.accuracy, 4),
            "categories": r.category_breakdown,
            "recent_threats": r.recent_threats,
            "antibody_count": r.antibody_count,
            "defender": r.defender_stats,
            "risk_trend": r.risk_trend,
        }
