
"""
CogniCore Webhook Alerts — Notify when agent performance changes.

Sends notifications via configurable hooks when performance drops,
streaks occur, or custom conditions are met.

Usage::

    from cognicore.webhooks import AlertSystem

    alerts = AlertSystem()
    alerts.on("accuracy_drop", threshold=0.5)
    alerts.on("failure_streak", count=3)
    alerts.check(stats)
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger("cognicore.webhooks")


class AlertRule:
    """A single alert rule."""

    def __init__(
        self,
        name: str,
        condition: Callable,
        message: str = "",
        severity: str = "WARNING",
        cooldown: float = 60.0,
    ):
        self.name = name
        self.condition = condition
        self.message = message
        self.severity = severity
        self.cooldown = cooldown
        self._last_fired = 0.0
        self.fire_count = 0

    def evaluate(self, data: Dict) -> Optional[Dict]:
        """Check rule and return alert if triggered."""
        now = time.time()
        if now - self._last_fired < self.cooldown:
            return None

        try:
            if self.condition(data):
                self._last_fired = now
                self.fire_count += 1
                return {
                    "rule": self.name,
                    "severity": self.severity,
                    "message": self.message or f"Alert: {self.name} triggered",
                    "fired_at": now,
                    "fire_count": self.fire_count,
                    "data": {k: v for k, v in data.items() if not callable(v)},
                }
        except Exception:
            pass
        return None


class AlertSystem:
    """Agent performance alert system.

    Monitors agent performance and fires alerts when conditions are met.
    Supports custom handlers (print, file, HTTP webhook).
    """

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.handlers: List[Callable] = [self._default_handler]
        self.alert_log: List[Dict] = []

    def on(
        self,
        event: str,
        threshold: float = 0.0,
        count: int = 0,
        severity: str = "WARNING",
        message: str = "",
        cooldown: float = 0.0,
    ) -> "AlertSystem":
        """Register an alert rule.

        Built-in events:
        - 'accuracy_drop': fires when accuracy < threshold
        - 'failure_streak': fires when streak >= count
        - 'score_drop': fires when score < threshold
        - 'high_cost': fires when cost > threshold
        - 'slow_response': fires when latency > threshold (ms)

        Parameters
        ----------
        event : str
            Event name or 'custom' with a condition.
        threshold : float
            Threshold value for comparison.
        count : int
            Count threshold for streak events.
        severity : str
            'INFO', 'WARNING', 'CRITICAL'
        """
        conditions = {
            "accuracy_drop": lambda d: d.get("accuracy", 1.0) < threshold,
            "failure_streak": lambda d: abs(d.get("streak", 0)) >= count,
            "score_drop": lambda d: d.get("score", 1.0) < threshold,
            "high_cost": lambda d: d.get("cost", 0) > threshold,
            "slow_response": lambda d: d.get("latency_ms", 0) > threshold,
            "memory_full": lambda d: d.get("memory_entries", 0) > threshold,
            "low_confidence": lambda d: d.get("confidence", 1.0) < threshold,
        }

        if event in conditions:
            condition = conditions[event]
            default_msg = f"{event}: threshold={threshold or count}"
        else:

            def condition(_d):
                return False

            default_msg = f"Custom: {event}"

        rule = AlertRule(
            name=event,
            condition=condition,
            message=message or default_msg,
            severity=severity,
            cooldown=cooldown,
        )
        self.rules.append(rule)
        return self

    def on_custom(
        self,
        name: str,
        condition: Callable,
        severity: str = "WARNING",
        message: str = "",
    ) -> "AlertSystem":
        """Register a custom alert rule."""
        self.rules.append(AlertRule(name, condition, message, severity))
        return self

    def add_handler(self, handler: Callable) -> "AlertSystem":
        """Add a custom alert handler.

        handler(alert_dict) -> None
        """
        self.handlers.append(handler)
        return self

    def add_file_handler(self, path: str) -> "AlertSystem":
        """Add a file-based alert handler."""

        def file_handler(alert):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert, default=str) + "\n")

        self.handlers.append(file_handler)
        return self

    def add_webhook_handler(self, url: str) -> "AlertSystem":
        """Add an HTTP webhook handler (requires httpx/requests)."""

        def webhook_handler(alert):
            try:
                import httpx

                httpx.post(url, json=alert, timeout=5)
            except ImportError:
                try:
                    import urllib.parse
                    import urllib.request

                    parsed = urllib.parse.urlparse(url)
                    if parsed.scheme not in ("http", "https"):
                        raise ValueError("Webhook URL must use http or https protocol")
                    data = json.dumps(alert, default=str).encode()
                    req = urllib.request.Request(
                        url,
                        data=data,
                        headers={"Content-Type": "application/json"},
                    )
                    urllib.request.urlopen(req, timeout=5)  # nosec B310
                except Exception:
                    pass

        self.handlers.append(webhook_handler)
        return self

    def check(self, data: Dict) -> List[Dict]:
        """Check all rules against current data.

        Parameters
        ----------
        data : dict
            Agent metrics: accuracy, streak, score, cost, etc.

        Returns
        -------
        List of triggered alerts.
        """
        fired = []
        for rule in self.rules:
            alert = rule.evaluate(data)
            if alert:
                fired.append(alert)
                self.alert_log.append(alert)
                for handler in self.handlers:
                    try:
                        handler(alert)
                    except Exception:
                        pass
        return fired

    def _default_handler(self, alert: Dict):
        """Default handler: print alert."""
        severity = alert["severity"]
        icons = {"INFO": "i", "WARNING": "!", "CRITICAL": "X"}
        icon = icons.get(severity, "?")
        logger.info(f"  [{icon}] [{severity}] {alert['message']}")

    def get_log(self, last_n: int = 20) -> List[Dict]:
        return self.alert_log[-last_n:]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_rules": len(self.rules),
            "total_alerts": len(self.alert_log),
            "rules": [
                {"name": r.name, "fires": r.fire_count, "severity": r.severity}
                for r in self.rules
            ],
        }
