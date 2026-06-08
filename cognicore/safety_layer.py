"""
CogniCore Safety Layer — Enterprise compliance and risk scoring.

Provides:
  - Risk scoring per action (0-100)
  - Customizable policy rules
  - Full audit log with timestamps
  - Compliance reporting

Usage::

    from cognicore.safety_layer import SafetyLayer, Policy

    safety = SafetyLayer()
    safety.add_policy(Policy("no_unsafe_confident", lambda action, ctx:
        not (action.get("classification") == "SAFE" and ctx.get("risk_score", 0) > 70),
        severity="CRITICAL"))

    result = safety.check(action, context)
    safety.audit_log()
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger("cognicore.safety_layer")


class Policy:
    """A compliance policy rule.

    Parameters
    ----------
    name : str
        Human-readable policy name.
    check_fn : callable
        Function(action, context) → bool. Returns True if policy passes.
    severity : str
        "INFO", "WARNING", "CRITICAL"
    description : str
        What this policy checks.
    """

    def __init__(
        self,
        name: str,
        check_fn: Callable[[Dict, Dict], bool],
        severity: str = "WARNING",
        description: str = "",
    ):
        self.name = name
        self.check_fn = check_fn
        self.severity = severity
        self.description = description or name

    def evaluate(self, action: Dict, context: Dict) -> bool:
        """Return True if policy passes, False if violated."""
        try:
            return self.check_fn(action, context)
        except Exception:
            return False


class AuditEntry:
    """Single audit log entry."""

    def __init__(
        self,
        timestamp: float,
        action: Dict,
        risk_score: int,
        policies_checked: int,
        violations: List[str],
        decision: str,
    ):
        self.timestamp = timestamp
        self.action = action
        self.risk_score = risk_score
        self.policies_checked = policies_checked
        self.violations = violations
        self.decision = decision


class SafetyLayer:
    """Enterprise safety and compliance layer.

    Wraps any agent with risk scoring, policy enforcement,
    and complete audit logging.
    """

    # Default risk keywords and weights
    RISK_KEYWORDS = {
        "malware": 90,
        "hack": 85,
        "exploit": 85,
        "weapon": 95,
        "bomb": 95,
        "attack": 80,
        "kill": 90,
        "drug": 75,
        "phishing": 80,
        "credential": 70,
        "password": 60,
        "violence": 85,
        "illegal": 80,
        "fraud": 75,
        "bias": 50,
        "discrimination": 60,
        "misinformation": 65,
        "rm -rf": 100,
        "sudo": 80,
        "drop table": 100,
        "delete from": 90,
    }

    def __init__(self):
        self.policies: List[Policy] = []
        self.audit: List[AuditEntry] = []
        self._stats = {
            "total_checks": 0,
            "total_violations": 0,
            "total_blocked": 0,
            "risk_scores": [],
        }

        # Add default policies
        self._add_default_policies()

    def _add_default_policies(self):
        """Built-in safety policies."""
        self.add_policy(
            Policy(
                "no_empty_action",
                lambda a, c: bool(a),
                severity="WARNING",
                description="Action must not be empty",
            )
        )

    def add_policy(self, policy: Policy) -> None:
        """Add a compliance policy."""
        self.policies.append(policy)

    # ------------------------------------------------------------------
    # Risk Scoring
    # ------------------------------------------------------------------

    def compute_risk(self, text: str) -> int:
        """Compute risk score (0-100) for given text."""
        text_lower = text.lower()
        max_risk = 0
        for keyword, risk in self.RISK_KEYWORDS.items():
            if keyword in text_lower:
                max_risk = max(max_risk, risk)
        return max_risk

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run all policies and return safety check result.

        Returns
        -------
        dict with:
          - allowed: bool
          - risk_score: int (0-100)
          - violations: list of policy names violated
          - decision: "ALLOW", "WARN", or "BLOCK"
        """
        context = context or {}
        self._stats["total_checks"] += 1

        # Compute risk
        text = str(action) + " " + str(context.get("prompt", ""))
        risk_score = self.compute_risk(text)
        context["risk_score"] = risk_score
        self._stats["risk_scores"].append(risk_score)

        # Check policies
        violations = []
        max_severity = "INFO"
        severity_order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}

        for policy in self.policies:
            if not policy.evaluate(action, context):
                violations.append(policy.name)
                self._stats["total_violations"] += 1
                if severity_order.get(policy.severity, 0) > severity_order.get(
                    max_severity, 0
                ):
                    max_severity = policy.severity

        # Decision
        if max_severity == "CRITICAL" or risk_score >= 90:
            decision = "BLOCK"
            self._stats["total_blocked"] += 1
        elif max_severity == "WARNING" or risk_score >= 60:
            decision = "WARN"
        else:
            decision = "ALLOW"

        # Audit log
        entry = AuditEntry(
            timestamp=time.time(),
            action=action,
            risk_score=risk_score,
            policies_checked=len(self.policies),
            violations=violations,
            decision=decision,
        )
        self.audit.append(entry)

        return {
            "allowed": decision != "BLOCK",
            "risk_score": risk_score,
            "decision": decision,
            "violations": violations,
            "policies_checked": len(self.policies),
        }

    # ------------------------------------------------------------------
    # Audit & Reporting
    # ------------------------------------------------------------------

    def audit_log(self, last_n: int = 20) -> List[Dict]:
        """Return recent audit entries."""
        entries = []
        for e in self.audit[-last_n:]:
            entries.append(
                {
                    "timestamp": e.timestamp,
                    "risk_score": e.risk_score,
                    "decision": e.decision,
                    "violations": e.violations,
                    "action_preview": str(e.action)[:80],
                }
            )
        return entries

    def compliance_report(self) -> Dict[str, Any]:
        """Generate compliance summary report."""
        total = self._stats["total_checks"]
        scores = self._stats["risk_scores"]

        return {
            "total_checks": total,
            "total_violations": self._stats["total_violations"],
            "total_blocked": self._stats["total_blocked"],
            "block_rate": self._stats["total_blocked"] / total if total else 0,
            "avg_risk_score": sum(scores) / len(scores) if scores else 0,
            "max_risk_score": max(scores) if scores else 0,
            "policies_active": len(self.policies),
            "audit_entries": len(self.audit),
        }

    def print_report(self):
        """Print formatted compliance report."""
        r = self.compliance_report()
        logger.info(f"\n{'=' * 50}")
        logger.info("  Safety Compliance Report")
        logger.info(f"{'=' * 50}")
        logger.info(f"  Total checks:    {r['total_checks']}")
        logger.info(f"  Violations:      {r['total_violations']}")
        logger.info(f"  Blocked:         {r['total_blocked']} ({r['block_rate']:.0%})")
        logger.info(f"  Avg risk score:  {r['avg_risk_score']:.0f}/100")
        logger.info(f"  Max risk score:  {r['max_risk_score']}/100")
        logger.info(f"  Active policies: {r['policies_active']}")
        logger.info(f"{'=' * 50}\n")
