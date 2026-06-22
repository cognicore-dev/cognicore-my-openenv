from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class LMEBaseMemory:
    memory_id: str = ""
    memory_type: str = ""
    text: str = ""
    session_id: str = ""
    role: str = "user"
    timestamp: float = 0.0
    source_turn_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RawTurnMemory(LMEBaseMemory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_type = "raw_turn"

@dataclass
class FactMemory(LMEBaseMemory):
    subject: str = "user"
    entity: str = ""           # favorite_color, location, employer, hobby
    value: str = ""
    confidence: float = 1.0
    status: str = "active"     # active / superseded / uncertain
    supersedes: Optional[str] = None
    
    def __init__(self, **kwargs):
        self.subject = kwargs.pop("subject", "user")
        self.entity = kwargs.pop("entity", "")
        self.value = kwargs.pop("value", "")
        self.confidence = kwargs.pop("confidence", 1.0)
        self.status = kwargs.pop("status", "active")
        self.supersedes = kwargs.pop("supersedes", None)
        kwargs["memory_type"] = "fact"
        super().__init__(**kwargs)

@dataclass
class UpdateMemory(LMEBaseMemory):
    subject: str = "user"
    entity: str = ""
    old_value: Optional[str] = None
    new_value: str = ""
    target_memory_id: Optional[str] = None

    def __init__(self, **kwargs):
        self.subject = kwargs.pop("subject", "user")
        self.entity = kwargs.pop("entity", "")
        self.old_value = kwargs.pop("old_value", None)
        self.new_value = kwargs.pop("new_value", "")
        self.target_memory_id = kwargs.pop("target_memory_id", None)
        kwargs["memory_type"] = "update"
        super().__init__(**kwargs)

@dataclass
class PreferenceEvidenceMemory(LMEBaseMemory):
    subject: str = "user"
    domain: str = ""           # food, travel, work, entertainment
    evidence_type: str = ""    # likes / dislikes / avoidance / repeated-choice
    normalized_signal: str = "" # "avoids_spicy_food"
    polarity: int = 1          # +1 or -1

    def __init__(self, **kwargs):
        self.subject = kwargs.pop("subject", "user")
        self.domain = kwargs.pop("domain", "")
        self.evidence_type = kwargs.pop("evidence_type", "")
        self.normalized_signal = kwargs.pop("normalized_signal", "")
        self.polarity = kwargs.pop("polarity", 1)
        kwargs["memory_type"] = "preference_evidence"
        super().__init__(**kwargs)

@dataclass
class PreferenceMemory(LMEBaseMemory):
    subject: str = "user"
    domain: str = ""
    preference_key: str = ""      # spicy_food, travel_style, movie_genre
    preference_value: str = ""    # dislikes_spicy / prefers_aisle / likes_sci_fi
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def __init__(self, **kwargs):
        self.subject = kwargs.pop("subject", "user")
        self.domain = kwargs.pop("domain", "")
        self.preference_key = kwargs.pop("preference_key", "")
        self.preference_value = kwargs.pop("preference_value", "")
        self.evidence_ids = kwargs.pop("evidence_ids", [])
        self.confidence = kwargs.pop("confidence", 0.0)
        kwargs["memory_type"] = "preference"
        super().__init__(**kwargs)

@dataclass
class EventMemory(LMEBaseMemory):
    event_name: str = ""
    event_type: str = ""          # workshop, webinar, trip, interview, purchase
    event_date: Optional[str] = None
    normalized_date_ts: Optional[float] = None
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None

    def __init__(self, **kwargs):
        self.event_name = kwargs.pop("event_name", "")
        self.event_type = kwargs.pop("event_type", "")
        self.event_date = kwargs.pop("event_date", None)
        self.normalized_date_ts = kwargs.pop("normalized_date_ts", None)
        self.participants = kwargs.pop("participants", [])
        self.location = kwargs.pop("location", None)
        kwargs["memory_type"] = "event"
        super().__init__(**kwargs)

@dataclass
class AssistantActionMemory(LMEBaseMemory):
    action_type: str = ""         # recommendation, summary, schedule, table, plan
    action_subject: str = ""      # what it was about
    artifact_ref: Optional[str] = None
    structured_payload: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        self.action_type = kwargs.pop("action_type", "")
        self.action_subject = kwargs.pop("action_subject", "")
        self.artifact_ref = kwargs.pop("artifact_ref", None)
        self.structured_payload = kwargs.pop("structured_payload", {})
        kwargs["memory_type"] = "assistant_action"
        super().__init__(**kwargs)

@dataclass
class ArtifactMemory(LMEBaseMemory):
    artifact_type: str = ""       # schedule, table, checklist, plan
    title: str = ""
    structured_payload: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        self.artifact_type = kwargs.pop("artifact_type", "")
        self.title = kwargs.pop("title", "")
        self.structured_payload = kwargs.pop("structured_payload", {})
        kwargs["memory_type"] = "artifact"
        super().__init__(**kwargs)
