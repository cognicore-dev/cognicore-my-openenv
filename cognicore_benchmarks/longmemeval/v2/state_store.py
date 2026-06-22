from typing import Dict, List, Any, Optional
from cognicore_benchmarks.longmemeval.v2.schema import (
    FactMemory, UpdateMemory, PreferenceEvidenceMemory, 
    PreferenceMemory, EventMemory, AssistantActionMemory, ArtifactMemory
)

class CurrentStateStore:
    def __init__(self):
        # Key: (subject, entity) e.g. ("user", "favorite_color")
        # Value: FactMemory
        self.state: Dict[tuple, FactMemory] = {}

    def update_fact(self, fact: FactMemory):
        self.state[(fact.subject, fact.entity)] = fact

    def get_fact(self, subject: str, entity: str) -> Optional[FactMemory]:
        return self.state.get((subject, entity))

    def get_all_facts(self) -> List[FactMemory]:
        return list(self.state.values())

    def clear(self):
        self.state.clear()


class PreferenceStateStore:
    def __init__(self):
        # Key: (subject, domain, preference_key)
        # Value: PreferenceMemory
        self.state: Dict[tuple, PreferenceMemory] = {}

    def update_preference(self, pref: PreferenceMemory):
        self.state[(pref.subject, pref.domain, pref.preference_key)] = pref

    def get_preference(self, subject: str, domain: str, preference_key: str) -> Optional[PreferenceMemory]:
        return self.state.get((subject, domain, preference_key))
    
    def get_all_preferences(self) -> List[PreferenceMemory]:
        return list(self.state.values())

    def clear(self):
        self.state.clear()


class TimelineStore:
    def __init__(self):
        self.events: List[EventMemory] = []

    def add_event(self, event: EventMemory):
        self.events.append(event)
        # Sort by normalized date
        self.events.sort(key=lambda x: x.normalized_date_ts if x.normalized_date_ts is not None else float('inf'))

    def get_all_events(self) -> List[EventMemory]:
        return self.events

    def clear(self):
        self.events.clear()


class ArtifactStore:
    def __init__(self):
        # Key: artifact title or id
        self.artifacts: Dict[str, ArtifactMemory] = {}

    def add_artifact(self, artifact: ArtifactMemory):
        self.artifacts[artifact.title] = artifact

    def get_all_artifacts(self) -> List[ArtifactMemory]:
        return list(self.artifacts.values())

    def clear(self):
        self.artifacts.clear()


class V2MemoryIndexes:
    def __init__(self):
        self.current_state = CurrentStateStore()
        self.preference_state = PreferenceStateStore()
        self.timeline_store = TimelineStore()
        self.artifact_store = ArtifactStore()
        
        # We will also inject the raw and typed BasicEmbeddingBackend instances from the adapter
        self.raw_store = None
        self.typed_store = None

    def clear_all(self):
        self.current_state.clear()
        self.preference_state.clear()
        self.timeline_store.clear()
        self.artifact_store.clear()
        if self.raw_store:
            self.raw_store.clear()
        if self.typed_store:
            self.typed_store.clear()
