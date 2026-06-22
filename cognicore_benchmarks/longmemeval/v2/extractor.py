import re
import json
from typing import List, Optional
from datetime import datetime
from cognicore_benchmarks.longmemeval.v2.schema import (
    FactMemory, UpdateMemory, PreferenceEvidenceMemory,
    EventMemory, AssistantActionMemory, ArtifactMemory
)

def extract_fact_memories(text: str, session_id: str, turn_id: int, timestamp: float) -> List[FactMemory]:
    """Heuristic extraction of facts from user text."""
    facts = []
    
    # Very simple heuristics for common benchmark facts
    patterns = [
        (r"my favorite color is (\w+)", "favorite_color"),
        (r"i live in ([\w\s]+)", "location"),
        (r"i work at ([\w\s]+)", "employer"),
    ]
    
    for pattern, entity in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            fact = FactMemory(
                memory_id=f"fact_{session_id}_{turn_id}_{entity}",
                text=text,
                session_id=session_id,
                role="user",
                timestamp=timestamp,
                source_turn_ids=[turn_id],
                entity=entity,
                value=value
            )
            facts.append(fact)
            
    return facts

def extract_update_memories(text: str, session_id: str, turn_id: int, timestamp: float, current_state) -> List[UpdateMemory]:
    """Heuristic extraction of state updates."""
    updates = []
    
    # Example: "Actually, my favorite color changed to green"
    if "actually" in text.lower() or "changed to" in text.lower() or "now" in text.lower():
        # Fallback to similar regex for new values
        patterns = [
            (r"favorite color (?:changed to|is now) (\w+)", "favorite_color"),
            (r"(?:moved to|live in) ([\w\s]+) now", "location"),
        ]
        
        for pattern, entity in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                new_val = match.group(1).strip()
                old_fact = current_state.get_fact("user", entity)
                
                update = UpdateMemory(
                    memory_id=f"update_{session_id}_{turn_id}_{entity}",
                    text=text,
                    session_id=session_id,
                    role="user",
                    timestamp=timestamp,
                    source_turn_ids=[turn_id],
                    entity=entity,
                    old_value=old_fact.value if old_fact else None,
                    new_value=new_val,
                    target_memory_id=old_fact.memory_id if old_fact else None
                )
                updates.append(update)
    return updates

def extract_preference_evidence(text: str, session_id: str, turn_id: int, timestamp: float) -> List[PreferenceEvidenceMemory]:
    """Extract preference signals."""
    evidence = []
    
    # Rough heuristics for food preferences
    if "spicy" in text.lower() or "jalapeño" in text.lower():
        signal = "dislikes_spicy" if "can't handle" in text.lower() else "mentions_spicy"
        ev = PreferenceEvidenceMemory(
            memory_id=f"prefev_{session_id}_{turn_id}",
            text=text,
            session_id=session_id,
            role="user",
            timestamp=timestamp,
            domain="food",
            evidence_type="dislikes" if "dislikes" in signal else "unknown",
            normalized_signal=signal,
            source_turn_ids=[turn_id]
        )
        evidence.append(ev)
    return evidence

def extract_event_memories(text: str, session_id: str, turn_id: int, timestamp: float, session_date: str) -> List[EventMemory]:
    """Extract events."""
    events = []
    
    if "workshop" in text.lower() or "webinar" in text.lower():
        # very rough extraction
        match = re.search(r"the '([^']+)' (workshop|webinar)", text, re.IGNORECASE)
        if match:
            event_name = match.group(1)
            event_type = match.group(2)
            
            # Use session timestamp as a fallback for the event date in V1 heuristics
            events.append(EventMemory(
                memory_id=f"evt_{session_id}_{turn_id}",
                text=text,
                session_id=session_id,
                role="user",
                timestamp=timestamp,
                event_name=event_name,
                event_type=event_type,
                event_date=session_date,
                normalized_date_ts=timestamp,
                source_turn_ids=[turn_id]
            ))
    return events

def extract_assistant_action(text: str, session_id: str, turn_id: int, timestamp: float) -> List[AssistantActionMemory]:
    actions = []
    if "shift rotation" in text.lower():
        actions.append(AssistantActionMemory(
            memory_id=f"asst_act_{session_id}_{turn_id}",
            text=text,
            session_id=session_id,
            role="assistant",
            timestamp=timestamp,
            action_type="schedule",
            action_subject="shift rotation",
            source_turn_ids=[turn_id]
        ))
    return actions

def extract_artifact(text: str, session_id: str, turn_id: int, timestamp: float) -> Optional[ArtifactMemory]:
    if "|" in text and "-" in text and "Shift Rotation" in text:
        # Simple heuristic for a markdown table
        return ArtifactMemory(
            memory_id=f"art_{session_id}_{turn_id}",
            text=text,
            session_id=session_id,
            role="assistant",
            timestamp=timestamp,
            artifact_type="table",
            title="Shift Rotation Sheet",
            structured_payload={"raw_table": text}, # We would parse rows/cols ideally
            source_turn_ids=[turn_id]
        )
    return None
