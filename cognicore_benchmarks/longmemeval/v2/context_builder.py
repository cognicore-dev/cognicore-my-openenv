from typing import Dict, Any

def build_context(question: str, query_type: str, evidence: Dict[str, Any]) -> str:
    """Formats retrieved typed memories into explicit evidence blocks."""
    
    prompt = f"Question Type: {query_type.upper()}\n\n"
    
    if query_type == "knowledge_update":
        prompt += "Relevant user facts (Current State):\n"
        for fact in evidence["primary_evidence"]:
            prompt += f"[ACTIVE FACT | entity={fact.entity} | timestamp={fact.timestamp}]\n{fact.value}\n\n"
            
        prompt += "Related prior context (Semantic fallback):\n"
        for supp in evidence["supporting_evidence"]:
            prompt += f"[RELATED MEMORY | timestamp={supp.timestamp}]\n{supp.text}\n\n"
            
        prompt += "\nInstruction: If facts conflict, always answer using the latest ACTIVE fact. Do not use superseded facts.\n"
        
    elif query_type == "preference":
        prompt += "Preference summary:\n"
        for pref in evidence["primary_evidence"]:
            prompt += f"[PREFERENCE | domain={pref.domain} | key={pref.preference_key}]\n{pref.preference_value}\n\n"
            
        prompt += "Supporting evidence:\n"
        for supp in evidence["supporting_evidence"]:
            prompt += f"- {supp.text}\n"
            
        prompt += "\nInstruction: Answer the preference question using the summary and supporting evidence. If no explicit summary exists, infer it from the evidence.\n"
        
    elif query_type == "temporal_reasoning":
        prompt += "Event timeline:\n"
        for evt in evidence["primary_evidence"]:
            prompt += f"[{evt.event_date or 'Unknown Date'}] {evt.event_name} ({evt.event_type})\n"
            
        prompt += "\nSupporting context:\n"
        for supp in evidence["supporting_evidence"]:
            prompt += f"[{supp.timestamp}] {supp.text}\n"
            
        prompt += "\nInstruction: Use the event dates to answer which event happened first, later, or the duration between them.\n"
        
    elif query_type == "assistant_recall":
        prompt += "Assistant artifacts:\n"
        for art in evidence["primary_evidence"]:
            prompt += f"Title: {art.title}\nStructured payload: {art.structured_payload}\nRaw: {art.text}\n\n"
            
        prompt += "Supporting assistant actions:\n"
        for supp in evidence["supporting_evidence"]:
            prompt += f"[ACTION] {supp.text}\n"
            
        prompt += "\nInstruction: Answer using the assistant artifact or action history. Do not guess.\n"
        
    else: # fact_lookup
        prompt += "Explicit current state facts:\n"
        for fact in evidence["primary_evidence"]:
            prompt += f"- {fact.entity}: {fact.value}\n"
            
        prompt += "\nRaw semantic context:\n"
        for supp in evidence["supporting_evidence"]:
            prompt += f"[Fact | timestamp={supp.timestamp}] {supp.text}\n"
            
        prompt += "\nInstruction: Answer using the explicitly retrieved facts.\n"
        
    prompt += f"\nQuestion: {question}"
    return prompt
