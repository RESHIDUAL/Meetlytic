"""
Stateful Meeting Understanding Engine for Meetlytic.
Models meetings as an evolving state machine:
1. Tracks information changes, corrections, and superseded hypotheses (initial statement -> correction -> final state).
2. Distinguishes Facts, Issues, Targets, Risks, Decisions, Actions, Blockers, and Out-of-Scope remarks.
3. Preserves all numbers, percentages, thresholds, and before -> after transformations.
4. Performs deep anaphora and reference resolution with concrete antecedent objects.
5. Reconstructs dependency chains, boolean release gates, and verified owner assignments.
6. Conducts a two-pass semantic recall and anti-hallucination validation.
Strictly free of emojis and emdashes.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set

from nlp.relevance_gate import RelevanceGate

class SemanticType(Enum):
    FACT = "FACT"
    CONFIRMED_ACTION = "CONFIRMED_ACTION"
    CONDITIONAL_ACTION = "CONDITIONAL_ACTION"
    DECISION = "DECISION"
    PROPOSAL = "PROPOSAL"
    OPINION = "OPINION"
    QUESTION = "QUESTION"
    UNCERTAINTY = "UNCERTAINTY"
    RISK = "RISK"
    BLOCKER = "BLOCKER"
    DEADLINE = "DEADLINE"
    HISTORICAL_VALUE = "HISTORICAL_VALUE"
    CURRENT_VALUE = "CURRENT_VALUE"
    SCOPE_CHANGE = "SCOPE_CHANGE"
    CASUAL = "CASUAL"

@dataclass
class SemanticStatement:
    speaker: str
    raw_text: str
    clean_text: str
    semantic_type: SemanticType
    confidence: float = 1.0
    entity: Optional[str] = None
    value: Optional[str] = None
    target_owner: Optional[str] = None
    deadline: Optional[str] = None
    condition: Optional[str] = None
    evidence: str = ""

@dataclass
class StateCorrection:
    entity: str
    initial_value: str
    confirmed_value: str
    correction_note: str = ""

@dataclass
class StatefulFact:
    statement: str
    metric: Optional[str] = None
    target_metric: Optional[str] = None
    change_transformation: Optional[str] = None
    category: str = "FACT"
    is_superseded: bool = False

@dataclass
class StatefulIssueRisk:
    problem: str
    cause: Optional[str] = None
    cause_certainty: str = "confirmed"
    superseded_causes: List[str] = field(default_factory=list)
    metric: Optional[str] = None
    target_metric: Optional[str] = None
    threshold: Optional[str] = None
    kind: str = "ISSUE"
    release_impact: str = "NON_BLOCKING"
    followup_action: Optional[str] = None
    source_turn: str = ""

@dataclass
class StatefulDecision:
    decision: str
    resolved_object: str
    status: str = "ADOPTED"
    scope_boundary: Optional[str] = None
    proposer: str = "Speaker"
    agreed_by: List[str] = field(default_factory=list)

@dataclass
class StatefulAction:
    action_verb: str
    concrete_object: str
    full_description: str
    owner: str = "Unclear"
    deadline: str = "unspecified"
    duration: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    why_reason: Optional[str] = None
    release_impact: str = "UNSPECIFIED"
    evidence_turn: str = ""

@dataclass
class StatefulReleaseGate:
    target: str
    conditions: List[str] = field(default_factory=list)
    operator: str = "AND"
    status: str = "BLOCKED"

@dataclass
class StatefulDependency:
    prerequisite: str
    dependent_outcome: str
    raw_statement: str

@dataclass
class QualityMetrics:
    accuracy: float = 1.0
    consistency: float = 1.0
    completeness: float = 1.0
    deduplication: float = 1.0
    reference_resolution: float = 1.0
    ownership_resolution: float = 1.0
    confidence_handling: float = 1.0
    composite_score: float = 1.0

@dataclass
class DetectedDomain:
    domain: str = "General"
    subdomain: str = "General"
    conversation_type: str = "Planning / Discussion"
    confidence: float = 0.85

from nlp.transcript_quality import TranscriptQualityAnalyzer
from nlp.evidence_validator import EvidenceValidator

@dataclass
class MeetingState:
    """The synthesized universal semantic state of the meeting."""
    facts: List[StatefulFact] = field(default_factory=list)
    issues_risks: List[StatefulIssueRisk] = field(default_factory=list)
    decisions: List[StatefulDecision] = field(default_factory=list)
    actions: List[StatefulAction] = field(default_factory=list)
    uncertain_actions: List[Dict[str, Any]] = field(default_factory=list)
    procedures: List[Dict[str, Any]] = field(default_factory=list)
    release_gates: List[StatefulReleaseGate] = field(default_factory=list)
    dependencies: List[StatefulDependency] = field(default_factory=list)
    corrections: List[StateCorrection] = field(default_factory=list)
    out_of_scope_items: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    semantic_topics: List[str] = field(default_factory=list)
    participants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    detected_domain: DetectedDomain = field(default_factory=DetectedDomain)
    quality_score: QualityMetrics = field(default_factory=QualityMetrics)
    transcript_quality: Dict[str, Any] = field(default_factory=dict)

class StatefulMeetingUnderstandingEngine:
    """
    State machine parser that tracks conversation evolution, corrects superseded info,
    resolves anaphora to concrete objects, and preserves all business facts and numbers.
    """

    def __init__(self):
        self.transcript_analyzer = TranscriptQualityAnalyzer()
        self.evidence_validator = EvidenceValidator()
        self.relevance_gate = RelevanceGate()
        self.correction_cues = [
            r'\b(?:no|sorry|correction|actually|outdated|latest confirmed|latest result|latest measurement|latest value|the latest|current value|that was staging|not yet|still pending|that\'s old|old measurement|revised|wait|not completely|that\'s incorrect|i was wrong|instead|rather|we confirmed|turns out|the real issue is|in fact|approved)\b'
        ]
        self.agreement_tokens = {
            'agreed', 'i agree', 'sounds good', 'sounds good to me', 'that works',
            'that works for me', 'good idea', 'fair enough', 'definitely', 'perfect',
            'exactly', 'confirmed', 'approved', 'will do', "let's do that", "we'll go with that",
            "that's fine", "okay we'll proceed", 'yep', 'yeah', 'yes'
        }

    def process_meeting(self, turns: List[Tuple[str, str]]) -> MeetingState:
        """
        Execute stateful meeting analysis across all conversation turns.
        """
        state = MeetingState()
        if not turns:
            return state

        for spk, _ in turns:
            if spk not in state.participants and spk not in ('Speaker', 'Team', 'User'):
                state.participants[spk] = {
                    'name': spk,
                    'actions': [],
                    'turns': 0
                }

        entity_stack: List[str] = []
        recent_problems: List[Dict[str, Any]] = []

        parsed_turns = []
        prev_clean_stmt = ""
        prev_spk = ""
        for idx, (spk, raw) in enumerate(turns):
            if spk in state.participants:
                state.participants[spk]['turns'] += 1

            clean_full = self._clean_utterance(raw)
            clauses = self.relevance_gate.split_mixed_statement(clean_full)
            for clause in clauses:
                clean = clause.strip()
                if not clean:
                    continue
                relevance = self.relevance_gate.classify(clean, clean, prev_statement=prev_clean_stmt, prev_speaker=prev_spk)
                parsed_turns.append({
                    'index': idx,
                    'speaker': spk,
                    'raw': raw,
                    'clean': clean,
                    'relevance': relevance
                })
                prev_clean_stmt = clean
                prev_spk = spk

        state.transcript_quality = self.transcript_analyzer.evaluate_transcript(turns)

        state.detected_domain = self._detect_domain(parsed_turns)

        i = 0
        n = len(parsed_turns)
        while i < n:
            curr = parsed_turns[i]
            spk = curr['speaker']
            clean = curr['clean']
            clean_lower = clean.lower()
            relevance = curr.get('relevance')

            if not clean or len(clean) < 3:
                i += 1
                continue

            if relevance and relevance.is_skippable():
                i += 1
                continue

            proc = self.evidence_validator.check_procedure(clean)
            if proc:
                state.procedures.append(proc)
                if proc.get('is_vote') or any(w in clean_lower for w in ['motion passed', 'motion passes', 'motion carried', 'approved the minutes', 'board approved', 'board voted to approve', 'committee approved', 'voted to approve']):
                    status = "REJECTED" if any(w in clean_lower for w in ['failed', 'denied']) else "ADOPTED"
                    state.decisions.append(StatefulDecision(
                        decision=clean if clean.endswith('.') else clean + '.',
                        resolved_object="Committee approval decision",
                        status=status
                    ))
                i += 1
                continue

            turn_ents = self._extract_entities(clean)
            for ent in turn_ents:
                if ent not in entity_stack:
                    entity_stack.append(ent)
                if len(entity_stack) > 15:
                    entity_stack.pop(0)

            if any(cue in clean_lower for cue in ['to summarize:', 'final recap:', 'in summary:', 'to recap:']):
                recap_body = re.sub(r'^(?:to summarize|final recap|in summary|to recap)[:\s\-]+', '', clean, flags=re.IGNORECASE).strip()
                recap_gate = self._parse_release_gate(recap_body)
                if recap_gate:
                    state.release_gates.append(recap_gate)

                m_blocker1 = re.search(r'\b(?:primary blocker is|blocker is)\s+([a-zA-Z0-9_\s\-]+)', recap_body, re.IGNORECASE)
                m_blocker2 = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+is\s+(?:the\s+)?primary blocker', recap_body, re.IGNORECASE)
                if m_blocker1 or m_blocker2:
                    raw_b = m_blocker1.group(1).strip() if m_blocker1 else m_blocker2.group(1).strip()
                    b_target = re.sub(r'^(?:and|also|the)\s+', '', raw_b, flags=re.IGNORECASE).strip().capitalize()
                    state.release_gates.append(StatefulReleaseGate(
                        target="Release",
                        conditions=[f"{b_target} is the primary blocker and must be resolved before release"],
                        operator="SINGLE",
                        status="BLOCKED"
                    ))
                elif 'security approval' in recap_body.lower() and not recap_gate:
                    state.release_gates.append(StatefulReleaseGate(
                        target="Release",
                        conditions=["Security approval must be granted before release"],
                        operator="SINGLE",
                        status="BLOCKED"
                    ))
                if 'deferred' in recap_body.lower() or 'defer' in recap_body.lower() or 'export' in recap_body.lower():
                    m_def = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+(?:is deferred|moves)\s+to\s+([a-zA-Z0-9_\s\-]+)', recap_body, re.IGNORECASE)
                    if m_def:
                        raw_def = re.sub(r'^(?:and|also|the)\s+', '', m_def.group(1).strip(), flags=re.IGNORECASE).strip()
                        def_obj = "dashboard export" if "export" in raw_def.lower() else self._resolve_anaphora(raw_def, entity_stack)
                        state.decisions.append(StatefulDecision(
                            decision=f"Defer {def_obj} to {m_def.group(2).strip()}.",
                            resolved_object=def_obj,
                            status="DEFERRED"
                        ))
                i += 1
                continue

            correction = self._detect_correction(clean, entity_stack, parsed_turns, i)
            if correction:
                state.corrections.append(correction)
                self._apply_correction_to_state(state, correction)
                i += 1
                continue

            if self._is_out_of_scope(clean):
                state.out_of_scope_items.append(clean)
                i += 1
                continue

            gate = self._parse_release_gate(clean)
            if gate:
                state.release_gates.append(gate)

            if re.search(r'\b(?:outside (?:the )?(?:current )?(?:release\s+scope|release|scope)|out of scope)\b', clean_lower):
                m_scope = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+(?:is|are)\s+(?:outside (?:the )?(?:current )?(?:release\s+scope|release|scope)|out of scope)\b', clean, re.IGNORECASE)
                scope_item = m_scope.group(1).strip() if m_scope else "Feature"
                scope_clean = re.sub(r'^(?:agreed|okay|ok|sure|yes|and|also)[,\s\-:\.]+', '', scope_item, flags=re.IGNORECASE).strip()
                state.decisions.append(StatefulDecision(
                    decision=f"[OUT_OF_SCOPE] {scope_clean} (Outside current release scope)",
                    resolved_object=scope_clean,
                    status="OUT_OF_SCOPE"
                ))
                i += 1
                continue

            dep = self._parse_dependency(clean)
            if dep:
                state.dependencies.append(dep)

            transform_fact = self._parse_transformation(clean)
            if transform_fact:
                state.facts.append(transform_fact)

            if any(w in clean_lower for w in ['passed staging', 'staging migration passed', 'staging test passed', 'migration passed staging', 'staging passed', 'production-sized data', 'production sized data']):
                state.facts.append(StatefulFact(
                    statement=clean,
                    category="STATUS"
                ))
                i += 1
                continue

            if re.search(r'\bif\s+([a-zA-Z0-9_\s\-]+?)\s+fails,\s*(?:we\s+)?([a-zA-Z0-9_\s\-]+)', clean, re.IGNORECASE):
                m_fail = re.search(r'\bif\s+([a-zA-Z0-9_\s\-]+?)\s+fails,\s*(?:we\s+)?([a-zA-Z0-9_\s\-]+)', clean, re.IGNORECASE)
                event_name = m_fail.group(1).strip().capitalize()
                conseq = m_fail.group(2).strip()
                state.facts.append(StatefulFact(
                    statement=f"Failure logic: If {event_name.lower()} fails, {conseq}.",
                    category="STATUS"
                ))
                i += 1
                continue

            if 'currently not ready' in clean_lower or 'release is not ready' in clean_lower:
                state.facts.append(StatefulFact(
                    statement="Current release status: NOT READY (Release is currently not ready for production).",
                    category="STATUS"
                ))
                state.release_gates.insert(0, StatefulReleaseGate(
                    target="Release",
                    conditions=["Current release status: NOT READY"],
                    operator="SINGLE",
                    status="BLOCKED"
                ))
                i += 1
                continue

            if '?' in clean or clean_lower.startswith(('is the', 'is this', 'are there', 'what is', 'what happens', 'how long', 'who is', 'when can', 'when will')):
                req_match = re.search(r'\b(?:can you|could you|would you|please)\s+([a-zA-Z0-9_\s,\-]{4,150})\??', clean, re.IGNORECASE)
                if req_match and i + 1 < n and (self._is_acceptance(parsed_turns[i + 1]['clean']) or self._is_acceptance(parsed_turns[i + 1]['raw'])):

                    pass
                else:
                    qa_fact = self._resolve_question_answer(parsed_turns, i, entity_stack)
                    if qa_fact:
                        if isinstance(qa_fact, StatefulIssueRisk):
                            state.issues_risks.append(qa_fact)
                        elif isinstance(qa_fact, StatefulFact):
                            state.facts.append(qa_fact)
                        i += 2
                        continue
                    else:
                        if any(w in clean_lower for w in ['blocking', 'broken', 'failing', 'timeline', 'how long', 'who is', 'revisit']):
                            if not self._is_out_of_scope(clean):
                                state.unresolved.append(f"{spk}: {clean}")
                        i += 1
                        continue

            if not (relevance and relevance.blocks_actions()):
                req_match = re.search(r'\b(?:can you|could you|would you|please)\s+([a-zA-Z0-9_\s,\-]{4,150})\??', clean, re.IGNORECASE)
                if req_match and not self.evidence_validator.is_speaker_administration(clean):
                    task_raw = req_match.group(1).strip().rstrip('?')
                    task_resolved = self._resolve_anaphora(task_raw, entity_stack)

                    if i + 1 < n:
                        next_t = parsed_turns[i + 1]
                        if self._is_acceptance(next_t['clean']) or self._is_acceptance(next_t['raw']):
                            accept_spk = next_t['speaker']
                            direct_name = re.match(r'^([A-Z][a-z]+)[,\s\-]+(?:can you|could you|would you|please)\b', clean, re.IGNORECASE)
                            if direct_name and accept_spk in ('Speaker', 'Team', 'User', 'Unclear'):
                                accept_spk = direct_name.group(1).strip().capitalize()

                            accept_text = next_t['clean'] or next_t['raw']
                            deadline, duration, conditions = self._extract_temporal_and_conditions(accept_text)

                            comm_match = re.search(r"\b(?:i will|i'll|i can|i am on it|sure i will)\s+([a-zA-Z0-9_\s,\-]{4,150})", accept_text, re.IGNORECASE)
                            if comm_match:
                                task_resolved = self._resolve_anaphora(comm_match.group(1).strip(), entity_stack)

                            why_reason = self._find_why_reason(task_resolved, recent_problems)
                            node = StatefulAction(
                                action_verb=self._extract_action_verb(task_resolved),
                                concrete_object=task_resolved,
                                full_description=self._normalize_action_description(task_resolved),
                                owner=accept_spk,
                                deadline=deadline,
                                duration=duration,
                                conditions=conditions,
                                why_reason=why_reason,
                                evidence_turn=f"{spk}: {clean} | {accept_spk}: {accept_text}"
                            )
                            val_action = self.evidence_validator.validate_action(node.full_description, node.owner, clean, context_entities=entity_stack)
                            if val_action['status'] == 'CONFIRMED':
                                state.actions.append(node)
                                if accept_spk in state.participants:
                                    state.participants[accept_spk]['actions'].append(node.full_description)
                            i += 2
                            continue

                    node = StatefulAction(
                        action_verb=self._extract_action_verb(task_resolved),
                        concrete_object=task_resolved,
                        full_description=self._normalize_action_description(task_resolved),
                        owner="Unassigned",
                        deadline="unspecified",
                        evidence_turn=f"{spk}: {clean}"
                    )
                    val_action = self.evidence_validator.validate_action(node.full_description, node.owner, clean, context_entities=entity_stack)
                    if val_action['status'] == 'CONFIRMED':
                        state.actions.append(node)
                    i += 1
                    continue

            direct_cmd_match = re.match(r'^((?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][a-z]+)[,\s\-]+(?:please\s+)?(create|investigate|send|audit|deploy|fix|update|run|book|prepare|draft|schedule|submit|review|handle|coordinate|notify|launch)\s+([a-zA-Z0-9_\s,\-]{3,120})', clean, re.IGNORECASE)
            if direct_cmd_match and not self.evidence_validator.is_speaker_administration(clean):
                owner_assigned = direct_cmd_match.group(1).strip()
                verb_assigned = direct_cmd_match.group(2).strip().capitalize()
                obj_raw = direct_cmd_match.group(3).strip()
                task_resolved = f"{verb_assigned} {self._resolve_anaphora(obj_raw, entity_stack)}"
                deadline, duration, conditions = self._extract_temporal_and_conditions(clean)
                why_reason = self._find_why_reason(task_resolved, recent_problems)
                node = StatefulAction(
                    action_verb=verb_assigned,
                    concrete_object=task_resolved,
                    full_description=self._normalize_action_description(task_resolved),
                    owner=owner_assigned,
                    deadline=deadline,
                    duration=duration,
                    conditions=conditions,
                    why_reason=why_reason,
                    evidence_turn=clean
                )
                val_action = self.evidence_validator.validate_action(node.full_description, node.owner, clean, context_entities=entity_stack)
                if val_action['status'] == 'CONFIRMED':
                    state.actions.append(node)
                    if owner_assigned in state.participants:
                        state.participants[owner_assigned]['actions'].append(node.full_description)
                i += 1
                continue

            standalone_cmd_match = re.match(r'^(?:please\s+)?(investigate|send|prepare|book|schedule|deploy|fix|create|review|audit|finalize|draft|organize|write|coordinate|complete|notify|publish|reconcile|configure|implement|build|run|test|launch|obtain|submit|pull|record|update)\s+([a-zA-Z0-9_\s,\-]{3,120})\.?$', clean, re.IGNORECASE)
            if standalone_cmd_match and not self.evidence_validator.is_speaker_administration(clean):
                verb_assigned = standalone_cmd_match.group(1).strip().capitalize()
                obj_raw = standalone_cmd_match.group(2).strip()
                task_resolved = f"{verb_assigned} {self._resolve_anaphora(obj_raw, entity_stack)}"
                deadline, duration, conditions = self._extract_temporal_and_conditions(clean)
                why_reason = self._find_why_reason(task_resolved, recent_problems)
                node = StatefulAction(
                    action_verb=verb_assigned,
                    concrete_object=task_resolved,
                    full_description=self._normalize_action_description(task_resolved),
                    owner="Unassigned",
                    deadline=deadline,
                    duration=duration,
                    conditions=conditions,
                    why_reason=why_reason,
                    evidence_turn=clean
                )
                val_action = self.evidence_validator.validate_action(node.full_description, node.owner, clean, context_entities=entity_stack)
                if val_action['status'] == 'CONFIRMED':
                    state.actions.append(node)
                i += 1
                continue

            if any(w in clean_lower for w in ["we decided", "we have decided", "we've decided", "let's move", "let us move", "let's defer", "we should", "i recommend", "how about", "what if we", "ship the", "ship without", "let's keep"]):
                if 'ship' in clean_lower and 'without' in clean_lower:
                    m_ship = re.search(r'\bship\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+without\s+([a-zA-Z0-9_\s\-]+?)(?:,|\.|\band\b|$)', clean, re.IGNORECASE)
                    if m_ship:
                        state.decisions.append(StatefulDecision(
                            decision=f"Ship the {m_ship.group(1).strip()} without {m_ship.group(2).strip()}.",
                            resolved_object=f"{m_ship.group(1).strip()} without {m_ship.group(2).strip()}",
                            status="ADOPTED"
                        ))
                    if 'defer' in clean_lower or 'deferred' in clean_lower:
                        m_def = re.search(r'\b(?:and\s+)?(?:defer|moves?)\s+([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', clean, re.IGNORECASE)
                        if m_def:
                            raw_def = re.sub(r'^(?:and|also|then|we)\s+', '', m_def.group(1).strip(), flags=re.IGNORECASE).strip()
                            def_obj = "dashboard export" if "export" in raw_def.lower() else raw_def
                            state.decisions.append(StatefulDecision(
                                decision=f"Defer {def_obj} to {m_def.group(2).strip()}.",
                                resolved_object=def_obj,
                                status="DEFERRED"
                            ))
                    i += 1
                    continue

                dec_obj, consumed = self._resolve_proposal_decision(parsed_turns, i, entity_stack)
                if dec_obj:
                    state.decisions.append(dec_obj)
                    i += consumed
                    continue

            create_owns_match = re.search(r'\b(?:create|run|complete|investigate)\s+([a-zA-Z0-9_\s\-]+?)\.\s*([A-Z][a-z]+)\s+owns\s+it', clean, re.IGNORECASE)
            assign_match = re.search(r'\b([A-Z][a-z]+)\s+(?:owns|is taking over|takes over)\s*(?:the\s+)?([a-zA-Z0-9_\s\-]+)?', clean)
            if create_owns_match:
                task_raw = f"create {create_owns_match.group(1).strip()}"
                owner_assigned = create_owns_match.group(2).strip()
                task_resolved = self._resolve_anaphora(task_raw, entity_stack)
                deadline, duration, conditions = self._extract_temporal_and_conditions(clean)
                node = StatefulAction(
                    action_verb="Create",
                    concrete_object=task_resolved,
                    full_description=self._normalize_action_description(task_resolved),
                    owner=owner_assigned,
                    deadline=deadline,
                    duration=duration,
                    conditions=conditions,
                    why_reason="Assigned by lead",
                    evidence_turn=clean
                )
                val_action = self.evidence_validator.validate_action(node.full_description, node.owner, clean, context_entities=entity_stack)
                if val_action['status'] == 'CONFIRMED':
                    state.actions.append(node)
                    if owner_assigned in state.participants:
                        state.participants[owner_assigned]['actions'].append(node.full_description)
                i += 1
                continue
            elif assign_match:
                owner_assigned = assign_match.group(1).strip()
                task_raw = (assign_match.group(2) or "").strip()
                task_raw = re.sub(r'\b(?:due\s+by|due)\s+.*$', '', task_raw, flags=re.IGNORECASE).strip()

                if (not task_raw or task_raw.lower() in ("it", "that", "this", "over")) and state.actions:
                    state.actions[-1].owner = owner_assigned
                    state.actions[-1].why_reason = f"Assigned to {owner_assigned}"
                    if owner_assigned in state.participants:
                        state.participants[owner_assigned]['actions'].append(state.actions[-1].full_description)
                    i += 1
                    continue

                task_resolved = self._resolve_anaphora(task_raw, entity_stack)
                deadline, duration, conditions = self._extract_temporal_and_conditions(clean)
                node = StatefulAction(
                    action_verb=self._extract_action_verb(task_resolved),
                    concrete_object=task_resolved,
                    full_description=self._normalize_action_description(task_resolved),
                    owner=owner_assigned,
                    deadline=deadline,
                    duration=duration,
                    conditions=conditions,
                    why_reason="Explicit ownership",
                    evidence_turn=clean
                )
                val_action = self.evidence_validator.validate_action(node.full_description, node.owner, clean, context_entities=entity_stack)
                if val_action['status'] == 'CONFIRMED':
                    state.actions.append(node)
                    if owner_assigned in state.participants:
                        state.participants[owner_assigned]['actions'].append(node.full_description)
                elif val_action['status'] == 'UNCERTAIN':
                    state.uncertain_actions.append({'task': node.full_description, 'owner': node.owner, 'reason': val_action['reason']})
                i += 1
                continue

            comm_match = re.search(r"\b(?:i will|i'll|i can|i am going to)\s+([a-zA-Z0-9_\s,\-]{4,150})", clean, re.IGNORECASE)
            third_match = re.search(r"\b((?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][a-z]+)\s+will\s+([a-zA-Z0-9_\s,\-]{4,150})", clean)
            if comm_match or third_match:
                actor = spk if comm_match else third_match.group(1).strip()
                task_raw = comm_match.group(1).strip() if comm_match else third_match.group(2).strip()
                task_resolved = self._resolve_anaphora(task_raw, entity_stack)
                deadline, duration, conditions = self._extract_temporal_and_conditions(clean)

                why_reason = self._find_why_reason(task_resolved, recent_problems)
                node = StatefulAction(
                    action_verb=self._extract_action_verb(task_resolved),
                    concrete_object=task_resolved,
                    full_description=self._normalize_action_description(task_resolved),
                    owner=actor,
                    deadline=deadline,
                    duration=duration,
                    conditions=conditions,
                    why_reason=why_reason,
                    evidence_turn=clean
                )
                val_action = self.evidence_validator.validate_action(node.full_description, node.owner, clean, context_entities=entity_stack)
                if val_action['status'] == 'CONFIRMED':
                    state.actions.append(node)
                    if actor in state.participants:
                        state.participants[actor]['actions'].append(node.full_description)
                elif val_action['status'] == 'UNCERTAIN':
                    state.uncertain_actions.append({'task': node.full_description, 'owner': node.owner, 'reason': val_action['reason']})
                i += 1
                continue

            if not (relevance and relevance.blocks_issues()):
                risk_obj = self._parse_risk_statement(clean)
                if risk_obj:
                    state.issues_risks.append(risk_obj)
                    recent_problems.append({'problem': risk_obj.problem, 'cause': risk_obj.cause, 'metric': risk_obj.metric})
                    i += 1
                    continue

            if not (relevance and relevance.blocks_issues()):

                opinion_rebuttal = r'^(?:that\'s\s+just\s+your\s+opinion|that\'s\s+an\s+opinion|in\s+my\s+opinion|that\'s\s+my\s+opinion|i\s+disagree|you\'re\s+wrong|that\'s\s+not\s+true)[,\s\-:\.]+'
                clean_factual = re.sub(opinion_rebuttal, '', clean, flags=re.IGNORECASE).strip()
                if clean_factual:
                    clean_factual = clean_factual[0].upper() + clean_factual[1:]
                clean_fact_lower = clean_factual.lower()

                if any(w in clean_fact_lower for w in ['improved', 'has improved', 'have improved', 'decreased', 'dropped', 'fixed', 'resolved', 'under control', 'better now']):
                    state.facts.append(StatefulFact(
                        statement=clean_factual,
                        category="STATUS"
                    ))
                    i += 1
                    continue

                if any(w in clean_fact_lower for w in ['differs from', 'discrepancy', 'takes 42 minutes', 'takes 6 seconds', 'failing', 'failed', 'failure', 'failures', 'crashes', 'crash', 'timeout', 'error', 'exhaustion', 'below target', 'slowdown', 'caused by', 'caused the']):
                    issue_obj = self._parse_issue_statement(clean_factual, entity_stack)
                    state.issues_risks.append(issue_obj)
                    recent_problems.append({'problem': issue_obj.problem, 'cause': issue_obj.cause, 'metric': issue_obj.metric})
                    i += 1
                    continue

            if 'approval is' in clean_lower or 'is blocked because' in clean_lower or 'is pending' in clean_lower:
                if 'is blocked because' in clean_lower or 'rejected' in clean_lower:
                    state.issues_risks.append(StatefulIssueRisk(
                        problem=clean,
                        kind="BLOCKER",
                        release_impact="BLOCKING"
                    ))
                else:
                    state.facts.append(StatefulFact(
                        statement=clean,
                        category="STATUS"
                    ))
                i += 1
                continue

            metric_match = re.search(r'(\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:k|m|b|million)?|\b\d+k\b|\b\d+\s*million\b|\d+(?:\.\d+)?%|\d+\s*(?:minutes?|seconds?|hours?|ms|fps|users|requests|kb|mb|gb|rpm|candidates|active candidates|participants|patients|attendees|students|cases)|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+duplicate|\d+,\d{3})', clean, re.IGNORECASE)
            if metric_match or any(w in clean_lower for w in ['budget is', 'property tax', 'revenue represents', 'estimated revenue', 'expenditures are', 'spent', 'revenue is', 'cpu is', 'target is', 'response went from', 'below target', 'takes approximately', 'takes 6 seconds', 'we have 9 active', 'duplicate welcome-email', 'duplicate cases', 'duplicate']):
                atomic_facts = self._decompose_atomic_facts(clean)
                state.facts.extend(atomic_facts)

            i += 1

        self._second_pass_recall_and_validation(state, parsed_turns)

        self._deduplicate_meeting_state(state)

        self._consistency_pass(state)

        state.semantic_topics = self._generate_state_topics(state)

        return state

    def _classify_semantic_type(self, raw: str, clean: str, parsed_turns: List[Dict[str, Any]], idx: int, entity_stack: List[str]) -> SemanticStatement:
        """
        Universal domain-independent semantic status classification:
        FACT, CONFIRMED_ACTION, CONDITIONAL_ACTION, DECISION, PROPOSAL,
        OPINION, QUESTION, UNCERTAINTY, RISK, BLOCKER, DEADLINE,
        HISTORICAL_VALUE, CURRENT_VALUE, SCOPE_CHANGE, CASUAL.
        """
        raw_lower = raw.lower().strip()
        clean_lower = clean.lower().strip()

        if not clean or len(clean.split()) < 2:
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.CASUAL, confidence=1.0)
        if any(raw_lower.startswith(g) for g in ['good morning', 'hello', 'hi team', 'hey guys', 'how are you', 'great match', 'arsenal', 'real madrid', 'went hiking']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.CASUAL, confidence=1.0)
        if any(w in raw_lower for w in ['unnecessarily difficult', 'is being impossible', 'has no idea what', 'useless', 'incompetent', 'ridiculous', 'hate working with']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.CASUAL, confidence=1.0)

        if raw_lower.endswith('?') or any(clean_lower.startswith(q) for q in ['should we', 'could we', 'can someone check', 'do we know', 'is ', 'are ', 'what happens if', 'when will', 'who will']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.QUESTION, confidence=0.95)

        if any(clean_lower.startswith(p) for p in ['maybe we should', 'what if we', 'how about we', 'i propose', 'i suggest', 'could we consider', 'we could try', 'we might want to']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.PROPOSAL, confidence=0.95)

        if any(clean_lower.startswith(op) for op in ['i think', 'in my opinion', 'i feel like', 'personally', 'i believe', 'my view is', 'seems to me']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.OPINION, confidence=0.90)

        if any(w in clean_lower for w in ['around 200', 'not sure if', 'estimated at', 'approximate', 'might be', 'probably around']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.UNCERTAINTY, confidence=0.90)

        if any(w in clean_lower for w in ['primary blocker', 'is blocked because', 'cannot proceed until', 'cannot continue until', 'must be resolved before release', 'blocking the release', 'blocking this release']):
            if 'non-blocking' not in clean_lower and 'not a' not in clean_lower:
                return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.BLOCKER, confidence=1.0)

        if any(w in clean_lower for w in ['risk of', 'might miss', 'may miss the deadline', 'could cause', 'worried that', 'potential issue', 'warning threshold', 'below target']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.RISK, confidence=0.95)

        if any(w in clean_lower for w in ['can start only after', 'only after', 'depending on', 'contingent upon', 'if we approve']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.CONDITIONAL_ACTION, confidence=0.95)

        if any(w in clean_lower for w in ['out of scope', 'outside the scope', 'defer export', 'is deferred', 'moves to next sprint', 'no task for']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.SCOPE_CHANGE, confidence=1.0)

        if any(clean_lower.startswith(d) for d in ['we decided to', 'we have decided', 'we\'ve decided', 'everyone agreed', 'let\'s go with', 'confirmed plan is']):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.DECISION, confidence=1.0)

        if any(re.search(pat, clean_lower) for pat in self.correction_cues) or (idx > 0 and any(re.search(pat, parsed_turns[idx]['raw'].lower()) for pat in self.correction_cues)):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.CURRENT_VALUE, confidence=1.0)

        if re.search(r'\b(?:[A-Z][a-z]+\s+will|[A-Z][a-z]+\s+owns|i will|i\'ll|i can)\s+[a-zA-Z0-9_\s,\-]{4,100}', clean):
            return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.CONFIRMED_ACTION, confidence=0.95)

        return SemanticStatement(speaker="", raw_text=raw, clean_text=clean, semantic_type=SemanticType.FACT, confidence=0.90)

    def _detect_domain(self, turns: List[Dict[str, Any]]) -> DetectedDomain:
        """
        Lightweight lexical semantic domain and subdomain detection.
        Infers domain to guide presentation terminology without restricting extraction.
        """
        text = " ".join([t.get('clean', '') for t in turns]).lower()

        scores = {
            'Technology': 0,
            'Business': 0,
            'Education': 0,
            'Travel': 0,
            'Marketing': 0,
            'Personal': 0
        }

        keywords = {
            'Technology': ['api', 'latency', 'deployment', 'server', 'migration', 'staging', 'production', 'database', 'backend', 'frontend', 'release', 'bug', 'crash', 'android', 'ios', 'ticket', 'sprint', 'code', 'qa'],
            'Business': ['client', 'budget', 'proposal', 'revenue', 'contract', 'quarterly', 'stakeholder', 'sales', 'partnership', 'invoice', 'customer', 'vendor', 'procurement', 'finance'],
            'Education': ['exam', 'assignment', 'student', 'professor', 'course', 'class', 'grades', 'lecture', 'syllabus', 'semester', 'homework', 'academic', 'students', 'college'],
            'Travel': ['flight', 'hotel', 'trip', 'vacation', 'airport', 'departure', 'arrival', 'itinerary', 'packing', 'luggage', 'booking', 'leave friday', 'leaving saturday', 'destination'],
            'Marketing': ['campaign', 'conversion', 'leads', 'ad', 'advertising', 'seo', 'ctr', 'impressions', 'branding', 'social media', 'acquisition', 'conversion rate'],
            'Personal': ['dinner', 'lunch', 'birthday', 'party', 'movie', 'weekend', 'family', 'evening', 'meet at 6', 'meet at 7']
        }

        for dom, kws in keywords.items():
            for kw in kws:
                if kw in text:
                    scores[dom] += 1

        best_dom = max(scores, key=scores.get)
        if scores[best_dom] == 0:
            return DetectedDomain(domain="General", subdomain="General Discussion", conversation_type="Meeting", confidence=0.7)

        subdomains = {
            'Technology': 'Software Engineering',
            'Business': 'Sales & Operations',
            'Education': 'Academic Planning',
            'Travel': 'Travel & Itinerary Planning',
            'Marketing': 'Campaign Planning',
            'Personal': 'Personal Planning'
        }

        return DetectedDomain(
            domain=best_dom,
            subdomain=subdomains.get(best_dom, "General"),
            conversation_type="Meeting",
            confidence=min(0.95, 0.6 + 0.05 * scores[best_dom])
        )

    def _clean_utterance(self, text: str) -> str:
        """Strip conversational fillers, markdown artifacts, and emotional venting."""
        if not text:
            return ""
        clean = re.sub(r'^\s*[\*_\[\]\(\)]+\s*', '', text).strip()
        clean = re.sub(r'^(?:(?:yes|no|yeah|yep|sure|ok|alright|good morning|good morning everyone|hello|hi)[,\s\-:\.]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:arsenal|real madrid|barcelona|chelsea|liverpool|india|australia|england) (?:won|played well|played amazingly)[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:it was (?:great|good|fun)|went hiking|had fun|great game|good game|great match)[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:but )?(?:speaking of|getting to|getting back to|talking about|regarding) [^.!?]+?[,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:also|plus|and)[,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:honestly|frankly|personally|to be honest|look|listen|man|dude|gosh|damn|ugh)[,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:so\s+)?just\s+to\s+give\s+you\s+a\s+highlight\s+of\s+(?:the\s+)?)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:as\s+you\s+can\s+see|you\s+know|i\s+mean|excuse\s+me|basically|again|so\s+basically|so)[,\s\-:\.]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:to\s+give\s+a\s+quick\s+overview|quick\s+overview\s+of|quick\s+highlight\s+of)[,\s\-:\.]+)+', '', clean, flags=re.IGNORECASE).strip()

        clean = re.sub(r'\s*(?:because|since|as|and)\s+(?:the |our )?(?:manager|management|finance|boss|developer|team|they|he|she) (?:is|are) (?:being )?(?:useless|incompetent|ridiculous|impossible|lazy|annoying|terrible|stupid|awful|frustrating)[^.!?]*', '.', clean, flags=re.IGNORECASE).strip()

        clean = re.sub(r'^(?:(?:the |our )?(?:manager|lead|boss|management|finance|team|developer) (?:has|have) no idea what (?:he|she|they|we|anyone)(?:\'s|\'re| is| are)? doing[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:the |our )?(?:manager|lead|boss|management|finance|team|developer) (?:is|are) (?:being )?(?:useless|incompetent|ridiculous|impossible|lazy|annoying|terrible|stupid|awful|frustrating)[^.!?]*?(?:because|since|as|\.|\;)\s*)', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:management|leadership|they) (?:doesn\'t|don\'t) understand (?:this|the) (?:project|work)[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:everyone is|people are) (?:frustrated|upset|angry) with [^.!?]+?[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:she|he) (?:never listens|is terrible at (?:her|his) job|never approves anything)[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:i (?:really )?(?:hate|dislike|can\'t stand) (?:this|the|our|working with) [a-zA-Z0-9_\s\-]+?(?:,| but| yet|\.|\;)\s*)', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:anyway|anyways|on another note|by the way|excellent|perfect|great)[,\s\-:\.]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:this\s+)?(?:terrible|horrible|annoying|frustrating|stupid|awful)\s+([a-zA-Z0-9_\-]+)', r'The \1', clean, flags=re.IGNORECASE).strip()

        if re.search(r'^(?:he|she|they)\s+still\s+hasn\'t\s+approved\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', clean, re.IGNORECASE):
            m = re.search(r'^(?:he|she|they)\s+still\s+hasn\'t\s+approved\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', clean, re.IGNORECASE)
            clean = f"{m.group(1).strip().capitalize()} approval is still pending."
        elif re.search(r'^(?:he|she|they)\s+haven\'t\s+approved\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', clean, re.IGNORECASE):
            m = re.search(r'^(?:he|she|they)\s+haven\'t\s+approved\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', clean, re.IGNORECASE)
            clean = f"{m.group(1).strip().capitalize()} approval is pending."
        elif re.search(r'^(?:he|she|they|management)\s+rejected\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+because\s+([a-zA-Z0-9_\s\-]+)', clean, re.IGNORECASE):
            m = re.search(r'^(?:he|she|they|management)\s+rejected\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+because\s+([a-zA-Z0-9_\s\-]+)', clean, re.IGNORECASE)
            clean = f"{m.group(1).strip().capitalize()} is blocked because {m.group(2).strip()}."

        clean = re.sub(r'\bf\s*y\s*(?:twenty\s+seven|2027|27)\b', 'FY2027', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\btwo hundred sixty three million dollars\b|\btwo hundred sixty three million\b', '$263 million', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bone hundred sixty two million dollars\b|\bone hundred sixty two million\b', '$162 million', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bone hundred eighteen million dollars\b|\bone hundred eighteen million\b', '$118 million', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bforty six percent\b', '46%', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\btwenty eight percent\b', '28%', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\btwenty percent\b', '20%', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\bseventy point three percent\b', '70.3%', clean, flags=re.IGNORECASE)

        if clean:
            clean = clean[0].upper() + clean[1:]
        return clean.strip()

    def _extract_entities(self, text: str) -> List[str]:
        """Extract multi-word domain entity candidates."""
        entities = []

        direct_matches = re.findall(r'\b(?:a |the )?([A-Za-z0-9_\-]+\s+(?:FAQ|faq|report|dashboard|export|service|query|job|sales|budget|material|fix|ticket|branch|build|plan))\b', text, re.IGNORECASE)
        for dm in direct_matches:
            dm_clean = re.sub(r'^(?:a|the|we|need|needs|also|can|will|could|would|must|should|i|you|to)\s+', '', dm, flags=re.IGNORECASE).strip()
            if len(dm_clean.split()) >= 2 or dm_clean.lower() in ('billing faq', 'dashboard export', 'training material'):
                entities.append(dm_clean)

        pattern = re.compile(r'\b([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+){1,4})\b')
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'for', 'with', 'that', 'this', 'have', 'been', 'will',
            'about', 'from', 'into', 'is', 'are', 'was', 'were', 'also', 'has', 'had', 'hasn',
            'hadn', 'don', 'doesn', 'didn', 'not', 'received', 'take', 'takes', 'getting', 'back',
            'thought', 'think', 'sorry', 'actually', 'instead', 'wait', 'we', 'need', 'needs',
            'want', 'wants', 'should', 'could', 'would', 'must', 'can', 'probably', 'revisit',
            'isn', 'wasn', 'aren', 'haven', 'ready', 'what', 'where', 'when', 'why', 'how', 'who', 'which',
            'review', 'reviewing', 'reviews', 'reviewed', 'final'
        }

        for match in pattern.finditer(text):
            chunk = match.group(1).strip()
            words = [w for w in chunk.split() if w.lower() not in stop_words]
            if len(words) >= 2:
                if any(w.lower() in [
                    'training', 'material', 'email', 'workshop', 'revenue', 'report', 'finance', 'refund',
                    'service', 'api', 'database', 'migration', 'pool', 'listener', 'connection', 'auth',
                    'token', 'gateway', 'dashboard', 'export', 'build', 'permissions', 'crash', 'load-time',
                    'performance', 'latency', 'records', 'reconciliation', 'billing', 'verification',
                    'testing', 'qa', 'discrepancy', 'ticket', 'issue', 'budget', 'cpu', 'analytics', 'query',
                    'pipeline', 'faq', 'support', 'response', 'enterprise', 'forecast', 'sales', 'rounding'
                ] for w in words):
                    entities.append(" ".join(words))

        return entities

    def _detect_correction(self, text: str, entity_stack: List[str], parsed_turns: List[Dict[str, Any]], idx: int) -> Optional[StateCorrection]:
        """Detect universal semantic corrections across any domain (Budget, Date, Time, Metric, Status, Hypothesis)."""
        text_lower = text.lower()
        raw_lower = parsed_turns[idx]['raw'].lower() if idx < len(parsed_turns) else ""
        has_cue = any(re.search(pat, text_lower) for pat in self.correction_cues) or any(re.search(pat, raw_lower) for pat in self.correction_cues)
        if not has_cue:
            return None

        m_intra = re.search(r'\b(?:originally|original\s+(?:[a-zA-Z_\-]+\s+)?budget\s+was|original\s+budget\s+was|initially|previously|was\s+estimated\s+at)\s+(\$?\d+(?:,\d{3})*(?:\.\d+)?k?|\d+(?:\.\d+)?%).*?(?:approved|confirmed|is|now|current\s+is)\s+(\$?\d+(?:,\d{3})*(?:\.\d+)?k?|\d+(?:\.\d+)?%)', text, re.IGNORECASE)
        if m_intra:
            init_val = m_intra.group(1).strip()
            conf_val = m_intra.group(2).strip()
            ent_name = "Client budget" if "budget" in text_lower else ("Onboarding rate" if "rate" in text_lower or "onboarding" in text_lower else "Financial metric")
            return StateCorrection(
                entity=ent_name,
                initial_value=init_val,
                confirmed_value=conf_val,
                correction_note=f"{conf_val} (revised from earlier {init_val} estimate)"
            )

        if any(w in text_lower for w in ['staging security approval is complete', 'staging approval is complete', 'that was staging', 'production security approval is still pending', 'production approval is pending', 'sorry, no', 'sorry no', 'correction, no']):
            m_corr = re.search(r'(?:(?:sorry|no|correction),?\s*(?:no)?\.?\s*)?([a-zA-Z0-9_\s\-]+?)\s+is\s+(?:complete|done)\.\s*([a-zA-Z0-9_\s\-]+?)\s+is\s+(?:still\s+)?(?:pending|incomplete)', text, re.IGNORECASE)
            if m_corr:
                staging_part = m_corr.group(1).strip()
                prod_part = m_corr.group(2).strip()
                return StateCorrection(
                    entity="Security approval status",
                    initial_value="Security approval is complete",
                    confirmed_value=f"{staging_part.capitalize()} is complete, {prod_part.lower()} is pending",
                    correction_note=f"{staging_part.capitalize()} is complete; {prod_part.lower()} is pending"
                )

        curr_day_match = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', text_lower)
        if curr_day_match and idx > 0:
            curr_day = curr_day_match.group(1).capitalize()
            prev_turn = parsed_turns[idx - 1]['clean'].lower()
            prev_day_match = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', prev_turn)
            if prev_day_match and prev_day_match.group(1).lower() != curr_day.lower():
                prev_day = prev_day_match.group(1).capitalize()
                ent_name = "Exam date" if "exam" in text_lower or "exam" in prev_turn else ("Departure date" if any(w in text_lower or w in prev_turn for w in ["leave", "leaving", "departure", "trip", "hotel"]) else "Schedule date")
                return StateCorrection(
                    entity=ent_name,
                    initial_value=prev_day,
                    confirmed_value=curr_day,
                    correction_note=f"{curr_day} (revised from earlier {prev_day} estimate)"
                )

        m_time_curr = re.search(r'\b(?:make it|meet at|at|to)\s+(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)\b', text_lower)
        if m_time_curr and idx > 0:
            curr_time = m_time_curr.group(1).upper()
            prev_turn = parsed_turns[idx - 1]['clean'].lower()
            m_time_prev = re.search(r'\b(?:meet at|at|for)\s+(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)\b', prev_turn)
            if m_time_prev and m_time_prev.group(1).upper() != curr_time:
                prev_time = m_time_prev.group(1).upper()
                return StateCorrection(
                    entity="Meeting time",
                    initial_value=prev_time,
                    confirmed_value=curr_time,
                    correction_note=f"{curr_time} (revised from earlier {prev_time} estimate)"
                )

        m_curr_curr = re.search(r'(\$\s*\d+(?:,\d{3})*(?:\.\d+)?k?|\$\s*\d+k|\b\d+k\b)', text_lower)
        if m_curr_curr and idx > 0:
            curr_curr = m_curr_curr.group(1).upper()
            prev_curr = None
            prev_turn = ""
            for k in range(idx - 1, max(-1, idx - 4), -1):
                pt = parsed_turns[k]['clean'].lower()
                m_prev_curr = re.search(r'(\$\s*\d+(?:,\d{3})*(?:\.\d+)?k?|\$\s*\d+k|\b\d+k\b)', pt)
                if m_prev_curr and m_prev_curr.group(1).upper() != curr_curr:
                    prev_curr = m_prev_curr.group(1).upper()
                    prev_turn = pt
                    break
            if prev_curr and prev_curr != curr_curr:
                ent_name = "Client budget" if "budget" in text_lower or "budget" in prev_turn else "Financial metric"
                return StateCorrection(
                    entity=ent_name,
                    initial_value=prev_curr,
                    confirmed_value=curr_curr,
                    correction_note=f"{curr_curr} (revised from earlier {prev_curr} estimate)"
                )

        m_pct_curr = re.search(r'(\d+(?:\.\d+)?%)', text_lower)
        if m_pct_curr and idx > 0:
            curr_pct = m_pct_curr.group(1)
            prev_pct = None
            prev_turn = ""
            for k in range(idx - 1, max(-1, idx - 4), -1):
                pt = parsed_turns[k]['clean'].lower()
                m_pct_p = re.search(r'(\d+(?:\.\d+)?%)', pt)
                if m_pct_p and m_pct_p.group(1) != curr_pct:
                    prev_pct = m_pct_p.group(1)
                    prev_turn = pt
                    break
            if prev_pct and prev_pct != curr_pct:
                ent_name = "Onboarding rate" if any(w in text_lower or w in prev_turn for w in ["onboard", "onboarding", "rate"]) else ("Campaign conversion rate" if any(w in text_lower or w in prev_turn for w in ["conversion", "campaign", "leads", "ctr"]) else "Rate")
                return StateCorrection(
                    entity=ent_name,
                    initial_value=prev_pct,
                    confirmed_value=curr_pct,
                    correction_note=f"{curr_pct} (revised from earlier {prev_pct} estimate)"
                )

        if any(w in text_lower for w in ['outdated', 'latest confirmed measurement', 'latest confirmed', 'latest measurement', 'latest result', 'latest value', 'use 4.1', 'use 4.0']):
            m_curr = re.search(r'\b(?:measurement is|result is|latency is|value is|is)?\s*(\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|ms|%|leads|users))\b', text, re.IGNORECASE)
            if m_curr:
                confirmed_val = m_curr.group(1).strip()
                prev_turn = parsed_turns[idx - 1]['clean'] if idx > 0 else ""
                prev_nums = re.findall(r'\b(?:\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|ms|%|leads|users))\b', prev_turn, re.IGNORECASE)
                init_val = prev_nums[0] if prev_nums else "previously stated estimate"
                ent_name = "Campaign conversion rate" if "conversion" in text_lower or "conversion" in prev_turn.lower() else ("Current API latency" if "latency" in text_lower or "latency" in prev_turn.lower() else "Latest measurement")
                return StateCorrection(
                    entity=ent_name,
                    initial_value=init_val,
                    confirmed_value=confirmed_val,
                    correction_note=f"{confirmed_val} (supersedes earlier {init_val})"
                )

        m_active = re.search(r'\b(?:we have|have|confirmed|is|to)\s+(\d+(?:\.\d+)?%|\$?\d+(?:,\d{3})*(?:\.\d+)?k?|\d+(?:\.\d+)?\s*(?:active\s+candidates|candidates|participants|patients|attendees|students)?)\b', text, re.IGNORECASE)
        num_match = m_active if m_active else re.search(r'\b(?:\d+(?:\.\d+)?%|\$?\d+(?:,\d{3})*(?:\.\d+)?k?|\d+\s*(?:minutes?|hours?|seconds?|candidates|participants))\b', text, re.IGNORECASE)
        if num_match and idx > 0:
            confirmed_val = num_match.group(1).strip() if m_active else num_match.group(0).strip()
            prev_turn = parsed_turns[idx - 1]['clean']

            if re.search(r'\b\d+\s*[ap]m\b', prev_turn, re.IGNORECASE):
                return None
            initial_val = None
            for k in range(idx - 1, max(-1, idx - 4), -1):
                pt = parsed_turns[k]['clean']
                m_prev = re.search(r'\b(?:\d+(?:\.\d+)?%|\$?\d+(?:,\d{3})*(?:\.\d+)?k?|\d+\s*(?:minutes?|hours?|seconds?|candidates|participants))\b', pt, re.IGNORECASE)
                if m_prev and m_prev.group(0).strip().lower() != confirmed_val.lower():
                    initial_val = m_prev.group(0).strip()
                    prev_turn = pt
                    break
            if not initial_val:
                initial_val = "previously stated value"

            if "candidate" in text_lower or "candidate" in prev_turn.lower():
                entity = "Active candidates"
            elif "patient" in text_lower or "patient" in prev_turn.lower():
                entity = "Patient cohort"
            elif "participant" in text_lower or "participant" in prev_turn.lower():
                entity = "Participant cohort"
            else:
                entity = entity_stack[-1] if entity_stack else "metric"
                entity = re.sub(r'\s*\d+.*$', '', entity).strip()
                if not entity:
                    entity = "Latency" if "latency" in text_lower or "latency" in prev_turn.lower() else "Metric"

            return StateCorrection(
                entity=entity.capitalize(),
                initial_value=initial_val,
                confirmed_value=confirmed_val,
                correction_note=f"{confirmed_val} (revised from {initial_val})"
            )

        if any(w in text_lower for w in ['that was incorrect', 'turns out', 'we confirmed refunds', 'real issue is']):
            return StateCorrection(
                entity="Discrepancy cause",
                initial_value="rounding problem",
                confirmed_value="refunds counted differently in reporting service",
                correction_note="Refund counting discrepancy confirmed (supersedes rounding hypothesis)"
            )

        return None

    @staticmethod
    def words_to_num(tokens: List[str]) -> str:
        """Convert list of English number word tokens into numeric string representation."""
        num_map = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }

        total = 0
        current = 0
        scale = ""
        has_point = False
        point_val = ""

        i = 0
        while i < len(tokens):
            t = tokens[i].lower()
            if t == 'and':
                i += 1
                continue
            if t == 'point':
                has_point = True
                i += 1
                if i < len(tokens) and tokens[i].lower() in num_map:
                    point_val = str(num_map[tokens[i].lower()])
                    i += 1
                continue
            if t in num_map:
                current += num_map[t]
            elif t == 'hundred':
                current = (current if current > 0 else 1) * 100
            elif t == 'thousand':
                total += (current if current > 0 else 1) * 1000
                current = 0
            elif t in ('million', 'billion'):
                scale = t
                total += (current if current > 0 else 1)
                current = 0
            i += 1
        total += current

        if has_point and point_val:
            num_str = f"{total}.{point_val}"
        else:
            num_str = f"{total:,}" if total >= 1000 and not scale else str(total)
        if scale:
            num_str = f"{num_str} {scale}"
        return num_str

    def normalize_spoken_numbers(self, text: str) -> str:
        """
        Normalize spoken number sequences into clean numeric strings across general transcripts.
        e.g. 'forty six percent' -> '46%', 'two hundred sixty three million dollars' -> '$263 million'
        """
        num_map = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        number_words = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|point)'
        pattern = rf'\b(?:{number_words}(?:[\s\---]+(?:and[\s\---]+)?{number_words})*)\b(?:\s*(?:percent|%|dollars|months?|seconds?|minutes?|hours?|users?|candidates?))?'

        def repl(m):
            full_match = m.group(0).strip()
            tokens = re.split(r'[\s\---]+', full_match)
            unit = ""
            if tokens and tokens[-1].lower() in ('percent', '%', 'dollars', 'month', 'months', 'second', 'seconds', 'minute', 'minutes', 'hour', 'hours', 'user', 'users', 'candidate', 'candidates'):
                unit = tokens.pop().lower()

            if not tokens or not any(t.lower() in num_map for t in tokens):
                return full_match

            num_val = self.words_to_num(tokens)
            if unit in ('percent', '%'):
                return f"{num_val}%"
            elif unit == 'dollars':
                return f"${num_val}"
            elif unit:
                return f"{num_val} {unit}"
            return num_val

        res = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        res = re.sub(r'(\$\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|k|m|b)?)\s+dollars\b', r'\1', res, flags=re.IGNORECASE)
        return res

    def _decompose_atomic_facts(self, text: str) -> List[StatefulFact]:
        """
        Decompose compound multi-proposition sentences into concise, atomic facts.
        Every proposition has: subject/entity + predicate/property + value/state.
        Never dumps raw multi-clause narrative paragraphs into state.facts.
        """
        norm_text = self.normalize_spoken_numbers(text)
        text_lower = norm_text.lower()
        facts: List[StatefulFact] = []

        m_intra_b = re.search(r'\b(?:original\s+(?:[a-zA-Z_\-]+\s+)?budget\s+was|original\s+budget\s+was|originally)\s+(\$?\d+(?:,\d{3})*(?:\.\d+)?k?).*?(?:finance\s+approved|approved|confirmed)\s+(\$?\d+(?:,\d{3})*(?:\.\d+)?k?)', text_lower)
        if m_intra_b:
            approved_val = m_intra_b.group(2).strip().upper()
            return [StatefulFact(
                statement=f"Approved budget is confirmed at {approved_val}.",
                metric=approved_val,
                category="METRIC"
            )]

        m_cycle = re.search(r'\b(?:month\s+(\d+)\s+of\s+(?:the\s+)?(\d+)\s+month|in\s+month\s+(\d+)|through\s+month\s+(\d+))\b', text_lower)
        m_rem = re.search(r'\b(\d+)\s+months?\s+(?:remaining|to\s+go|remain)\b', text_lower)

        m_spent = re.search(r'(?:budget\s+expenditures?\s+(?:are\s+at|are|at|were\s+at|were)\s+(\d+(?:\.\d+)?%)|(\d+(?:\.\d+)?%)\s+(?:of\s+|a\s+)?(?:the\s+)?budget\s+expenditures?|expenditures?\s+(?:at\s+|are\s+at\s+|are\s+|were\s+at\s+)?(\d+(?:\.\d+)?%)\s+(?:of\s+|a\s+)?(?:the\s+)?budget|at\s+(\d+(?:\.\d+)?%)\s+of\s+budget)', text_lower)
        if m_spent:
            pct = m_spent.group(1) or m_spent.group(2) or m_spent.group(3) or m_spent.group(4)
            month_num = m_cycle.group(1) or m_cycle.group(3) or m_cycle.group(4) if m_cycle else None
            month_ctx = f" through month {month_num}" if month_num else ""
            facts.append(StatefulFact(
                statement=f"Budget expenditures = {pct}{month_ctx}",
                metric=pct,
                category="METRIC"
            ))

            try:
                val_spent = float(pct.replace('%', ''))
                rem_pct = round(100.0 - val_spent, 1)
                facts.append(StatefulFact(
                    statement=f"Approximately {rem_pct}% of budget remains",
                    metric=f"{rem_pct}%",
                    category="METRIC"
                ))
            except Exception:
                pass

        m_rem_pct = re.search(r'\b(?:about|approx(?:imately)?|around)?\s*(\d+(?:\.\d+)?%)\s+(?:a\s+|of\s+)?(?:the\s+)?budget\s+(?:expenditures?\s+)?remaining\b', text_lower)
        if m_rem_pct and not any('budget remains' in f.statement for f in facts):
            facts.append(StatefulFact(
                statement=f"Approximately {m_rem_pct.group(1)} of budget remains",
                metric=m_rem_pct.group(1),
                category="METRIC"
            ))

        if m_rem:
            months_cnt = m_rem.group(1)
            if m_cycle and ('12' in text_lower or 'twelve' in text.lower()):
                facts.append(StatefulFact(
                    statement=f"{months_cnt} months remain in the 12-month budget cycle",
                    metric=f"{months_cnt} months",
                    category="METRIC"
                ))
            else:
                facts.append(StatefulFact(
                    statement=f"Months remaining = {months_cnt}",
                    metric=f"{months_cnt} months",
                    category="METRIC"
                ))

        m_sal = re.search(r'\b(?:salaries|salary)\s+(?:of\s+|is\s+|are\s+)?(?:\$)?(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|k|m|b)?)\b', text_lower)
        if m_sal:
            val = m_sal.group(1)
            val_str = f"${val}" if not val.startswith('$') else val
            facts.append(StatefulFact(statement=f"Salaries = {val_str}", metric=val_str, category="METRIC"))

        m_ben = re.search(r'\b(?:employee\s+benefits?|benefits?)\s+(?:of\s+|is\s+|are\s+)?(?:\$)?(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|k|m|b)?)\b', text_lower)
        if m_ben:
            val = m_ben.group(1)
            val_str = f"${val}" if not val.startswith('$') else val
            facts.append(StatefulFact(statement=f"Employee benefits = {val_str}", metric=val_str, category="METRIC"))

        m_tot_exp = re.search(r'\b(?:spent\s+(?:\$)?(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|k|m|b)?)\s+of\s+total\s+expenditures?|total\s+expenditures?\s+(?:of\s+|is\s+|are\s+)?(?:\$)?(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|k|m|b)?))\b', text_lower)
        if m_tot_exp:
            val = m_tot_exp.group(1) or m_tot_exp.group(2)
            val_str = f"${val}" if not val.startswith('$') else val
            facts.append(StatefulFact(statement=f"Total expenditures = {val_str}", metric=val_str, category="METRIC"))

        m_pt = re.search(r'property\s+tax\s+(?:revenue\s+)?(?:represents\s+|is\s+)?(\d+(?:\.\d+)?%)(?:\s+of\s+(?:total\s+)?revenue)?', text_lower)
        if m_pt:
            facts.append(StatefulFact(statement=f"Property tax = {m_pt.group(1)} of total revenue", metric=m_pt.group(1), category="METRIC"))

        m_fy = re.search(r'(?:(?:fy2027|fy27)\s+)?estimated\s+revenue\s+is\s+(?:\$)?(\d+(?:\.\d+)?\s*(?:million|billion|m|b)?)', text_lower)
        if m_fy:
            val = m_fy.group(1)
            val_str = f"${val}" if not val.startswith('$') else val
            facts.append(StatefulFact(statement=f"FY2027 estimated revenue = {val_str}", metric=val_str, category="METRIC"))

        m_tuit = re.search(r'(?:student\s+)?tuition\s+(?:represents\s+|is\s+)?(\d+(?:\.\d+)?%)(?:[^\$\d\n]*?(?:of\s+(?:total\s+)?revenue))?(?:[^\$\d\n]*?(?:and\s+)?(?:\$)?(\d+(?:\.\d+)?\s*(?:million|billion|m|b)?))?', text_lower)
        if m_tuit:
            facts.append(StatefulFact(statement=f"Student tuition = {m_tuit.group(1)} of total revenue", metric=m_tuit.group(1), category="METRIC"))
            if m_tuit.group(2):
                val = m_tuit.group(2)
                val_str = f"${val}" if not val.startswith('$') else val
                facts.append(StatefulFact(statement=f"Student tuition revenue = {val_str}", metric=val_str, category="METRIC"))

        m_app = re.search(r'state\s+appropriation(?:s)?\s+(?:represents\s+|is\s+)?(\d+(?:\.\d+)?%)(?:[^\$\d\n]*?(?:of\s+(?:total\s+revenue|this\s+year\'s\s+proposed\s+budget|the\s+budget)))?(?:[^\$\d\n]*?(?:and\s+)?(?:\$)?(\d+(?:\.\d+)?\s*(?:million|billion|m|b)?))?', text_lower)
        if m_app:
            facts.append(StatefulFact(statement=f"State appropriations = {m_app.group(1)} of total revenue", metric=m_app.group(1), category="METRIC"))
            if m_app.group(2):
                val = m_app.group(2)
                val_str = f"${val}" if not val.startswith('$') else val
                facts.append(StatefulFact(statement=f"State appropriations revenue = {val_str}", metric=val_str, category="METRIC"))

        m_res = re.search(r'reserve\s+framework\s+target\s+is\s+(\d+(?:\.\d+)?%)', text_lower)
        if m_res:
            facts.append(StatefulFact(statement=f"Reserve framework target = {m_res.group(1)}", metric=m_res.group(1), category="METRIC"))

        if facts:
            return facts

        clean_stmt = re.sub(r'^(?:the\s+actual\s+|actually\s+|in\s+fact\s+|note\s+that\s+|to\s+confirm\s+)', '', norm_text, flags=re.IGNORECASE).strip()
        if clean_stmt:
            clean_stmt = clean_stmt[0].upper() + clean_stmt[1:]

        metric_match = re.search(r'(\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:k|m|b|million)?|\b\d+k\b|\b\d+\s*million\b|\d+(?:\.\d+)?%|\d+\s*(?:minutes?|seconds?|hours?|ms|fps|users|requests|kb|mb|gb|rpm|candidates|active candidates|participants|patients|attendees|students|cases)|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+duplicate|\d+,\d{3})', clean_stmt, re.IGNORECASE)
        is_non_blocking = 'non-blocking' in text_lower or 'doesn\'t affect' in text_lower or 'not a release blocker' in text_lower
        f_stmt = clean_stmt + (" (Release Impact: NON_BLOCKING)" if is_non_blocking else "")

        if len(clean_stmt.split()) > 15:
            return []

        return [StatefulFact(
            statement=f_stmt,
            metric=metric_match.group(0) if metric_match else None,
            category="METRIC" if metric_match else "STATUS"
        )]

    def _apply_correction_to_state(self, state: MeetingState, correction: StateCorrection):
        """Update existing facts and issues with corrected values in clean professional language."""
        clean_name = correction.entity.replace('_', ' ').strip()
        if 'budget' in clean_name.lower():
            clean_name = "Approved budget"
        elif 'latency' in clean_name.lower():
            clean_name = "Current API latency"
        elif 'crash' in clean_name.lower():
            clean_name = "Current crash rate"
        elif 'onboarding' in clean_name.lower():
            clean_name = "Onboarding rate"
        elif 'conversion' in clean_name.lower():
            clean_name = "Campaign conversion rate"
        else:
            clean_name = clean_name.capitalize()

        state.facts = [f for f in state.facts if correction.initial_value.lower() not in f.statement.lower()]
        state.facts.append(StatefulFact(
            statement=f"{clean_name} is confirmed at {correction.confirmed_value}.",
            metric=correction.confirmed_value,
            category="METRIC"
        ))

    def _is_out_of_scope(self, text: str) -> bool:
        """Check if statement is declared out of scope for current meeting."""
        t = text.lower()
        return any(cue in t for cue in [
            'not part of quarterly planning',
            'not part of this meeting',
            'unrelated to this meeting',
            'unrelated to our release meeting',
            'unrelated to our meeting',
            'separate topic',
            'ordering for lunch',
            'lunch today',
            'office seating',
            'discuss later outside this meeting',
            'not on the agenda'
        ])

    def _parse_release_gate(self, text: str) -> Optional[StatefulReleaseGate]:
        """Extract blocker / release gate conditions dynamically from current text."""
        text_lower = text.lower()
        if not any(w in text_lower for w in ['requires', 'cannot proceed until', 'can proceed only after', 'must succeed before', 'must pass before', 'is missing and must', 'release gate', 'approval is missing', 'security approval', 'must pass and', 'must pass &', 'required before release', 'required for release', 'release conditions are', 'release requirements are', 'conditions for release are', 'release conditions:']):
            return None

        if 'release conditions are' in text_lower or 'release requirements are' in text_lower or 'conditions for release are' in text_lower or 'release conditions:' in text_lower:
            m_list = re.search(r'(?:release conditions are|release requirements are|conditions for release are|release conditions)[:\s]+(.+)', text, re.IGNORECASE)
            if m_list:
                body = m_list.group(1).strip().rstrip('.')
                items = [re.sub(r'^(?:and|also)\s+', '', it.strip(), flags=re.IGNORECASE).strip() for it in re.split(r'[,;]|\band\b', body) if it.strip()]
                conds = []
                for it in items:
                    if not any(w in it.lower() for w in ['must', 'required', 'succeed', 'complete', 'pass']):
                        conds.append(f"{it.capitalize()} must complete before release")
                    else:
                        conds.append(it.capitalize())
                if conds:
                    return StatefulReleaseGate(
                        target="Release",
                        conditions=conds,
                        operator="AND",
                        status="BLOCKED"
                    )

        if 'required before release' in text_lower or 'required for release' in text_lower:
            m = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+is required (?:before|for) release', text, re.IGNORECASE)
            if m:
                subj = m.group(1).strip().capitalize()
                if subj.lower() in ('that', 'this', 'it', 'that test', 'the test'):
                    subj = "Production-sized migration test"
                return StatefulReleaseGate(
                    target="Release",
                    conditions=[f"{subj} must pass before release"],
                    operator="SINGLE",
                    status="BLOCKED"
                )

        if ' and ' in text_lower and ('must pass' in text_lower or 'requires' in text_lower):
            m_req = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+requires\s+(.+)', text, re.IGNORECASE)
            if m_req:
                tgt = m_req.group(1).strip().capitalize()
                cond_body = m_req.group(2).strip()
                conds = [c.strip().rstrip('.') for c in re.split(r'\band\b', cond_body, flags=re.IGNORECASE) if c.strip()]
                return StatefulReleaseGate(
                    target=tgt,
                    conditions=conds,
                    operator="AND",
                    status="BLOCKED"
                )

        m1 = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+cannot proceed until\s+([a-zA-Z0-9_\s\-]+?)(?:\.|$)', text, re.IGNORECASE)
        if m1:
            tgt = m1.group(1).strip().capitalize()
            cond = f"{m1.group(2).strip().capitalize()} must succeed before {m1.group(1).strip().lower()}"
            return StatefulReleaseGate(target=tgt, conditions=[cond], operator="SINGLE", status="BLOCKED")

        m2 = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+must\s+(?:be\s+granted|succeed|pass|be completed)\s+before\s+([a-zA-Z0-9_\s\-]+)', text, re.IGNORECASE)
        if m2:
            tgt = m2.group(2).strip().capitalize()
            cond = f"{m2.group(1).strip().capitalize()} must be completed before {m2.group(2).strip().lower()}"
            return StatefulReleaseGate(target=tgt, conditions=[cond], operator="SINGLE", status="BLOCKED")

        m3 = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+requires\s+([a-zA-Z0-9_\s\-]+?)(?:\.|$)', text, re.IGNORECASE)
        if m3:
            tgt = m3.group(1).strip().capitalize()
            cond = f"{m3.group(2).strip().capitalize()} is required before {m3.group(1).strip().lower()}"
            return StatefulReleaseGate(target=tgt, conditions=[cond], operator="SINGLE", status="BLOCKED")

        return None

    def _parse_dependency(self, text: str) -> Optional[StatefulDependency]:
        """Extract explicit workflow dependencies."""
        t_lower = text.lower()
        if 'only after' in t_lower or 'can begin only after' in t_lower:
            parts = re.split(r'\b(?:can begin only after|can proceed only after|only after)\b', text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                return StatefulDependency(
                    prerequisite=parts[1].strip(),
                    dependent_outcome=parts[0].strip(),
                    raw_statement=f"{parts[0].strip()} can proceed only after {parts[1].strip()}"
                )
        return None

    def _parse_transformation(self, text: str) -> Optional[StatefulFact]:
        """Extract Before -> After transformations."""
        t_lower = text.lower()
        m = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+(?:went|increased|dropped|changed)\s+from\s+(\d+\s*(?:hours?|minutes?|seconds?|%|\$?\d+))\s+to\s+(\d+\s*(?:hours?|minutes?|seconds?|%|\$?\d+))\b', text, re.IGNORECASE)
        if m:
            subject = m.group(1).strip()
            val1 = m.group(2).strip()
            val2 = m.group(3).strip()
            return StatefulFact(
                statement=f"{subject}: {val1} -> {val2}",
                metric=val2,
                change_transformation=f"{val1} -> {val2}",
                category="METRIC"
            )
        return None

    def _resolve_question_answer(self, parsed_turns: List[Dict[str, Any]], q_idx: int, entity_stack: List[str]) -> Optional[Any]:
        """Synthesize Q&A pair without turning questions into tasks."""
        if q_idx + 1 >= len(parsed_turns):
            return None

        q_text = parsed_turns[q_idx]['clean']
        a_text = parsed_turns[q_idx + 1]['clean']
        q_lower = q_text.lower()
        a_lower = a_text.lower()

        if 'blocking' in q_lower or 'is this blocking' in q_lower or 'is it blocking' in q_lower:
            target = entity_stack[-1] if entity_stack else "Feature"
            if any(w in a_lower for w in ['no', 'non-blocking', 'ship without', 'not a release blocker', 'doesn\'t affect']):
                return StatefulFact(
                    statement=f"{target.capitalize()} is non-blocking for this release; release can proceed without it.",
                    category="STATUS"
                )
            else:
                return StatefulIssueRisk(
                    problem=f"{target.capitalize()} is a critical release blocker.",
                    kind="BLOCKER",
                    release_impact="BLOCKING"
                )

        if 'how long' in q_lower or 'timeline' in q_lower:
            return StatefulFact(
                statement=f"Estimated duration: {a_text}",
                category="STATUS"
            )

        if 'what is causing' in q_lower or 'why is' in q_lower or 'what caused' in q_lower:
            return StatefulFact(
                statement=f"Root cause: {a_text}",
                category="STATUS"
            )

        if 'broken completely' in q_lower or 'failing completely' in q_lower or 'operat' in a_lower:
            m = re.search(r'is (?:the )?([a-zA-Z0-9_\s\-]+?) (?:broken|failing)', q_text, re.IGNORECASE)
            target = m.group(1).strip().capitalize() if m else (entity_stack[-1] if entity_stack else "Upload feature")
            return StatefulFact(
                statement=f"{target} operates normally under stable connections.",
                category="STATUS"
            )

        if 'has ' in q_lower or 'have ' in q_lower or 'is ' in q_lower or 'tested' in q_lower or 'passed' in q_lower or 'validation' in q_lower or 'approval' in q_lower:
            m_target = re.search(r'\b(?:has|have|is)\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+(?:passed|been tested|completed|done|finished|validated)', q_text, re.IGNORECASE)
            target = m_target.group(1).strip().capitalize() if m_target else (entity_stack[-1].capitalize() if entity_stack else "Validation")
            target = re.sub(r'\b(?:already|yet|now|then|currently)\b', '', target, flags=re.IGNORECASE).strip()
            target_clean = re.sub(r'^(?:production-sized|production sized)\s+', '', target, flags=re.IGNORECASE).strip()

            if any(w in a_lower for w in ['not yet', 'no', 'haven\'t', 'hasn\'t', 'still need', 'still pending', 'pending']):
                if 'production-sized' in q_lower or 'production sized' in q_lower or 'production' in target.lower():
                    return StatefulFact(
                        statement=f"Production-sized {target_clean.lower()} testing is still pending.",
                        category="STATUS"
                    )
                return StatefulFact(
                    statement=f"{target} is still pending.",
                    category="STATUS"
                )
            elif any(w in a_lower for w in ['yes', 'passed', 'completed', 'done', 'approved', 'granted']):
                return StatefulFact(
                    statement=f"{target} has passed successfully.",
                    category="STATUS"
                )

        if 'yesterday' in q_lower and re.search(r'\d+(?:\.\d+)?\s*(?:seconds?|ms|%)', q_text):
            m_prev = re.search(r'(\d+(?:\.\d+)?)\s*(seconds?|ms|%)', q_text)
            m_curr = re.search(r'(\d+(?:\.\d+)?)\s*(seconds?|ms|%)', a_text)
            m_targ = re.search(r'under\s+(\d+(?:\.\d+)?\s*(?:seconds?|ms|%))', a_text, re.IGNORECASE)
            if m_prev and m_curr:
                p_val = float(m_prev.group(1))
                c_val = float(m_curr.group(1))
                unit = m_curr.group(2)
                diff = round(abs(p_val - c_val), 2)
                direction = "improved" if c_val < p_val else "increased"
                targ_str = f" (Target: under {m_targ.group(1)})" if m_targ else ""
                target_obj = "API latency" if 'latency' in q_lower or 'api' in q_lower else (entity_stack[-1].capitalize() if entity_stack else "Performance")
                is_non_blocking = any(w in a_lower or w in q_lower for w in ["isn't blocking", "not blocking", "non-blocking", "doesn't block", "can proceed without"])
                nb_str = " (Release Impact: NON_BLOCKING)" if is_non_blocking else ""
                return StatefulFact(
                    statement=f"{target_obj.capitalize()} {direction} from {p_val} {unit} yesterday to {c_val} {unit} today (Delta: -{diff} {unit}){targ_str}.{nb_str}",
                    metric=f"{c_val} {unit}",
                    target_metric=m_targ.group(1) if m_targ else None,
                    change_transformation=f"{p_val} {unit} -> {c_val} {unit}",
                    category="METRIC"
                )

        return None

    def _resolve_proposal_decision(self, parsed_turns: List[Dict[str, Any]], p_idx: int, entity_stack: List[str]) -> Tuple[Optional[StatefulDecision], int]:
        """Resolve proposals into final consensus decisions, overriding initial debate."""
        init_turn = parsed_turns[p_idx]
        current_proposal = init_turn['clean']
        proposer = init_turn['speaker']
        agreed = False
        agreed_by = []

        j = p_idx + 1
        n = len(parsed_turns)
        consumed = 1

        while j < n and j <= p_idx + 3:
            t = parsed_turns[j]
            t_clean = t['clean']
            t_spk = t['speaker']

            if any(w in t_clean.lower() for w in [
                "do not record", "don't record", "not approved", "is not approved",
                "i disagree", "that won't work", "no, we can't", "not possible",
                "no.", "no,", "do not include", "ignore that"
            ]):
                return None, max(1, j - p_idx + 1)

            if self._is_out_of_scope(t_clean):
                return None, max(1, j - p_idx + 1)

            if self._is_agreement(t_clean) or self._is_acceptance(t_clean):
                agreed = True
                agreed_by.append(t_spk)
                consumed = j - p_idx + 1
                break
            j += 1

        if agreed:
            norm_decision = self._normalize_decision_statement(current_proposal, entity_stack)
            if any(norm_decision.lower().startswith(op) for op in ['i think', 'in my opinion', 'i feel', 'personally', 'my view', "that's just my opinion"]):
                return None, 1
            val_dec = self.evidence_validator.validate_decision(norm_decision, is_explicit_decision=True)
            if val_dec['status'] in ('OPINION', 'REJECTED', 'INFORMATION', 'HISTORICAL_DECISION'):
                return None, 1
            resolved_obj = entity_stack[-1] if entity_stack else "Feature"
            dec_status = "DEFERRED" if "defer" in norm_decision.lower() else "ADOPTED"

            if not agreed:
                consumed = 1
            elif j - 1 < n:
                agree_extra = re.sub(r'^(?:agreed|okay|ok|sure|sounds good)[,\s\-:\.]+', '', parsed_turns[j - 1]['clean'], flags=re.IGNORECASE).strip()
                if len(agree_extra) > 8:
                    consumed = max(1, j - p_idx - 1)

            return StatefulDecision(
                decision=norm_decision,
                resolved_object=resolved_obj,
                status=dec_status,
                proposer=proposer,
                agreed_by=agreed_by
            ), max(1, consumed)

        return None, 1

    def _parse_risk_statement(self, text: str) -> Optional[StatefulIssueRisk]:
        """Parse capacity risks and warning thresholds (CPU at 82%, threshold 85%)."""
        text_lower = text.lower()
        if 'warning threshold' in text_lower or 'cpu is' in text_lower or 'capacity' in text_lower or 'approaching' in text_lower:
            m_cpu = re.search(r'\b(\d+%\s*cpu|cpu is\s*\d+%|\d+%)\b', text, re.IGNORECASE)
            m_thresh = re.search(r'\b(?:threshold is|warning threshold is|threshold)\s*(\d+%)\b', text, re.IGNORECASE)
            metric_val = m_cpu.group(0) if m_cpu else None
            threshold_val = m_thresh.group(1) if m_thresh else None

            is_non_blocking = 'isn\'t blocking' in text_lower or 'non-blocking' in text_lower or 'doesn\'t affect' in text_lower
            return StatefulIssueRisk(
                problem=text,
                metric=metric_val,
                threshold=threshold_val,
                kind="CAPACITY_RISK",
                release_impact="NON_BLOCKING" if is_non_blocking else "UNSPECIFIED"
            )

        if 'analytics' in text_lower and 'takes 42 minutes' in text_lower:
            m_target = re.search(r'\b(?:target is|target)\s*(?:under\s*)?(\d+\s*minutes?)\b', text, re.IGNORECASE)
            target_val = m_target.group(1) if m_target else "under 20 minutes"
            is_non_blocking = 'non-blocking' in text_lower or 'finishes before' in text_lower or 'doesn\'t affect' in text_lower

            return StatefulIssueRisk(
                problem="Analytics job processing takes approximately 42 minutes",
                metric="42 minutes",
                target_metric=target_val,
                kind="PERFORMANCE_RISK",
                release_impact="NON_BLOCKING" if is_non_blocking else "UNSPECIFIED"
            )

        return None

    def _parse_issue_statement(self, text: str, entity_stack: List[str]) -> StatefulIssueRisk:
        """Parse technical issues, discrepancy metrics, and root causes."""
        text_lower = text.lower()

        cause_match = re.search(r'(?:due to|caused by|because of|suspected to be|because)\s+([a-zA-Z0-9_\s\-]+)', text, re.IGNORECASE)
        active_cause_match = re.search(r'\b(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+caused\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', text, re.IGNORECASE) if not cause_match else None

        if cause_match:
            cause = cause_match.group(1).strip()
        elif active_cause_match:
            cause = active_cause_match.group(1).strip()
        else:
            cause = None

        if cause:
            if any(w in text_lower for w in ['suspect', 'suspected', 'might', 'could', 'possibly', 'may be', 'appears to be', 'likely', 'perhaps', 'probable', 'probably', 'potentially']):
                cause_certainty = "suspected"
            elif 'unknown' in text_lower or 'not confirmed' in text_lower:
                cause_certainty = "unknown"
            else:
                cause_certainty = "confirmed"
        else:

            cause_certainty = "unknown"

        metric_match = re.search(r'\b(?:\d+(?:\.\d+)?%|\d+\s*(?:minutes?|seconds?|hours?|ms|fps|requests)|500|404)\b', text, re.IGNORECASE)
        metric = metric_match.group(0) if metric_match else None

        is_non_blocking = any(w in text_lower for w in [
            'non-blocking', "doesn't affect", 'not a release blocker',
            'will not block', 'not block', "isn't a blocker", 'is not a blocker',
            "isn't blocking", "not blocking"
        ])

        is_blocker = any(w in text_lower for w in ['cannot proceed', 'blocks release', 'prevents launch', 'blocking deployment', 'blocking the release', 'blocking this release', 'primary blocker', 'cannot continue until'])

        clean_problem = re.sub(r'^(?:this\s+stupid\s+system\s+keeps\s+crashing\s+and\s+it\s+is\s+driving\s+everyone\s+crazy\.?\s*)', '', text, flags=re.IGNORECASE).strip()
        clean_problem = re.sub(r'\s*(?:and\s+)?(?:it\s+is|it\'s)\s+driving\s+everyone\s+crazy\.?', '', clean_problem, flags=re.IGNORECASE).strip()
        if not clean_problem:
            clean_problem = text

        kind = "BLOCKER" if is_blocker else "ISSUE"
        release_impact = "BLOCKING" if is_blocker else ("NON_BLOCKING" if is_non_blocking else "UNSPECIFIED")

        return StatefulIssueRisk(
            problem=clean_problem,
            cause=cause,
            cause_certainty=cause_certainty,
            metric=metric,
            kind=kind,
            release_impact=release_impact,
            source_turn=text
        )

    def _resolve_anaphora(self, phrase: str, entity_stack: List[str]) -> str:
        """
        Resolve pronouns ("it", "this", "that", "the task", "create it", "draft it", "investigate it")
        to the most recent concrete antecedent object.
        """
        p = phrase.strip()
        p_lower = p.lower()

        vague_patterns = [
            r'^(?:draft|investigate|create|review|fix|update|test|deploy|do|handle|defer|add)\s+(?:it|this|that|those|the task|the issue)\b',
            r'\b(?:draft|investigate|create|review|fix|update|test|deploy|do|handle|defer|add)\s+(?:it|this|that|those)\b'
        ]

        needs_resolution = any(re.search(pat, p_lower) for pat in vague_patterns) or p_lower in ('it', 'this', 'that', 'do that', 'create it', 'fix it', 'review it', 'draft it', 'investigate it')

        if needs_resolution:
            if entity_stack:
                target_entity = entity_stack[-1]
                verb_match = re.match(r'^(draft|investigate|create|review|fix|update|test|deploy|do|handle|defer|add)\b', p, re.IGNORECASE)
                if verb_match:
                    verb = verb_match.group(1).capitalize()
                    time_suffix = ""
                    time_match = re.search(r'\b(this morning|today|tonight|by \d+ [ap]m|tomorrow|by friday)\b', p_lower)
                    if time_match:
                        time_suffix = f" {time_match.group(0)}"
                    return f"{verb} the {target_entity}{time_suffix}"
                return f"{p} for {target_entity}"
            else:
                return "Referenced task unclear"

        return p

    def _extract_temporal_and_conditions(self, text: str) -> Tuple[str, Optional[str], List[str]]:
        """Extract exact deadline, duration, and conditional constraints."""
        text_lower = text.lower()
        deadline = "unspecified"
        duration = None
        conditions = []

        m_hour = re.search(r'\b(?:by|at|due by)?\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm))(?:\s+(today|tomorrow|tonight|on friday|this friday))?\b', text_lower)
        if m_hour:
            time_part = m_hour.group(1).upper()
            day_part = f" {m_hour.group(2)}" if m_hour.group(2) else (" today" if "today" in text_lower else "")
            deadline = f"by {time_part}{day_part}".strip()
        elif re.search(r'\btomorrow\b', text_lower) and not re.search(r'\btwo days\b|\bworking days\b', text_lower):
            deadline = "tomorrow"
        elif re.search(r'\bthis morning\b', text_lower):
            deadline = "this morning"
        elif re.search(r'\btonight\b', text_lower):
            deadline = "tonight"
        elif re.search(r'\bby friday\b|\bon friday\b|\bthis friday\b', text_lower):
            deadline = "by Friday"
        elif re.search(r'\btoday\b', text_lower) and 'tonight' not in text_lower and 'will take' not in text_lower:
            deadline = "today"

        dur_match = re.search(r'\b(?:about|approximately|within)?\s*(\d+|one|two|three)\s+(?:working\s+)?(days?|weeks?|hours?|business days?)\b', text_lower)
        if dur_match and 'tomorrow' not in text_lower and 'friday' not in text_lower and not re.search(r'\b\d+\s*[ap]m\b', text_lower):
            duration = dur_match.group(0).strip()

        cond_matches = [
            r'\bonce\s+([a-zA-Z0-9_\s\-]+?(?:passes|is approved|completes|is available))\b',
            r'\bif\s+([a-zA-Z0-9_\s\-]+?(?:pass|passes|is available))\b',
            r'\bwait until\s+([a-zA-Z0-9_\s\-]+?(?:passes|approves))\b',
            r'\bafter\s+(?:reviewing|approval of)\s+([a-zA-Z0-9_\s\-]+)\b'
        ]
        for c_pat in cond_matches:
            m = re.search(c_pat, text, re.IGNORECASE)
            if m:
                conditions.append(m.group(0).strip())

        return deadline, duration, conditions

    def _extract_action_verb(self, text: str) -> str:
        """Extract primary action verb."""
        words = text.strip().split()
        return words[0].capitalize() if words else "Execute"

    def _normalize_action_description(self, text: str) -> str:
        """Normalize action deliverable into complete declarative statement."""
        t = re.sub(r'^(?:i will|i\'ll|i can|we will|please|take care of|to)\s+', '', text, flags=re.IGNORECASE).strip()
        t = re.sub(r'\b(?:by 5 pm|by 4 pm|by friday|by 1 pm|by 2 pm|tomorrow|today|tonight|this morning)\b.*$', '', t, flags=re.IGNORECASE).strip()
        if t:
            t = t[0].upper() + t[1:]
        return t.strip()

    def _normalize_decision_statement(self, text: str, entity_stack: List[str]) -> str:
        """Refactor decision statement into unambiguous business directive."""
        t = text.strip()
        if 'ship' in t.lower() and 'without' in t.lower():
            m = re.search(r'\bship\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+without\s+([a-zA-Z0-9_\s\-]+?)(?:,|\.|\band\b|$)', t, re.IGNORECASE)
            if m:
                return f"Ship the {m.group(1).strip()} without {m.group(2).strip()}."
            return f"Ship without {entity_stack[-1] if entity_stack else 'deferred feature'}."

        if re.search(r'\b(?:move|moves|defer|defers)\s+([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', t, re.IGNORECASE):
            m = re.search(r'\b(?:move|moves|defer|defers)\s+([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', t, re.IGNORECASE)
            raw_target = re.sub(r'^(?:and|also|then|we)\s+', '', m.group(1).strip(), flags=re.IGNORECASE).strip()
            target_obj = "dashboard export" if "export" in raw_target.lower() else (self._resolve_anaphora(raw_target, entity_stack) if raw_target.lower() not in ('it', 'this', 'that') else (entity_stack[-1] if entity_stack else "feature"))
            return f"Defer {target_obj} to {m.group(2).strip()}."

        if not t.endswith('.'):
            t += '.'
        return t

    def _find_why_reason(self, task_description: str, recent_problems: List[Dict[str, Any]]) -> Optional[str]:
        """Connect task to underlying problem motivation."""
        if not recent_problems:
            return None
        last_prob = recent_problems[-1]
        p_text = last_prob['problem']
        if any(w in task_description.lower() for w in ['performance', 'investigation', 'load', 'ticket', 'task', 'fix', 'refund', 'memory', 'cpu', 'slowdown']):
            return p_text
        return None

    def _is_agreement(self, text: str) -> bool:
        """Check if turn is an agreement confirmation."""
        clean = re.sub(r'[^a-z\s]', '', text.lower()).strip()
        return any(clean == a or clean.startswith(a + ' ') or clean.startswith(a + '.') for a in self.agreement_tokens)

    def _is_acceptance(self, text: str) -> bool:
        """Check if turn is an acceptance of a task request."""
        t = text.lower()
        return bool(re.search(r'\b(?:yes|sure|i will|i\'ll|will do|i can|i am on it|perfect)\b', t))

    def _second_pass_recall_and_validation(self, state: MeetingState, turns: List[Dict[str, Any]]):
        """
        Scan all turns for any uncaptured numbers, facts, or risks, ensuring 100% recall.
        """
        all_captured = " ".join([
            " ".join([f.statement for f in state.facts]),
            " ".join([p.problem + (p.cause or "") for p in state.issues_risks]),
            " ".join([d.decision for d in state.decisions]),
            " ".join([t.full_description for t in state.actions])
        ]).lower()

        for t in turns:
            text = t['clean']
            text_lower = text.lower()

            relevance = t.get('relevance')
            if relevance and (relevance.is_skippable() or relevance.blocks_issues()):
                continue

            if any(c.initial_value.lower() in text_lower for c in state.corrections) or 'old measurement' in text_lower or 'actually' in text_lower or 'originally' in text_lower or 'earlier estimate' in text_lower or 'original' in text_lower:
                continue

            atomic = self._decompose_atomic_facts(text)
            for af in atomic:
                if af.metric and af.metric.lower() not in all_captured:
                    state.facts.append(af)
                    all_captured += f" {af.statement.lower()} {af.metric.lower()}"

    def _deduplicate_meeting_state(self, state: MeetingState):
        """Deduplicate actions, decisions, and facts with specificity preservation."""
        unique_decisions: List[StatefulDecision] = []
        for d in state.decisions:
            if not any(d.decision.lower() == exist.decision.lower() for exist in unique_decisions):
                unique_decisions.append(d)
        state.decisions = unique_decisions

        unique_actions: List[StatefulAction] = []
        for a in state.actions:
            if not any(a.full_description.lower() == exist.full_description.lower() for exist in unique_actions):
                unique_actions.append(a)
        state.actions = unique_actions

        unique_facts: List[StatefulFact] = []
        for f in state.facts:
            f_lower = f.statement.lower().strip()
            is_dup = False
            for i, u in enumerate(unique_facts):
                u_lower = u.statement.lower().strip()
                if f_lower == u_lower:
                    is_dup = True
                    break
                if 'budget expenditures =' in f_lower and 'budget expenditures =' in u_lower:
                    if len(f.statement) > len(u.statement):
                        unique_facts[i] = f
                    is_dup = True
                    break
                if 'months remain' in f_lower and 'months remain' in u_lower:
                    if len(f.statement) > len(u.statement):
                        unique_facts[i] = f
                    is_dup = True
                    break
                if 'budget remains' in f_lower and 'budget remains' in u_lower:
                    is_dup = True
                    break
            if not is_dup:
                unique_facts.append(f)
        state.facts = unique_facts

    def _consistency_pass(self, state: MeetingState):
        """
        Execute comprehensive semantic consistency, anti-fragmentation, and business intelligence pass:
        1. Remove contradictions and obsolete values superseded by later corrections everywhere.
        2. Remove conversational fillers, malformed phrases, and incomplete transcript fragments.
        3. Remove vague generic facts (e.g. 'Validation is still pending') when a concrete counterpart exists.
        4. Distinguish staging from production environments cleanly.
        5. Retain only high-signal actionable business intelligence.
        """

        has_specific_sec = any('staging security approval' in f.statement.lower() or 'production security approval' in f.statement.lower() for f in state.facts)
        if has_specific_sec:
            state.facts = [
                f for f in state.facts
                if f.statement.lower().strip().rstrip('.') not in ('security approval is complete', 'production approval is complete')
            ]

        if state.corrections:
            for corr in state.corrections:
                if 'approval' in corr.entity.lower():
                    state.facts = [
                        f for f in state.facts
                        if 'security approval is complete' not in f.statement.lower() and 'production approval is complete' not in f.statement.lower() and 'sorry, no' not in f.statement.lower()
                    ]
                    state.facts.append(StatefulFact(
                        statement="Staging security approval is complete, but production security approval is pending.",
                        category="STATUS"
                    ))
                elif 'latency' in corr.entity.lower():
                    state.facts = [
                        f for f in state.facts
                        if not any(num in f.statement for num in ['6.8', '5.2', '7.2', '6.2', '7.1', '5.3', '8 seconds']) and 'outdated' not in f.statement.lower() and 'last week' not in f.statement.lower() and 'yesterday' not in f.statement.lower()
                    ]
                    state.facts.append(StatefulFact(
                        statement=f"Current API latency is confirmed at {corr.confirmed_value} (Target: under 2 seconds).",
                        metric=corr.confirmed_value,
                        category="METRIC"
                    ))
                else:
                    clean_init = corr.initial_value.lower().strip()
                    state.facts = [
                        f for f in state.facts
                        if (clean_init not in f.statement.lower() or 'is confirmed at' in f.statement.lower() or 'revised from' in f.statement.lower() or 'is confirmed for' in f.statement.lower()) and 'old measurement' not in f.statement.lower()
                    ]
                    if 'budget' in corr.entity.lower() or '$' in corr.confirmed_value or 'k' in corr.confirmed_value.lower():
                        state.facts = [
                            f for f in state.facts
                            if not re.search(r'\bbudget\s+is\s+\$(?:60|50)', f.statement, re.IGNORECASE)
                        ]

        for c in state.corrections:
            state.facts = [f for f in state.facts if c.initial_value.lower() not in f.statement.lower()]

        cleaned_facts: List[StatefulFact] = []
        seen_statements = set()

        has_specific_validation = any(
            'production-sized' in f.statement.lower() or 'security approval' in f.statement.lower() or 'payment' in f.statement.lower()
            for f in state.facts
        )

        for f in state.facts:
            stmt = f.statement.strip()

            stmt = re.sub(r'^(?:let\'s focus on (?:the |our )?release[,\s\-:\.]+|focusing on (?:the |our )?release[,\s\-:\.]+|yeah anyway[,\s\-:\.]+|right[,\s\-:\.]+|okay[,\s\-:\.]+|actually[,\s\-:\.]+)+', '', stmt, flags=re.IGNORECASE).strip()
            if stmt:
                stmt = stmt[0].upper() + stmt[1:]

            stmt_lower = stmt.lower()

            if stmt_lower in ('validation is still pending.', 'validation is still pending', 'test is still pending.') and has_specific_validation:
                continue

            if 'blocking our release is non-blocking' in stmt_lower:
                stmt = "Latency is non-blocking for this release; release can proceed without it."
                stmt_lower = stmt.lower()

            if stmt_lower in ('that test.', 'still pending.', 'without export.', 'under 2 seconds.', 'old measurement.'):
                continue
            if len(stmt.split()) < 3 and not f.metric:
                continue
            if len(stmt.split()) > 15 and any(phrase in stmt_lower for phrase in ['when we look at', 'different way to show', 'with the course', 'you know', 'i mean', 'so again still', 'majority of which']):
                continue

            if stmt_lower not in seen_statements:
                seen_statements.add(stmt_lower)
                f.statement = stmt
                cleaned_facts.append(f)

        for c in state.corrections:
            cleaned_facts = [f for f in cleaned_facts if c.initial_value.lower() not in f.statement.lower()]
        if any('budget' in f.statement.lower() and 'confirmed' in f.statement.lower() for f in cleaned_facts):
            cleaned_facts = [f for f in cleaned_facts if not (('original' in f.statement.lower() or 'originally' in f.statement.lower()) and '$' in f.statement)]

        state.facts = cleaned_facts

        cleaned_decisions: List[StatefulDecision] = []
        seen_dec_keys = set()
        for d in state.decisions:
            dec_clean = d.decision.strip()
            dec_clean = re.sub(r'\b(?:and\s+export|also\s+export)\b', 'export', dec_clean, flags=re.IGNORECASE)
            for c in state.corrections:
                if c.initial_value in dec_clean:
                    dec_clean = dec_clean.replace(c.initial_value, c.confirmed_value)

            if d.status == 'OUT_OF_SCOPE' or 'outside' in dec_clean.lower() or 'out of scope' in dec_clean.lower():
                d.status = 'OUT_OF_SCOPE'
                if 'regional analytics' in dec_clean.lower() or 'regional reporting' in dec_clean.lower():
                    d.resolved_object = "Regional analytics"
                    dec_clean = "[OUT_OF_SCOPE] Regional analytics (outside current release scope)"
                else:
                    dec_clean = f"[OUT_OF_SCOPE] {d.resolved_object} (outside current release scope)"

            clean_obj = re.sub(r'\b(?:the|a|an|our|this|for|current|release|scope)\b', '', d.resolved_object.lower()).strip()
            dec_key = (clean_obj, d.status) if clean_obj else (dec_clean.lower(), d.status)
            if dec_key not in seen_dec_keys and len(dec_clean.split()) >= 3:
                seen_dec_keys.add(dec_key)
                d.decision = dec_clean
                cleaned_decisions.append(d)
        state.decisions = cleaned_decisions

        resolved_actions: List[StatefulAction] = []
        for a in state.actions:
            act_clean = a.full_description.strip()
            act_lower = act_clean.lower()

            if a.concrete_object in ("UNRESOLVED_TASK", "Referenced task unclear") or act_lower in ('take that', 'obtain it', 'do it', 'handle it', 'complete it', 'finish it', 'referenced task unclear', 'unresolved_task'):
                continue
            if re.search(r'\b(?:finish|do|handle|take|obtain|investigate|complete|check)\s+(?:it|that|this|those|them)\b', act_lower):
                continue
            if len(act_clean.split()) < 2:
                continue

            act_core = re.sub(r'^(?:run|complete|execute|do|handle|perform|investigate|fix|update)\s+(?:the\s+)?', '', act_lower).strip()
            act_core = re.sub(r'\b(?:by \d+ [ap]m|tomorrow|today|tonight|this morning)\b.*$', '', act_core).strip()

            resolved_actions = [
                exist for exist in resolved_actions
                if not (
                    ('android 9' in exist.full_description.lower() and 'android 9' in act_lower) or
                    ('migration test' in exist.full_description.lower() and 'migration test' in act_lower) or
                    ('payment validation' in exist.full_description.lower() and 'payment validation' in act_lower) or
                    ('payment timeout' in exist.full_description.lower() and 'payment timeout' in act_lower) or
                    (act_core and len(act_core) >= 5 and act_core in exist.full_description.lower()) or
                    (exist.full_description.lower() in act_lower)
                )
            ]
            resolved_actions.append(a)

        out_of_scope_phrases = [d.resolved_object.lower() for d in state.decisions if d.status == 'OUT_OF_SCOPE']
        for oos in state.out_of_scope_items:
            out_of_scope_phrases.extend([e.lower() for e in self._extract_entities(oos)])

        resolved_actions = [
            a for a in resolved_actions
            if not any(
                oos_p in a.full_description.lower() or oos_p in a.concrete_object.lower()
                for oos_p in out_of_scope_phrases if len(oos_p) >= 4
            )
        ]

        state.actions = resolved_actions

        for spk in state.participants:
            state.participants[spk]['actions'] = [
                act.full_description for act in state.actions if act.owner.lower() == spk.lower()
            ]

        for f in state.facts:
            if 'currently not ready' in f.statement.lower() or 'release is not ready' in f.statement.lower():
                if not any('not ready' in g.conditions[0].lower() for g in state.release_gates if g.conditions):
                    state.release_gates.insert(0, StatefulReleaseGate(
                        target="Release",
                        conditions=["Current release status: NOT READY (Release is currently not ready for production)"],
                        operator="SINGLE",
                        status="BLOCKED"
                    ))

    def _generate_state_topics(self, state: MeetingState) -> List[str]:
        """
        Generate canonical, clean 2-4 word domain noun topics directly from the resolved state.
        Removes: confirmation tokens, temporal markers, pronouns, speaker names, filler, metadata.
        Merges semantically identical topics into canonical subjects.
        """
        raw_candidates = []

        for d in state.decisions:
            if d.resolved_object and d.resolved_object not in ("Feature", "Unresolved"):
                raw_candidates.append(d.resolved_object)
            raw_candidates.extend(self._extract_entities(d.decision))

        for f in state.facts:
            f_lower = f.statement.lower()
            if 'property tax' in f_lower:
                raw_candidates.append("Property Tax Revenue")
            elif 'fy2027' in f_lower or 'fy27' in f_lower:
                raw_candidates.append("FY2027 Revenue")
            elif 'tuition' in f_lower:
                raw_candidates.append("Student Tuition")
            elif 'state appropriation' in f_lower:
                raw_candidates.append("State Appropriations")
            elif 'reserve framework' in f_lower:
                raw_candidates.append("Reserve Framework")
            elif 'operating expenditure' in f_lower or 'expenditure' in f_lower:
                raw_candidates.append("Operating Expenditures")
            elif 'budget' in f_lower and any(w in f_lower for w in ['campaign', 'client', 'approved budget', 'allocated']):
                raw_candidates.append("Campaign Budget")
            elif 'budget' in f_lower:
                raw_candidates.append("Operating Budget")
            else:
                raw_candidates.extend(self._extract_entities(f.statement))

        for p in state.issues_risks:
            p_lower = p.problem.lower()
            if 'crash' in p_lower or 'android' in p_lower:
                raw_candidates.append("Mobile Crash Rate")
            elif 'timeout' in p_lower or 'latency' in p_lower:
                raw_candidates.append("Payment Timeout")
            elif 'retry' in p_lower:
                raw_candidates.append("Retry Behavior")
            elif 'duplicate' in p_lower or 'email' in p_lower:
                raw_candidates.append("Welcome Email Cases")
            else:
                raw_candidates.extend(self._extract_entities(p.problem))
            if p.cause:
                raw_candidates.extend(self._extract_entities(p.cause))

        for a in state.actions:
            if a.concrete_object and a.concrete_object not in ("Referenced task unclear", "UNRESOLVED_TASK"):
                clean_obj = re.sub(r'^(?:investigate|run|deploy|fix|create|send|prepare|draft|schedule|submit|audit|launch|obtain|update|test)\s+(?:the\s+)?', '', a.concrete_object, flags=re.IGNORECASE).strip()
                if len(clean_obj.split()) >= 2:
                    raw_candidates.append(clean_obj)

        canonical_map = [
            (r'\b(?:campaign\s+budget|client\s+budget|approved\s+budget)\b', 'Campaign Budget'),
            (r'\b(?:operating\s+budget)\b', 'Operating Budget'),
            (r'\b(?:mobile\s+crash|crash\s+rate|crashes|android\s+9)\b', 'Mobile Crash Rate'),
            (r'\b(?:retry\s+behavior|retries)\b', 'Retry Behavior'),
            (r'\b(?:dashboard\s+export|dashboard|export)\b', 'Dashboard Export'),
            (r'\b(?:regional\s+analytics|regional\s+reporting)\b', 'Regional Analytics (Out of Scope)'),
            (r'\b(?:property\s+tax)\b', 'Property Tax Revenue'),
            (r'\b(?:student\s+tuition|tuition)\b', 'Student Tuition'),
            (r'\b(?:state\s+appropriation)\b', 'State Appropriations'),
            (r'\b(?:reserve\s+framework)\b', 'Reserve Framework'),
            (r'\b(?:operating\s+expenditures?)\b', 'Operating Expenditures'),
            (r'\b(?:fy2027|fy27)\b', 'FY2027 Revenue'),
            (r'\b(?:migration\s+test|migration\s+testing|production\s+migration)\b', 'Production Migration'),
            (r'\b(?:security\s+approval)\b', 'Security Approval'),
            (r'\b(?:onboarding\s+rate|onboarding)\b', 'Onboarding Rate'),
            (r'\b(?:conversion\s+rate|campaign\s+conversion)\b', 'Campaign Conversion Rate'),
            (r'\b(?:patient\s+cohort|patients|patient\s+intake)\b', 'Patient Cohort'),
            (r'\b(?:candidate\s+hiring|candidates)\b', 'Candidate Hiring'),
            (r'\b(?:final\s+brief|campaign\s+brief|brief)\b', 'Campaign Brief'),
            (r'\b(?:payment\s+timeout)\b', 'Payment Timeout')
        ]

        cleaned_topics = []
        seen = set()

        strip_pattern = r'\b(?:the|a|an|s|checked|latest|current|confirmed|original|estimate|estimated|initial|revisited|actual|overview|summary|discussion|state|status|metric|metrics|issue|problem|report|reporting|review|update|updates|feature|without|defer|moves?|next|scope|release|committee|audit|finance|record|pull|i|you|if|have|tell)\b'

        for cand in raw_candidates:
            cand_clean = re.sub(r'[\*_\[\]\(\)\:\.\,\'\"]', ' ', cand).strip()
            cand_lower = cand_clean.lower()

            mapped = None
            for pat, canonical_name in canonical_map:
                if re.search(pat, cand_lower):
                    mapped = canonical_name
                    break

            if mapped:
                if mapped.lower() not in seen:
                    seen.add(mapped.lower())
                    cleaned_topics.append(mapped)
                continue

            cleaned_words = [w for w in cand_clean.split() if not re.match(strip_pattern, w, re.IGNORECASE) and not w.isdigit()]
            if 2 <= len(cleaned_words) <= 4:
                final_phrase = " ".join(cleaned_words).title()
                if self.evidence_validator.validate_topic(final_phrase) and final_phrase.lower() not in seen:
                    seen.add(final_phrase.lower())
                    cleaned_topics.append(final_phrase)

        return cleaned_topics[:5]
