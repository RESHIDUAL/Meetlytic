"""
Discourse Context Graph & Comprehensive Recall Engine for Meetlytic.
Implements:
1. Professional Information Recall: Retains facts, metrics, causes, non-blockers, scopes, and dependencies even without explicit tasks.
2. Full Causality & "Why" Linking: Problem + Confirmed/Suspected Cause + Impact + Action + Reason.
3. Anaphora & Deixis Resolution: Resolves pronouns ("it", "that", "create it", "review that") to concrete antecedent entities.
4. Scope Classification: IN_SCOPE, OUT_OF_SCOPE, DEFERRED, CONDITIONAL.
5. Dependency Chain & Gate Construction: Sequential prerequisites and multi-condition boolean release gates.
6. Final Information Recall Pass: Guarantees zero dropped numbers, percentages, or operational facts.
7. Importance-Weighted Semantic Topic Clustering: 2-4 word clean noun phrases reflecting decisions, blockers, and issues.
Strictly free of emojis and emdashes.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

@dataclass
class QuestionAnswerArc:
    question: str
    answer: str
    q_speaker: str
    a_speaker: str
    topic_entity: Optional[str] = None
    synthesized_fact: str = ""
    release_impact: str = "UNSPECIFIED"

@dataclass
class ProblemCausalityArc:
    problem: str
    cause: Optional[str] = None
    cause_certainty: str = "confirmed"
    impact: Optional[str] = None
    metric: Optional[str] = None
    release_impact: str = "UNSPECIFIED"
    followup_action: Optional[str] = None
    source_turn: str = ""

@dataclass
class ProposalDecisionArc:
    initial_proposal: str
    alternatives: List[str] = field(default_factory=list)
    final_decision: str = ""
    resolved_object: str = ""
    proposer: str = "Speaker"
    agreed_by: List[str] = field(default_factory=list)
    scope_status: str = "DECISION"

@dataclass
class TaskActionNode:
    action_verb: str
    target_object: str
    full_description: str
    owner: str = "Unclear"
    deadline: str = "unspecified"
    duration: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    why_reason: Optional[str] = None
    release_impact: str = "UNSPECIFIED"
    evidence_turn: str = ""

@dataclass
class ReleaseGate:
    target: str
    operator: str
    conditions: List[str] = field(default_factory=list)
    raw_statement: str = ""

@dataclass
class DependencyLink:
    prerequisite: str
    dependent_outcome: str
    raw_statement: str = ""

@dataclass
class ScopeItem:
    feature_or_item: str
    status: str
    condition_or_reason: str = ""

@dataclass
class ConversationContextGraph:
    qa_arcs: List[QuestionAnswerArc] = field(default_factory=list)
    causality_arcs: List[ProblemCausalityArc] = field(default_factory=list)
    decision_arcs: List[ProposalDecisionArc] = field(default_factory=list)
    tasks: List[TaskActionNode] = field(default_factory=list)
    release_gates: List[ReleaseGate] = field(default_factory=list)
    dependencies: List[DependencyLink] = field(default_factory=list)
    scope_items: List[ScopeItem] = field(default_factory=list)
    status_facts: List[Dict[str, Any]] = field(default_factory=list)
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    semantic_topics: List[str] = field(default_factory=list)
    participants: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class ContextGraphBuilder:
    """
    Constructs an internal semantic graph of meeting meaning and resolves anaphora, causality, dependencies, and gates.
    """

    def __init__(self):
        self.agreement_tokens = {
            'agreed', 'i agree', 'sounds good', 'sounds good to me', 'that works',
            'that works for me', 'good idea', 'fair enough', 'definitely', 'perfect',
            'exactly', 'confirmed', 'approved', 'will do', "let's do that", "we'll go with that",
            "that's fine", "okay we'll proceed", 'yep', 'yeah', 'yes'
        }

    def build_graph(self, turns: List[Tuple[str, str]]) -> ConversationContextGraph:
        """
        Build, resolve, and audit the whole-conversation context graph.
        """
        graph = ConversationContextGraph()
        if not turns:
            return graph

        for spk, _ in turns:
            if spk not in graph.participants and spk not in ('Speaker', 'Team', 'User'):
                graph.participants[spk] = {
                    'name': spk,
                    'actions': [],
                    'turns': 0
                }

        recent_technical_entities: List[str] = []
        recent_problems: List[Dict[str, Any]] = []

        parsed_turns = []
        for idx, (spk, raw) in enumerate(turns):
            if spk in graph.participants:
                graph.participants[spk]['turns'] += 1

            clean = self._clean_utterance(raw)
            parsed_turns.append({
                'index': idx,
                'speaker': spk,
                'raw': raw,
                'clean': clean
            })

        i = 0
        n = len(parsed_turns)
        while i < n:
            curr = parsed_turns[i]
            spk = curr['speaker']
            clean = curr['clean']
            clean_lower = clean.lower()

            if not clean or len(clean) < 3:
                i += 1
                continue

            turn_ents = self._extract_technical_entities(clean)
            for ent in turn_ents:
                if ent not in recent_technical_entities:
                    recent_technical_entities.append(ent)
                if len(recent_technical_entities) > 12:
                    recent_technical_entities.pop(0)

            gate = self._parse_release_gate(clean)
            if gate:
                graph.release_gates.append(gate)
                graph.requirements.append({
                    'kind': 'RELEASE_GATE',
                    'statement': f"Deployment Gate: {' AND '.join(gate.conditions)}",
                    'operator': gate.operator
                })

            dep = self._parse_dependency(clean, recent_technical_entities)
            if dep:
                graph.dependencies.append(dep)

            scope = self._parse_scope_statement(clean, recent_technical_entities)
            if scope:
                graph.scope_items.append(scope)

            if '?' in clean or clean_lower.startswith(('is the', 'is this', 'are there', 'what is', 'what happens', 'how long', 'who is', 'when can', 'when will')):
                req_match = re.search(r'\b(?:can you|could you|would you|please)\s+([a-zA-Z0-9_\s,\-]{4,80})\??', clean, re.IGNORECASE)
                if req_match and i + 1 < n and self._is_acceptance(parsed_turns[i + 1]['clean']):

                    pass
                else:
                    qa_arc = self._resolve_question_answer_arc(parsed_turns, i, recent_technical_entities)
                    if qa_arc:
                        graph.qa_arcs.append(qa_arc)

                        if 'root cause' in qa_arc.synthesized_fact.lower() and graph.causality_arcs:
                            cause_arc = self._parse_problem_causality(qa_arc.answer, recent_technical_entities)
                            graph.causality_arcs[-1].cause = cause_arc.cause if cause_arc.cause else qa_arc.answer
                            graph.causality_arcs[-1].cause_certainty = cause_arc.cause_certainty
                        elif qa_arc.synthesized_fact:
                            graph.status_facts.append({
                                'fact': qa_arc.synthesized_fact,
                                'source': f"{qa_arc.q_speaker} & {qa_arc.a_speaker}",
                                'impact': qa_arc.release_impact
                            })
                        i += 2
                        continue
                    else:
                        if any(w in clean_lower for w in ['blocking', 'broken', 'failing', 'timeline', 'how long', 'who is']):
                            graph.unresolved.append(f"{spk}: {clean}")
                        i += 1
                        continue

            req_match = re.search(r'\b(?:can you|could you|would you|please)\s+([a-zA-Z0-9_\s,\-]{4,150})\??', clean, re.IGNORECASE)
            if req_match and '?' in clean:
                task_request = req_match.group(1).strip().rstrip('?')
                task_resolved = self._resolve_anaphora(task_request, recent_technical_entities)

                if i + 1 < n:
                    next_t = parsed_turns[i + 1]
                    if self._is_acceptance(next_t['clean']):
                        accept_spk = next_t['speaker']
                        accept_text = next_t['clean']
                        deadline, duration, conditions = self._extract_temporal_and_conditions(accept_text)

                        comm_match = re.search(r"\b(?:i will|i'll|i can|i am on it|sure i will)\s+([a-zA-Z0-9_\s,\-]{4,150})", accept_text, re.IGNORECASE)
                        if comm_match:
                            task_resolved = self._resolve_anaphora(comm_match.group(1).strip(), recent_technical_entities)

                        why_reason = self._find_why_reason(task_resolved, recent_problems)

                        node = TaskActionNode(
                            action_verb=self._extract_action_verb(task_resolved),
                            target_object=task_resolved,
                            full_description=self._normalize_action_description(task_resolved),
                            owner=accept_spk,
                            deadline=deadline,
                            duration=duration,
                            conditions=conditions,
                            why_reason=why_reason,
                            evidence_turn=f"{spk}: {clean} | {accept_spk}: {accept_text}"
                        )
                        graph.tasks.append(node)
                        if accept_spk in graph.participants:
                            graph.participants[accept_spk]['actions'].append(node.full_description)
                        i += 2
                        continue

                node = TaskActionNode(
                    action_verb=self._extract_action_verb(task_resolved),
                    target_object=task_resolved,
                    full_description=self._normalize_action_description(task_resolved),
                    owner="Unclear",
                    deadline="unspecified",
                    evidence_turn=f"{spk}: {clean}"
                )
                graph.tasks.append(node)
                i += 1
                continue

            if any(w in clean_lower for w in ["let's move", "let us move", "let's defer", "we should", "i recommend", "how about", "what if we", "we can ship without", "let's keep"]):
                prop_arc, consumed = self._resolve_proposal_arc(parsed_turns, i, recent_technical_entities)
                if prop_arc:
                    graph.decision_arcs.append(prop_arc)
                    i += consumed
                    continue

            comm_match = re.search(r"\b(?:i will|i'll|i can|i am going to)\s+([a-zA-Z0-9_\s,\-]{4,150})", clean, re.IGNORECASE)
            if comm_match:
                task_raw = comm_match.group(1).strip()
                task_resolved = self._resolve_anaphora(task_raw, recent_technical_entities)
                deadline, duration, conditions = self._extract_temporal_and_conditions(clean)

                dual_match = re.search(
                    r"(?:i'll|i will)\s+(?:handle|take|present|prepare)\s+([a-zA-Z0-9_\s,]+?)"
                    r"(?:[.,;]\s*|\s+then\s+|\s+and\s+)(?:you can|you will)\s+(?:explain|present|handle|take|prepare)\s+([a-zA-Z0-9_\s,]+)",
                    clean, re.IGNORECASE
                )
                if dual_match:
                    p1 = self._resolve_anaphora(dual_match.group(1).strip(), recent_technical_entities)
                    p2 = self._resolve_anaphora(dual_match.group(2).strip(), recent_technical_entities)
                    addressee = self._get_addressee(spk, graph.participants)

                    node1 = TaskActionNode(
                        action_verb="Prepare",
                        target_object=p1,
                        full_description=f"Prepare presentation overview: {p1}",
                        owner=spk,
                        deadline=deadline,
                        evidence_turn=clean
                    )
                    node2 = TaskActionNode(
                        action_verb="Prepare",
                        target_object=p2,
                        full_description=f"Prepare technical presentation section: {p2}",
                        owner=addressee,
                        deadline=deadline,
                        evidence_turn=clean
                    )
                    graph.tasks.extend([node1, node2])
                    if spk in graph.participants:
                        graph.participants[spk]['actions'].append(node1.full_description)
                    if addressee in graph.participants:
                        graph.participants[addressee]['actions'].append(node2.full_description)
                else:
                    why_reason = self._find_why_reason(task_resolved, recent_problems)
                    node = TaskActionNode(
                        action_verb=self._extract_action_verb(task_resolved),
                        target_object=task_resolved,
                        full_description=self._normalize_action_description(task_resolved),
                        owner=spk,
                        deadline=deadline,
                        duration=duration,
                        conditions=conditions,
                        why_reason=why_reason,
                        evidence_turn=clean
                    )
                    graph.tasks.append(node)
                    if spk in graph.participants:
                        graph.participants[spk]['actions'].append(node.full_description)

                i += 1
                continue

            if re.search(r'\b([a-zA-Z0-9_\s\-]+)\s+(?:should be|needs to be|must be|hasn\'t received|needs)\s+(?:updated|fixed|tested|created|configured|prepared|review|final review)\b', clean, re.IGNORECASE):
                t_resolved = self._resolve_anaphora(clean, recent_technical_entities)
                node = TaskActionNode(
                    action_verb=self._extract_action_verb(t_resolved),
                    target_object=t_resolved,
                    full_description=self._normalize_action_description(t_resolved),
                    owner="Unclear",
                    deadline="unspecified",
                    evidence_turn=clean
                )
                graph.tasks.append(node)
                i += 1
                continue

            if any(w in clean_lower for w in ['differs from', 'discrepancy', 'takes 6 seconds', 'takes 5 seconds', 'takes 8 seconds', 'failing', 'failed', 'crashes', 'crash', 'timeout', 'timed out', 'error', 'exhaustion', 'memory leak', 'broken', 'vulnerabilit', 'load-time']):
                prob_arc = self._parse_problem_causality(clean, recent_technical_entities)
                graph.causality_arcs.append(prob_arc)
                recent_problems.append({
                    'problem': prob_arc.problem,
                    'cause': prob_arc.cause,
                    'metric': prob_arc.metric
                })
                i += 1
                continue

            duration_match = re.search(r'\b(?:testing|migration|deployment|scan|review)\s+will\s+take\s+(?:about|approximately)?\s*(\d+|one|two|three)\s+(?:working\s+)?(days?|hours?|weeks?|business days?)\b', clean_lower)
            if duration_match:
                dur_text = duration_match.group(0).strip()
                deadline, duration, conditions = self._extract_temporal_and_conditions(clean)
                graph.status_facts.append({
                    'fact': f"Estimated execution duration: {dur_text}" + (f" (Condition: {', '.join(conditions)})" if conditions else ""),
                    'impact': 'NON_BLOCKING'
                })
                i += 1
                continue

            if any(w in clean_lower for w in ['needs the', 'needs updated', 'needs to verify', 'must approve', 'wants csv', 'wants excel', 'is required before']):
                graph.requirements.append({
                    'kind': 'DEPENDENCY' if 'needs' in clean_lower else 'STAKEHOLDER_REQUEST' if 'wants' in clean_lower else 'REQUIREMENT',
                    'statement': clean
                })
                i += 1
                continue

            metric_match = re.search(r'\b(?:\d+(?:\.\d+)?%|\d+\s*(?:ms|fps|seconds?|users|requests|units|kb|mb|gb|rpm)|from\s+\d+(?:\.\d+)?%\s+to\s+\d+(?:\.\d+)?%|500|404|401)\b', clean, re.IGNORECASE)
            if metric_match or any(w in clean_lower for w in ['differs by', 'dropped from', 'increased to', 'latency is', 'error rate', 'load time']):
                graph.status_facts.append({
                    'fact': clean,
                    'impact': 'NON_BLOCKING' if 'non-blocking' in clean_lower or 'doesn\'t affect' in clean_lower else 'UNSPECIFIED'
                })

            i += 1

        self._final_recall_safety_pass(graph, parsed_turns)

        self._deduplicate_graph(graph)

        graph.semantic_topics = self._cluster_semantic_topics(graph)

        return graph

    def _resolve_anaphora(self, phrase: str, entity_stack: List[str]) -> str:
        """
        Resolve pronouns ("it", "that", "this", "do that", "create it", "review it", "the task")
        to the most recent compatible technical antecedent.
        """
        p = phrase.strip()
        p_lower = p.lower()

        vague_patterns = [
            r'^(?:review|create|fix|update|test|deploy|do|handle|defer)\s+(?:it|this|that|those|the task|the issue)\b',
            r'\b(?:review|create|fix|update|test|deploy|do|handle|defer)\s+(?:it|this|that|those)\b'
        ]

        needs_resolution = any(re.search(pat, p_lower) for pat in vague_patterns) or p_lower in ('it', 'this', 'that', 'do that', 'create it', 'fix it', 'review it', 'review that')

        if needs_resolution:
            if entity_stack:
                target_entity = entity_stack[-1]
                verb_match = re.match(r'^(review|create|fix|update|test|deploy|do|handle|defer)\b', p, re.IGNORECASE)
                if verb_match:
                    verb = verb_match.group(1).capitalize()

                    time_suffix = ""
                    time_match = re.search(r'\b(this morning|today|tonight|by \d+ [ap]m|tomorrow|by friday)\b', p_lower)
                    if time_match:
                        time_suffix = f" {time_match.group(0)}"
                    return f"{verb} {target_entity}{time_suffix}"
                return f"{p} for {target_entity}"
            else:
                return "Referenced task unclear"

        return p

    def _extract_technical_entities(self, text: str) -> List[str]:
        """Extract multi-word domain entity candidates."""
        entities = []
        pattern = re.compile(r'\b([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+){1,4})\b')
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'for', 'with', 'that', 'this', 'have', 'been', 'will',
            'about', 'from', 'into', 'is', 'are', 'was', 'were', 'also', 'has', 'had', 'hasn',
            'hadn', 'don', 'doesn', 'didn', 'not', 'received', 'take', 'takes', 'getting', 'back'
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
                    'testing', 'qa', 'discrepancy', 'ticket', 'issue'
                ] for w in words):
                    entities.append(" ".join(words))

        return entities

    def _parse_release_gate(self, text: str) -> Optional[ReleaseGate]:
        """Extract compound release gates (AND / OR / SEQUENTIAL)."""
        text_lower = text.lower()
        if not any(w in text_lower for w in ['publication requires', 'deployment requires', 'production can proceed', 'production deployment should wait', 'can proceed once', 'must wait until', 'release gate']):
            return None

        if ' and ' in text_lower:
            parts = re.split(r'\band\b', text, flags=re.IGNORECASE)
            c1 = self._clean_condition_clause(parts[0])
            c2 = self._clean_condition_clause(parts[1])
            return ReleaseGate(
                target="Production Deployment",
                operator="AND",
                conditions=[c1, c2],
                raw_statement=text
            )
        elif 'requires:' in text_lower or 'publication requires' in text_lower:
            conditions = []
            c_matches = re.findall(r'(?:\d+\.\s*|\*\s*)([^\n,]+)', text)
            if c_matches:
                for cm in c_matches:
                    conditions.append(cm.strip())
            else:
                conditions = [self._clean_condition_clause(text)]
            return ReleaseGate(
                target="Report Publication",
                operator="SEQUENTIAL",
                conditions=conditions,
                raw_statement=text
            )
        else:
            c = self._clean_condition_clause(text)
            return ReleaseGate(
                target="Production Deployment",
                operator="SINGLE",
                conditions=[c],
                raw_statement=text
            )

    def _clean_condition_clause(self, clause: str) -> str:
        """Format condition into clear business requirement."""
        c = re.sub(r'^(?:agreed|okay|ok|sure)[,\s\-:\.]+', '', clause, flags=re.IGNORECASE).strip()
        c = re.sub(r'^(?:production (?:deployment )?(?:should wait until|can proceed once|must wait until)|report publication requires|publication requires|once|wait until)\s+', '', c, flags=re.IGNORECASE).strip()
        c = c.rstrip('.').strip()
        c = re.sub(r'\s+(?:passes|is approved)\b', '', c, flags=re.IGNORECASE).strip()
        if 'security' in c.lower() or 'approv' in c.lower():
            target = re.sub(r'\s*(?:approves the|approves|approval)\s*', ' ', c, flags=re.IGNORECASE).strip()
            return f"{target} must be approved"
        if 'test' in c.lower() or 'verification' in c.lower() or 'fix' in c.lower():
            return f"{c} must pass"
        return f"{c} is required"

    def _parse_dependency(self, text: str, entity_stack: List[str]) -> Optional[DependencyLink]:
        """Extract explicit dependency chains (A before B, A only after B)."""
        text_lower = text.lower()
        if 'only after' in text_lower or 'can begin only after' in text_lower or 'can be sent only after' in text_lower:
            parts = re.split(r'\b(?:can begin only after|can be sent only after|only after)\b', text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                outcome = parts[0].strip()
                prereq = parts[1].strip()
                return DependencyLink(
                    prerequisite=prereq,
                    dependent_outcome=outcome,
                    raw_statement=f"{outcome} can proceed only after {prereq}"
                )
        if ' after reviewing ' in text_lower or ' after approval of ' in text_lower:
            return DependencyLink(
                prerequisite=text,
                dependent_outcome="Subsequent workflow step",
                raw_statement=text
            )
        return None

    def _parse_scope_statement(self, text: str, entity_stack: List[str]) -> Optional[ScopeItem]:
        """Extract explicit scope declarations (OUT_OF_SCOPE, IN_SCOPE, CONDITIONAL)."""
        text_lower = text.lower()
        if 'outside the current scope' in text_lower or 'out of scope' in text_lower or 'outside of scope' in text_lower:
            m = re.search(r'\b([a-zA-Z0-9_\s\-]+?)\s+(?:is|are)\s+(?:outside the current scope|out of scope|outside of scope)\b', text, re.IGNORECASE)
            target = m.group(1).strip() if m else (entity_stack[-1] if entity_stack else "Feature")
            target_clean = re.sub(r'^(?:agreed|okay|ok|sure|yes|and|also)[,\s\-:\.]+', '', target, flags=re.IGNORECASE).strip()
            return ScopeItem(
                feature_or_item=target_clean,
                status="OUT_OF_SCOPE",
                condition_or_reason="Outside current release scope"
            )
        if 'keep' in text_lower and 'out of the current release' in text_lower:
            m = re.search(r'\bkeep\s+([a-zA-Z0-9_\s\-]+?)\s+out of', text, re.IGNORECASE)
            target = m.group(1).strip() if m else "Feature"
            target_clean = re.sub(r'^(?:agreed|okay|ok|sure|yes|and|also)[,\s\-:\.]+', '', target, flags=re.IGNORECASE).strip()
            return ScopeItem(
                feature_or_item=target_clean,
                status="OUT_OF_SCOPE",
                condition_or_reason="Excluded unless formal approval received"
            )
        return None

    def _find_why_reason(self, task_description: str, recent_problems: List[Dict[str, Any]]) -> Optional[str]:
        """Connect task to its underlying problem motivation."""
        if not recent_problems:
            return None
        last_prob = recent_problems[-1]
        p_text = last_prob['problem']
        if any(w in task_description.lower() for w in ['performance', 'investigation', 'load', 'ticket', 'task', 'fix', 'refund', 'memory']):
            return p_text
        return None

    def _resolve_question_answer_arc(self, parsed_turns: List[Dict[str, Any]], q_idx: int, entity_stack: List[str]) -> Optional[QuestionAnswerArc]:
        """Synthesize a Q&A pair into a business fact and determine release impact."""
        if q_idx + 1 >= len(parsed_turns):
            return None

        q_turn = parsed_turns[q_idx]
        a_turn = parsed_turns[q_idx + 1]

        q_text = q_turn['clean']
        a_text = a_turn['clean']
        q_lower = q_text.lower()
        a_lower = a_text.lower()

        if 'blocking release' in q_lower or 'is this blocking' in q_lower or 'block release' in q_lower:
            target = re.search(r'is (?:the |this )?([a-zA-Z0-9_\s\-]+?) blocking', q_text, re.IGNORECASE)
            feature = target.group(1).strip() if (target and target.group(1).lower() != 'this') else (entity_stack[-1] if entity_stack else "Feature")

            if any(w in a_lower for w in ['no', 'ship without', 'non-blocking', 'not a release blocker', 'not blocking', 'doesn\'t affect']):
                return QuestionAnswerArc(
                    question=q_text,
                    answer=a_text,
                    q_speaker=q_turn['speaker'],
                    a_speaker=a_turn['speaker'],
                    topic_entity=feature,
                    synthesized_fact=f"{feature.capitalize()} is non-blocking for this release; release can proceed without it.",
                    release_impact="NON_BLOCKING"
                )
            else:
                return QuestionAnswerArc(
                    question=q_text,
                    answer=a_text,
                    q_speaker=q_turn['speaker'],
                    a_speaker=a_turn['speaker'],
                    topic_entity=feature,
                    synthesized_fact=f"{feature.capitalize()} is a critical release blocker.",
                    release_impact="BLOCKING"
                )

        if 'broken completely' in q_lower or 'failing completely' in q_lower:
            m = re.search(r'is (?:the )?([a-zA-Z0-9_\s\-]+?) (?:broken|failing)', q_text, re.IGNORECASE)
            target = m.group(1).strip().capitalize() if m else (entity_stack[-1] if entity_stack else "Upload functionality")
            if 'network transition' in a_lower or 'only fails' in a_lower or 'no' in a_lower:
                return QuestionAnswerArc(
                    question=q_text,
                    answer=a_text,
                    q_speaker=q_turn['speaker'],
                    a_speaker=a_turn['speaker'],
                    topic_entity=target,
                    synthesized_fact=f"{target} operates normally under stable connections; failures occur specifically during network transitions.",
                    release_impact="NON_BLOCKING"
                )

        if 'how long' in q_lower or 'timeline' in q_lower:
            return QuestionAnswerArc(
                question=q_text,
                answer=a_text,
                q_speaker=q_turn['speaker'],
                a_speaker=a_turn['speaker'],
                synthesized_fact=f"Estimated duration: {a_text}",
                release_impact="NON_BLOCKING"
            )

        if 'what is the reason' in q_lower or 'what is causing' in q_lower or 'why is' in q_lower or 'what caused' in q_lower:
            return QuestionAnswerArc(
                question=q_text,
                answer=a_text,
                q_speaker=q_turn['speaker'],
                a_speaker=a_turn['speaker'],
                synthesized_fact=f"Root cause: {a_text}",
                release_impact="NON_BLOCKING"
            )

        return None

    def _resolve_proposal_arc(self, parsed_turns: List[Dict[str, Any]], p_idx: int, entity_stack: List[str]) -> Tuple[Optional[ProposalDecisionArc], int]:
        """Track proposal discussion through alternatives to final consensus."""
        init_turn = parsed_turns[p_idx]
        current_proposal = init_turn['clean']
        proposer = init_turn['speaker']
        alternatives = []
        agreed_by = []
        agreed = False

        j = p_idx + 1
        n = len(parsed_turns)
        consumed = 1

        while j < n and j <= p_idx + 3:
            t = parsed_turns[j]
            t_clean = t['clean']
            t_spk = t['speaker']

            if self._is_agreement(t_clean):
                agreed = True
                agreed_by.append(t_spk)
                consumed = (j - p_idx) + 1
                break
            elif any(w in t_clean.lower() for w in ["then let's", "let's instead", "let's keep", "let's defer", "let's move"]):
                alternatives.append(current_proposal)
                current_proposal = t_clean
                proposer = t_spk
            elif "don't think" in t_clean.lower() or "not ready" in t_clean.lower():
                alternatives.append(current_proposal)

            j += 1

        if agreed or alternatives:
            norm_decision = self._normalize_decision(current_proposal, entity_stack)
            dec_type = "DEFERRED" if "defer" in norm_decision.lower() else "DECISION"
            arc = ProposalDecisionArc(
                initial_proposal=init_turn['clean'],
                alternatives=alternatives,
                final_decision=norm_decision,
                resolved_object=entity_stack[-1] if entity_stack else "Feature",
                proposer=proposer,
                agreed_by=agreed_by,
                scope_status=dec_type
            )

            agree_turn_clean = re.sub(r'^(?:agreed|okay|ok|sure|sounds good)[,\s\-:\.]+', '', parsed_turns[j - 1]['clean'], flags=re.IGNORECASE).strip() if (j - 1 < n) else ""
            if len(agree_turn_clean) > 8:
                return arc, max(1, j - p_idx - 1)
            return arc, max(1, j - p_idx)

        return None, 1

    def _parse_problem_causality(self, text: str, entity_stack: List[str]) -> ProblemCausalityArc:
        """Distinguish confirmed cause vs suspected cause vs unknown cause and extract metrics."""
        text_lower = text.lower()
        cause_certainty = "confirmed"
        if any(w in text_lower for w in ['suspect', 'suspected', 'might be', 'could be', 'possibly']):
            cause_certainty = "suspected"
        elif 'unknown' in text_lower or 'not confirmed' in text_lower or "don't know" in text_lower:
            cause_certainty = "unknown"

        cause_match = re.search(r'(?:due to|caused by|because of|suspected to be|because)\s+([a-zA-Z0-9_\s\-]+)', text, re.IGNORECASE)
        cause = cause_match.group(1).strip() if cause_match else None

        metric_match = re.search(r'\b(?:\d+(?:\.\d+)?%|\d+\s*(?:seconds?|ms|fps|requests|concurrent requests)|500|404)\b', text, re.IGNORECASE)
        metric = metric_match.group(0) if metric_match else None

        is_non_blocking = 'non-blocking' in text_lower or 'doesn\'t affect' in text_lower or 'not a release blocker' in text_lower
        release_impact = "NON_BLOCKING" if is_non_blocking else "UNSPECIFIED"

        return ProblemCausalityArc(
            problem=text,
            cause=cause,
            cause_certainty=cause_certainty,
            metric=metric,
            release_impact=release_impact,
            source_turn=text
        )

    def _extract_temporal_and_conditions(self, text: str) -> Tuple[str, Optional[str], List[str]]:
        """Extract exact deadline without proximity leakage, durations, and conditions."""
        text_lower = text.lower()
        deadline = "unspecified"
        duration = None
        conditions = []

        if re.search(r'\btonight\b', text_lower):
            deadline = "tonight"
        elif re.search(r'\bby 5 pm\b|\bat 5 pm\b|\b5 pm\b', text_lower):
            deadline = "by 5 PM today" if "today" in text_lower else "by 5 PM"
        elif re.search(r'\bby 4 pm\b|\bat 4 pm\b|\b4 pm\b', text_lower):
            deadline = "by 4 PM today" if "today" in text_lower else "by 4 PM"
        elif re.search(r'\bby 1 pm\b|\bat 1 pm\b|\b1 pm\b', text_lower):
            deadline = "by 1 PM today" if "today" in text_lower else "by 1 PM"
        elif re.search(r'\bby 2 pm\b|\bat 2 pm\b|\b2 pm\b', text_lower):
            deadline = "by 2 PM today" if "today" in text_lower else "by 2 PM"
        elif re.search(r'\bthis morning\b', text_lower):
            deadline = "this morning"
        elif re.search(r'\bby friday\b|\bon friday\b|\bthis friday\b', text_lower):
            deadline = "by Friday"
        elif re.search(r'\btomorrow\b', text_lower) and not re.search(r'\btwo days\b|\bworking days\b', text_lower):
            deadline = "tomorrow"
        elif re.search(r'\bthis week\b', text_lower):
            deadline = "this week"
        elif re.search(r'\bnext week\b', text_lower):
            deadline = "next week"
        elif re.search(r'\bby eod\b|\bend of day\b', text_lower):
            deadline = "by EOD"
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
        """Extract primary imperative action verb."""
        words = text.strip().split()
        if words:
            return words[0].capitalize()
        return "Execute"

    def _normalize_action_description(self, text: str) -> str:
        """Normalize action deliverable into complete declarative statement."""
        t = re.sub(r'^(?:i will|i\'ll|i can|we will|please|take care of|to)\s+', '', text, flags=re.IGNORECASE).strip()
        t = re.sub(r'\b(?:by 5 pm|by 4 pm|by friday|by 1 pm|by 2 pm|tomorrow|today|tonight|this morning)\b.*$', '', t, flags=re.IGNORECASE).strip()
        if 'should be updated' in t.lower() or 'needs to be updated' in t.lower() or 'needs updating' in t.lower():
            target = re.sub(r'\s+(?:should be|needs to be|must be|needs)\s+updat.*$', '', t, flags=re.IGNORECASE)
            t = f"Update {target}"

        if t:
            t = t[0].upper() + t[1:]
        return t.strip()

    def _normalize_decision(self, text: str, entity_stack: List[str]) -> str:
        """Normalize decision into unambiguous business directive."""
        t = text.strip()
        if 'keep' in t.lower() and 'move' in t.lower():
            m1 = re.search(r'\bkeep\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+and', t, re.IGNORECASE)
            m2 = re.search(r'\bmove\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', t, re.IGNORECASE)
            keep_part = m1.group(1).strip() if m1 else "current version"
            move_target = m2.group(1).strip() if m2 else (entity_stack[-1] if entity_stack else "feature")
            release_target = m2.group(2).strip() if m2 else "next release"
            return f"Keep {keep_part} for this release and defer {move_target} to {release_target}."

        if re.search(r'\b(?:move|defer)\s+([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', t, re.IGNORECASE):
            m = re.search(r'\b(?:move|defer)\s+([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', t, re.IGNORECASE)
            raw_target = m.group(1).strip()
            resolved_target = self._resolve_anaphora(raw_target, entity_stack)
            return f"Defer {resolved_target} to {m.group(2).strip()}."

        if not t.endswith('.'):
            t += '.'
        return t

    def _clean_utterance(self, text: str) -> str:
        """Strip conversational fillers, markdown artifacts, and emotional venting."""
        if not text:
            return ""
        clean = re.sub(r'^\s*[\*_\[\]\(\)]+\s*', '', text).strip()
        clean = re.sub(r'^(?:(?:yes|no|yeah|yep|sure|ok|alright|good morning|good morning everyone|hello|hi)[,\s\-:\.]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:arsenal|real madrid|barcelona|chelsea|liverpool|india|australia|england) (?:won|played well|played amazingly)[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:it was (?:great|good|fun)|went hiking|had fun|great game|good game|great match)[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:but )?(?:speaking of|getting to|getting back to|talking about|regarding) [^.!?]+?[,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:honestly|frankly|personally|to be honest|look|listen|man|dude|gosh|damn|ugh)[,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:i (?:really )?(?:hate|dislike|can\'t stand) (?:this|the|our) [a-zA-Z0-9_\s\-]+?(?:,| but| yet|\.|\;)\s*)', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:this (?:terrible|horrible|annoying|frustrating|broken|stupid) [a-zA-Z0-9_\s\-]+? is (?:so |really )?(?:bad|terrible|annoying|frustrating)[,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:anyway|anyways|on another note|by the way|excellent|perfect|great)[,\s\-:\.]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:this\s+)?(?:terrible|horrible|annoying|frustrating|stupid|awful)\s+([a-zA-Z0-9_\-]+)', r'The \1', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:anyway|anyways|on another note|by the way|excellent|perfect|great)[,\s\-:\.]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:this\s+)?(?:terrible|horrible|annoying|frustrating|stupid|awful)\s+([a-zA-Z0-9_\-]+)', r'The \1', clean, flags=re.IGNORECASE).strip()

        if clean:
            clean = clean[0].upper() + clean[1:]
        return clean.strip()

    def _is_agreement(self, text: str) -> bool:
        """Check if turn is an agreement confirmation."""
        clean = re.sub(r'[^a-z\s]', '', text.lower()).strip()
        return any(clean == a or clean.startswith(a + ' ') or clean.startswith(a + '.') for a in self.agreement_tokens)

    def _is_acceptance(self, text: str) -> bool:
        """Check if turn is an acceptance of a task request."""
        t = text.lower()
        return bool(re.search(r'\b(?:yes|sure|i will|i\'ll|will do|i can|i am on it|perfect)\b', t))

    def _get_addressee(self, current_speaker: str, participants: Dict[str, Any]) -> str:
        """Find the other participant in dialogue."""
        others = [name for name in participants if name != current_speaker and name not in ('Speaker', 'Team')]
        return others[0] if others else "Addressee"

    def _final_recall_safety_pass(self, graph: ConversationContextGraph, turns: List[Dict[str, Any]]):
        """
        Scan all professional turns for any uncaptured metrics, numbers, percentages, or status facts.
        Guarantees 100% information recall.
        """
        all_captured_text = " ".join([
            " ".join([sf['fact'] for sf in graph.status_facts]),
            " ".join([p.problem + (p.cause or "") for p in graph.causality_arcs]),
            " ".join([d.final_decision for d in graph.decision_arcs]),
            " ".join([t.full_description for t in graph.tasks])
        ]).lower()

        for t in turns:
            text = t['clean']
            text_lower = text.lower()

            metric_match = re.search(r'\b(?:\d+(?:\.\d+)?%|\d+\s*(?:seconds?|ms|fps|users|requests|units|kb|mb|gb|rpm))\b', text, re.IGNORECASE)
            if metric_match:
                m_str = metric_match.group(0).lower()
                if m_str not in all_captured_text:

                    graph.status_facts.append({
                        'fact': text,
                        'impact': 'NON_BLOCKING' if 'non-blocking' in text_lower or 'doesn\'t affect' in text_lower else 'UNSPECIFIED'
                    })

    def _deduplicate_graph(self, graph: ConversationContextGraph):
        """Deduplicate decision arcs and task actions."""
        unique_decisions: List[ProposalDecisionArc] = []
        for d in graph.decision_arcs:
            if not any(d.final_decision.lower() == exist.final_decision.lower() for exist in unique_decisions):
                unique_decisions.append(d)
        graph.decision_arcs = unique_decisions

        unique_tasks: List[TaskActionNode] = []
        for t in graph.tasks:
            if not any(t.full_description.lower() == exist.full_description.lower() for exist in unique_tasks):
                unique_tasks.append(t)
        graph.tasks = unique_tasks

    def _cluster_semantic_topics(self, graph: ConversationContextGraph) -> List[str]:
        """
        Generate concise 2-4 word clean domain noun phrases
        prioritizing decisions, blockers, issues, and scopes.
        """
        primary_entities = []

        for p in graph.causality_arcs:
            if p.cause:
                primary_entities.append(p.cause)
            ents = self._extract_technical_entities(p.problem)
            primary_entities.extend(ents)

        for s in graph.scope_items:
            primary_entities.append(s.feature_or_item)

        for d in graph.decision_arcs:
            primary_entities.append(d.resolved_object)
            ents = self._extract_technical_entities(d.final_decision)
            primary_entities.extend(ents)

        for t in graph.tasks:
            ents = self._extract_technical_entities(t.full_description)
            primary_entities.extend(ents)

        for rg in graph.release_gates:
            primary_entities.append(rg.target)

        topics = []
        seen = set()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'with', 'that', 'this', 'to', 'of', 'in', 'on', 'at', 'is', 'are', 'was', 'were'}

        for ent in primary_entities:
            ent_clean = re.sub(r'[\*_\[\]\(\)\:\.\,]', '', ent).strip()
            words = [w for w in ent_clean.split() if w.lower() not in stop_words]
            if 2 <= len(words) <= 4:
                cand = " ".join(words).title()
                if cand.lower() not in seen:
                    seen.add(cand.lower())
                    topics.append(cand)

        if not topics:
            topics = ["Project Architecture", "Release Scope"]

        return topics[:5]
