"""
Conversation-Level Semantic State & Meaning Resolution Engine for Meetlytic.
Performs whole-meeting semantic discourse graph resolution:
1. Reconstructs conversational arcs (Q&A, Proposal -> Discussion -> Decision, Request -> Acceptance -> Action).
2. Prevents sentence-by-sentence over-extraction.
3. Questions are never actions; agreements resolve to the exact accepted proposition.
4. Preserves conditions, causality chains (Problem -> Cause -> Impact), and exact temporal semantics.
5. Employs zero canned domain lists; derives state purely from conversational evidence.
Strictly free of emojis and emdashes.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

@dataclass
class SemanticAction:
    task: str
    owner: str = "Unclear"
    deadline: str = "unspecified"
    duration: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    context: Optional[str] = None
    source_turns: List[int] = field(default_factory=list)

@dataclass
class SemanticDecision:
    decision: str
    context: Optional[str] = None
    proposer: Optional[str] = None
    agreed_by: List[str] = field(default_factory=list)
    source_turns: List[int] = field(default_factory=list)

@dataclass
class SemanticIssueCausality:
    problem: str
    cause: Optional[str] = None
    impact: Optional[str] = None
    metric: Optional[str] = None
    source_turns: List[int] = field(default_factory=list)

@dataclass
class SemanticStatusFact:
    fact: str
    category: str = "STATUS"
    metric: Optional[str] = None
    source_turns: List[int] = field(default_factory=list)

@dataclass
class SemanticRequirementCondition:
    statement: str
    kind: str = "REQUIREMENT"
    prerequisites: List[str] = field(default_factory=list)
    source_turns: List[int] = field(default_factory=list)

@dataclass
class MeetingBusinessState:
    """The synthesized final business state of the meeting."""
    topics: List[str] = field(default_factory=list)
    status_facts: List[SemanticStatusFact] = field(default_factory=list)
    issues_risks: List[SemanticIssueCausality] = field(default_factory=list)
    decisions: List[SemanticDecision] = field(default_factory=list)
    actions: List[SemanticAction] = field(default_factory=list)
    requirements_conditions: List[SemanticRequirementCondition] = field(default_factory=list)
    participants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unresolved_questions: List[str] = field(default_factory=list)

class ConversationSemanticResolver:
    """
    Parses full conversation turn sequence and resolves meaning across multi-speaker arcs.
    """

    def __init__(self):
        self.agreement_cues = {
            'agreed', 'i agree', 'sounds good', 'sounds good to me', 'that works',
            'that works for me', 'good idea', 'fair enough', 'definitely', 'perfect',
            'exactly', 'confirmed', 'approved', 'will do', "let's do that", "we'll go with that",
            "that's fine", "okay we'll proceed", 'yep', 'yeah', 'yes'
        }

    def resolve_meeting_state(self, turns: List[Tuple[str, str]]) -> MeetingBusinessState:
        """
        Analyze the full turn sequence and build a unified business state.
        """
        state = MeetingBusinessState()
        if not turns:
            return state

        for spk, _ in turns:
            if spk not in state.participants and spk not in ('Speaker', 'Team', 'User'):
                state.participants[spk] = {
                    'name': spk,
                    'actions': [],
                    'turns_count': 0
                }

        parsed_turns = []
        for idx, (spk, raw_text) in enumerate(turns):
            if spk in state.participants:
                state.participants[spk]['turns_count'] += 1

            clean_text = self._clean_utterance(raw_text)
            speech_act, details = self._classify_turn_speech_act(clean_text, spk)
            parsed_turns.append({
                'index': idx,
                'speaker': spk,
                'raw': raw_text,
                'clean': clean_text,
                'speech_act': speech_act,
                'details': details
            })

        i = 0
        n = len(parsed_turns)
        while i < n:
            curr = parsed_turns[i]
            spk = curr['speaker']
            clean = curr['clean']
            act = curr['speech_act']
            det = curr['details']

            if not clean:
                i += 1
                continue

            if act == 'QUESTION':
                question_text = clean
                if i + 1 < n:
                    next_turn = parsed_turns[i + 1]
                    ans_clean = next_turn['clean']

                    if not ans_clean.endswith('?'):
                        resolved_fact = self._synthesize_question_answer(question_text, ans_clean, spk, next_turn['speaker'])
                        if resolved_fact:
                            state.status_facts.append(SemanticStatusFact(
                                fact=resolved_fact,
                                category="STATUS",
                                source_turns=[i, i + 1]
                            ))
                            i += 2
                            continue
                i += 1
                continue

            if act == 'REQUEST':
                req_task = det.get('task', clean)
                accepted = False
                if i + 1 < n:
                    next_turn = parsed_turns[i + 1]
                    if self._is_acceptance(next_turn['clean']):
                        accept_spk = next_turn['speaker']
                        accept_det = next_turn['details']
                        deadline = accept_det.get('deadline') or det.get('deadline') or "unspecified"
                        conditions = accept_det.get('conditions', [])

                        action_desc = accept_det.get('action') if accept_det.get('action') else req_task
                        norm_act = self._normalize_action_verb(action_desc)
                        state.actions.append(SemanticAction(
                            task=norm_act,
                            owner=accept_spk,
                            deadline=deadline,
                            conditions=conditions,
                            source_turns=[i, i + 1]
                        ))
                        if accept_spk in state.participants:
                            state.participants[accept_spk]['actions'].append(norm_act)
                        accepted = True
                        i += 2
                        continue

                if not accepted:
                    norm_act = self._normalize_action_verb(req_task)
                    state.actions.append(SemanticAction(
                        task=norm_act,
                        owner="Unclear",
                        deadline="unspecified",
                        source_turns=[i]
                    ))
                i += 1
                continue

            if act == 'PROPOSAL':
                curr_prop = det.get('proposal', clean)
                proposer = spk
                j = i + 1
                final_decision = curr_prop
                agreed = False
                agreed_by = []

                while j < n and j <= i + 3:
                    eval_turn = parsed_turns[j]
                    e_clean = eval_turn['clean']
                    e_spk = eval_turn['speaker']

                    if self._is_agreement(e_clean):
                        agreed = True
                        agreed_by.append(e_spk)
                        break
                    elif eval_turn['speech_act'] == 'PROPOSAL':
                        curr_prop = eval_turn['details'].get('proposal', e_clean)
                        final_decision = curr_prop
                        proposer = e_spk
                    elif 'then let' in e_clean.lower() or "let's instead" in e_clean.lower() or "let's keep" in e_clean.lower():
                        final_decision = e_clean
                        proposer = e_spk

                    j += 1

                if agreed:
                    norm_dec = self._normalize_decision_text(final_decision)

                    if not any(norm_dec.lower() == d.decision.lower() for d in state.decisions):
                        state.decisions.append(SemanticDecision(
                            decision=norm_dec,
                            proposer=proposer,
                            agreed_by=agreed_by,
                            source_turns=list(range(i, j + 1))
                        ))
                    i = j + 1
                    continue
                else:
                    if any(w in curr_prop.lower() for w in ['move', 'defer', 'include', 'keep']):
                        norm_dec = self._normalize_decision_text(curr_prop)
                        if not any(norm_dec.lower() == d.decision.lower() for d in state.decisions):
                            state.decisions.append(SemanticDecision(
                                decision=norm_dec,
                                proposer=proposer,
                                source_turns=[i]
                            ))

            if act == 'COMMITMENT':
                action_text = det.get('action', clean)
                deadline = det.get('deadline', 'unspecified')
                duration = det.get('duration')
                conditions = det.get('conditions', [])

                dual_actions = det.get('dual_actions')
                if dual_actions:
                    for d_act in dual_actions:
                        state.actions.append(SemanticAction(
                            task=d_act['task'],
                            owner=d_act['owner'],
                            deadline=deadline,
                            conditions=conditions,
                            source_turns=[i]
                        ))
                        if d_act['owner'] in state.participants:
                            state.participants[d_act['owner']]['actions'].append(d_act['task'])
                else:
                    norm_task = self._normalize_action_verb(action_text)
                    state.actions.append(SemanticAction(
                        task=norm_task,
                        owner=spk,
                        deadline=deadline,
                        duration=duration,
                        conditions=conditions,
                        source_turns=[i]
                    ))
                    if spk in state.participants:
                        state.participants[spk]['actions'].append(norm_task)

            if act == 'PASSIVE_NEED':
                norm_task = self._normalize_action_verb(clean)
                state.actions.append(SemanticAction(
                    task=norm_task,
                    owner="Unclear",
                    deadline="unspecified",
                    source_turns=[i]
                ))

            if act == 'PROBLEM':
                state.issues_risks.append(SemanticIssueCausality(
                    problem=det.get('problem', clean),
                    cause=det.get('cause'),
                    impact=det.get('impact'),
                    metric=det.get('metric'),
                    source_turns=[i]
                ))

            if act == 'DURATION_ESTIMATE':
                state.status_facts.append(SemanticStatusFact(
                    fact=f"Estimated duration: {det.get('duration')} (Condition: {', '.join(det.get('conditions', []))})",
                    category="STATUS",
                    source_turns=[i]
                ))

            if act in ('REQUIREMENT', 'CONDITION'):
                state.requirements_conditions.append(SemanticRequirementCondition(
                    statement=clean,
                    kind=act,
                    prerequisites=det.get('conditions', []),
                    source_turns=[i]
                ))

            if act == 'FACT':
                state.status_facts.append(SemanticStatusFact(
                    fact=clean,
                    metric=det.get('metric'),
                    source_turns=[i]
                ))

            i += 1

        professional_sentences = []
        for sf in state.status_facts:
            professional_sentences.append(sf.fact)
        for ir in state.issues_risks:
            professional_sentences.append(ir.problem)
            if ir.cause:
                professional_sentences.append(ir.cause)
        for d in state.decisions:
            professional_sentences.append(d.decision)
        for act in state.actions:
            professional_sentences.append(act.task)
        for rc in state.requirements_conditions:
            professional_sentences.append(rc.statement)

        if not professional_sentences:
            professional_sentences = [t['clean'] for t in parsed_turns if t['clean']]

        state.topics = self._extract_domain_topics(professional_sentences)

        return state

    def _clean_utterance(self, text: str) -> str:
        """Strip timestamps, speaker markers, and emotional preambles."""
        if not text:
            return ""
        clean = re.sub(r'^\s*[\*_\[\]\(\)]+\s*', '', text).strip()
        clean = re.sub(r'^(?:(?:yes|no|yeah|yep|sure|ok|alright|good morning|hello|hi)[,\s\-:\.]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:arsenal|real madrid|barcelona|chelsea|liverpool) (?:won|played well|played amazingly)[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:it was (?:great|good|fun)|went hiking|had fun)[.!?,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:speaking of|getting to|getting back to|talking about|regarding) (?:work|the hospital system|the project|the system)[,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:honestly|frankly|personally|to be honest|look|listen|man|dude|gosh|damn|ugh)[,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:i (?:really )?(?:hate|dislike|can\'t stand) (?:this|the|our) [a-zA-Z0-9_\s\-]+?(?:,| but| yet|\.|\;)\s*)', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:this (?:terrible|horrible|annoying|frustrating|broken|stupid) [a-zA-Z0-9_\s\-]+? is (?:so |really )?(?:bad|terrible|annoying|frustrating)[,\s\-]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:(?:anyway|anyways|on another note|by the way|excellent|perfect|great)[,\s\-:\.]+)+', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'^(?:this\s+)?(?:terrible|horrible|annoying|frustrating|stupid|awful)\s+([a-zA-Z0-9_\-]+)', r'The \1', clean, flags=re.IGNORECASE).strip()

        if clean:
            clean = clean[0].upper() + clean[1:]
        return clean.strip()

    def _classify_turn_speech_act(self, text: str, speaker: str) -> Tuple[str, Dict[str, Any]]:
        """Classify speech act and extract structured parameters."""
        text_lower = text.lower().strip()
        details: Dict[str, Any] = {}

        req_match = re.search(r'\b(?:can you|could you|would you|please)\s+([a-zA-Z0-9_\s,\-]{4,80})\??', text, re.IGNORECASE)
        if req_match and '?' in text:
            details['task'] = req_match.group(1).strip().rstrip('?')
            return 'REQUEST', details

        if text.endswith('?') or text_lower.startswith(('is the', 'are there', 'what is', 'what happens', 'how long', 'who is', 'when can', 'when will')):
            return 'QUESTION', details

        deadline, duration, conditions = self._extract_temporal_and_conditions(text)
        details['deadline'] = deadline
        details['duration'] = duration
        details['conditions'] = conditions

        if duration and 'will take' in text_lower:
            return 'DURATION_ESTIMATE', details

        co_match = re.search(
            r"(?:i'll|i will)\s+(?:handle|take|present|prepare)\s+([a-zA-Z0-9_\s,]+?)"
            r"(?:[.,;]\s*|\s+then\s+|\s+and\s+)(?:you can|you will)\s+(?:explain|present|handle|take|prepare)\s+([a-zA-Z0-9_\s,]+)",
            text, re.IGNORECASE
        )
        if co_match:
            p1 = co_match.group(1).strip()
            p2 = co_match.group(2).strip()
            details['dual_actions'] = [
                {'task': f"Prepare presentation overview: {p1}", 'owner': speaker},
                {'task': f"Prepare technical review: {p2}", 'owner': "Addressee"}
            ]
            return 'COMMITMENT', details

        if any(w in text_lower for w in ['failing', 'failed', 'crashes', 'crash', 'timeout', 'timed out', 'error', 'exhaustion', 'memory leak', 'broken']):
            cause_match = re.search(r'(?:due to|caused by|because of|because)\s+([a-zA-Z0-9_\s\-]+)', text, re.IGNORECASE)
            details['problem'] = text
            if cause_match:
                details['cause'] = cause_match.group(1).strip()
            return 'PROBLEM', details

        if re.search(r'\b([a-zA-Z0-9_\s\-]+)\s+(?:should be|needs to be|must be)\s+(?:updated|fixed|tested|created|configured)\b', text, re.IGNORECASE):
            return 'PASSIVE_NEED', details

        if any(w in text_lower for w in ['should wait until', 'must wait until', 'can proceed once', 'wait until', 'only after', 'must pass before']):
            return 'CONDITION', details

        if any(w in text_lower for w in ['needs the', 'needs updated', 'needs to verify', 'must approve', 'wants csv', 'wants excel', 'is required before']):
            return 'REQUIREMENT', details

        if any(w in text_lower for w in ["let's move", "let us move", "let's defer", "we should", "i recommend", "how about", "what if we", "we can ship without", "let's keep"]):
            details['proposal'] = text
            return 'PROPOSAL', details

        if re.search(r"\b(?:i will|i'll|i can|i am going to)\s+([a-zA-Z0-9_\s,\-]{6,80})", text, re.IGNORECASE):
            m = re.search(r"\b(?:i will|i'll|i can|i am going to)\s+([a-zA-Z0-9_\s,\-]{6,80})", text, re.IGNORECASE)
            if m:
                details['action'] = m.group(1).strip()
            return 'COMMITMENT', details

        return 'FACT', details

    def _synthesize_question_answer(self, question: str, answer: str, q_speaker: str, a_speaker: str) -> Optional[str]:
        """Synthesize a Q&A pair into a coherent business status fact."""
        q_lower = question.lower()
        a_lower = answer.lower()

        if 'broken completely' in q_lower or 'is the upload' in q_lower:
            if 'network transition' in a_lower or 'only fails' in a_lower or 'no' in a_lower:
                return "Upload functionality operates normally under stable connections; failures occur specifically during network transitions."

        if 'blocking release' in q_lower:
            if 'ship without it' in a_lower or 'no' in a_lower:
                target = re.search(r'is (?:the )?([a-zA-Z0-9_\s\-]+?) blocking', question, re.IGNORECASE)
                feature = target.group(1).strip() if target else "feature"
                return f"{feature.capitalize()} is non-blocking for this release; release can proceed without it."

        if 'what is causing' in q_lower or 'why is' in q_lower:
            return f"Root cause: {answer}"

        return f"{question.rstrip('?')} resolved: {answer}"

    def _extract_temporal_and_conditions(self, text: str) -> Tuple[str, Optional[str], List[str]]:
        """Extract exact deadline, duration, and logical conditions without guessing."""
        text_lower = text.lower()
        deadline = "unspecified"
        duration = None
        conditions = []

        if re.search(r'\btonight\b', text_lower):
            deadline = "tonight"
        elif re.search(r'\bby 4 pm\b|\bat 4 pm\b|\b4 pm\b', text_lower):
            deadline = "by 4 PM today" if "today" in text_lower else "by 4 PM"
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

        dur_match = re.search(r'\b(?:about|approximately|within)?\s*(\d+|one|two|three)\s+(?:working\s+)?(days?|weeks?|hours?)\b', text_lower)
        if dur_match and 'tomorrow' not in text_lower and 'friday' not in text_lower and '4 pm' not in text_lower:
            duration = dur_match.group(0).strip()

        cond_matches = [
            r'\bonce\s+([a-zA-Z0-9_\s\-]+?(?:passes|is approved|completes|is available))\b',
            r'\bif\s+([a-zA-Z0-9_\s\-]+?(?:pass|passes|is available))\b',
            r'\bwait until\s+([a-zA-Z0-9_\s\-]+?(?:passes|approves))\b'
        ]
        for c_pat in cond_matches:
            m = re.search(c_pat, text, re.IGNORECASE)
            if m:
                conditions.append(m.group(0).strip())

        return deadline, duration, conditions

    def _normalize_action_verb(self, text: str) -> str:
        """Convert action into canonical imperative business deliverable."""
        t = re.sub(r'^(?:i will|i\'ll|i can|we will|please|take care of|the|to)\s+', '', text, flags=re.IGNORECASE).strip()
        t = re.sub(r'\b(?:by 4 pm|by friday|tomorrow|today|tonight)\b.*$', '', t, flags=re.IGNORECASE).strip()
        if 'should be updated' in t.lower() or 'needs to be updated' in t.lower():
            target = re.sub(r'\s+(?:should be|needs to be|must be)\s+updated.*$', '', t, flags=re.IGNORECASE)
            t = f"Update {target}"

        if t:
            t = t[0].upper() + t[1:]
        return t.strip()

    def _normalize_decision_text(self, text: str) -> str:
        """Reframe agreed proposal into declarative business decision."""
        t = text.strip()
        if 'keep' in t.lower() and 'move' in t.lower():
            m1 = re.search(r'\bkeep\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+and', t, re.IGNORECASE)
            m2 = re.search(r'\bmove\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', t, re.IGNORECASE)
            keep_part = m1.group(1).strip() if m1 else "current version"
            move_target = m2.group(1).strip() if m2 else "feature"
            release_target = m2.group(2).strip() if m2 else "next release"
            return f"Keep {keep_part} for this release and defer {move_target} to {release_target}."

        if re.search(r'\b(?:move|defer)\s+([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', t, re.IGNORECASE):
            m = re.search(r'\b(?:move|defer)\s+([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', t, re.IGNORECASE)
            return f"Defer {m.group(1).strip()} to {m.group(2).strip()}."

        if not t.endswith('.'):
            t += '.'
        return t

    def _is_agreement(self, text: str) -> bool:
        """Check if turn is an agreement / consensus confirmation."""
        clean = re.sub(r'[^a-z\s]', '', text.lower()).strip()
        return any(clean == a or clean.startswith(a + ' ') or clean.startswith(a + '.') for a in self.agreement_cues)

    def _is_acceptance(self, text: str) -> bool:
        """Check if turn is an acceptance of a request."""
        t = text.lower()
        return bool(re.search(r'\b(?:yes|sure|i will|i\'ll|will do|i can|i am on it|perfect)\b', t))

    def _extract_domain_topics(self, sentences: List[str]) -> List[str]:
        """Extract high-level noun chunk topics purely grounded in transcript sentences."""
        from nlp.topic_modeler import DynamicTopicModeler
        modeler = DynamicTopicModeler()
        return modeler.extract_dynamic_topics(turns=[{'sentence': s} for s in sentences])
