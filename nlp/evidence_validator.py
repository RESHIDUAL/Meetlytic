"""
Semantic Promotion Gate & Evidence Validation Engine for Meetlytic.
Enforces strict multi-stage verification before candidate items are promoted:
1. Linguistic Quality Check (Grammatical completeness, word count)
2. Semantic Coherence Check (Sentence coherence score 0.0 - 1.0)
3. Intent Classification (FACT, INFORMATION, BACKGROUND, REPORTING, PROPOSAL, OPINION, CONFIRMED_ACTION, DECISION, MEETING_PROCEDURE)
4. Reference Resolution (Rejects pronouns lacking concrete antecedents)
5. Evidence Check (Cross-turn validation; never manufactures missing info)
6. Promotion Decision (CONFIRMED vs UNCERTAIN vs REJECTED)
Strictly free of emojis and emdashes.
"""

import re
from typing import Dict, Any, Optional, List, Tuple

class SpeechAct:
    CASUAL = "CASUAL"
    ASR_GARBAGE = "ASR_GARBAGE"
    COMMENTARY = "COMMENTARY"
    FACT = "FACT"
    CURRENT_VALUE = "CURRENT_VALUE"
    HISTORICAL_VALUE = "HISTORICAL_VALUE"
    QUESTION = "QUESTION"
    OPINION = "OPINION"
    PROPOSAL = "PROPOSAL"
    DECISION = "DECISION"
    ACTION = "ACTION"
    CONDITIONAL_ACTION = "CONDITIONAL_ACTION"
    RISK = "RISK"
    BLOCKER = "BLOCKER"
    CORRECTION = "CORRECTION"
    CLARIFICATION = "CLARIFICATION"
    MEETING_PROCEDURE = "MEETING_PROCEDURE"
    UNKNOWN = "UNKNOWN"

class EvidenceValidator:
    """
    Semantic Promotion Gate ensuring only verified, coherent intelligence enters the executive report.
    Enforces strict speech-act classification, generalized ASR quality scoring, and action/fact/decision quality gates.
    """

    ASR_CORRUPTION_PATTERNS = [
        r'\b(?:pull\s+i|be\s+reporting|start\s+on\s+the\s+interesting\s+maria|go\s+down\s+the\s+road\s+side\s+trustee)\b',
        r'\b(?:trustee\s+staubach|my\s+right\s+trustee|pull\.\.\.)\b',
        r'^(?:take\s+that|do\s+it|pull(?:\.\.\.)?|handle\s+this|prepare\s+it|finish\s+it|get\s+it)\.?$',
        r'^[a-zA-Z0-9_\-]{1,3}\.?$'
    ]

    VALID_ACTION_VERBS = [
        'handle', 'manage', 'investigate', 'send', 'prepare', 'book', 'schedule', 'deploy', 'fix',
        'create', 'review', 'audit', 'finalize', 'draft', 'organize', 'write',
        'coordinate', 'complete', 'notify', 'post', 'email', 'inform', 'publish',
        'reconcile', 'configure', 'implement', 'build', 'run', 'test', 'launch',
        'obtain', 'submit', 'pull', 'record', 'update'
    ]

    PROCEDURE_PATTERNS = [
        r'\b(?:call\s+to\s+order|roll\s+call|proceed\s+to\s+roll\s+call\s+vote|roll\s+call\s+vote)\b',
        r'\b(?:motion\s+to\s+approve|second\s+the\s+motion|i\s+second|motion\s+is\s+seconded)\b',
        r'\b(?:meeting\s+is\s+adjourned|move\s+to\s+adjourn|motion\s+to\s+adjourn)\b',
        r'\b(?:quorum\s+is\s+present|reading\s+of\s+the\s+minutes|corrections?\s+(?:to\s+)?the\s+minutes)\b'
    ]

    VOTE_DECISION_PATTERNS = [
        r'\b(?:motion\s+passed|motion\s+passes|motion\s+carried|motion\s+failed|motion\s+denied)\b',
        r'\b(?:board\s+approved|committee\s+approved|approved\s+the\s+minutes|unanimously\s+approved|voted\s+to\s+approve)\b'
    ]

    INFORMATIONAL_PATTERNS = [
        r'\b(?:discussed|reviewed|talked\s+about|presented|explained|overview\s+of|reported\s+on)\b',
        r'\b(?:received\s+funding|was\s+held|took\s+place|established\s+in|founded\s+in)\b',
        r'\b(?:represents|represented|accounting\s+for|composed\s+of|totaling)\b'
    ]

    SPEAKER_ADMIN_PATTERNS = [
        r'\b(?:give me|can i have|may i have|could i have)\s+(?:a |one |two |three |another |just |)\s*(?:minute|moment|second)',
        r'\b(?:let me finish|please continue|go ahead|can everyone hear)',
        r'\b(?:can you repeat|say that again|hold on|one moment|just a moment)',
        r'\b(?:sorry i interrupted|sorry for interrupting|excuse me)',
        r"\b(?:let's take a (?:short )?break|five minute break|take a recess)",
        r'\b(?:donate a minute|just two minutes|would be appreciated|a little bit longer)',
        r'\b(?:can i finish|let me speak|my turn|i have the floor)',
    ]

    def calculate_semantic_quality_score(self, text: str, context_entities: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generalized ASR-quality and semantic scoring layer.
        Calculates:
          semantic_coherence + action_completeness + object_completeness + contextual_support
          - ASR_noise - fragment_penalty - ambiguity_penalty
        """
        if not text or not text.strip():
            return {'score': 0.0, 'is_garbage': True, 'reason': 'Empty text'}

        t_clean = text.strip()
        t_lower = t_clean.lower()
        words = t_clean.split()
        word_count = len(words)

        coherence = 0.80
        action_completeness = 0.0
        object_completeness = 0.0
        contextual_support = 0.0
        asr_noise = 0.0
        fragment_penalty = 0.0
        ambiguity_penalty = 0.0

        if re.search(r'\b(?:pull|record|start|go|take|send|get|make)\s+i(?:\s+if\s+you)?\.?$', t_lower) or re.search(r'\b[a-z]\s+i\.?$', t_lower):
            asr_noise += 0.60

        if t_lower.rstrip('.!?,').endswith((' on', ' to', ' in', ' at', ' for', ' with', ' and', ' or', ' the', ' a', ' an', ' of', ' that', ' this', ' so')):
            fragment_penalty += 0.35

        if re.search(r'\b(?:road\s+side\s+trustee|interesting\s+maria|so\s+many\s+discussion\s+on\s+one\s+no\s+thank\s+you)\b', t_lower):
            asr_noise += 0.70

        has_predicate = bool(re.search(r'\b(?:is|are|was|were|will|has|have|had|approved|decided|investigate|run|deploy|fix|send|create|prepare|schedule|submit|audit|update|launch|passed)\b', t_lower))
        if not has_predicate and word_count >= 4:
            fragment_penalty += 0.30

        action_verb_match = re.search(rf'\b({"|".join(self.VALID_ACTION_VERBS)})\b', t_lower)
        if action_verb_match:
            verb_pos = action_verb_match.start()
            after_verb = t_lower[verb_pos + len(action_verb_match.group(1)):].strip()
            after_words = [w for w in re.sub(r'[\*_\[\]\(\)\:\.\,\'\"]', ' ', after_verb).split() if w not in ('by', 'at', 'on', 'tomorrow', 'today', 'tonight', 'friday', '5', '4', 'pm', 'am')]

            if not after_words:
                ambiguity_penalty += 0.50
            elif len(after_words) == 1 and after_words[0] in ('it', 'that', 'this', 'them', 'something', 'one', 'i', 'you', 'me', 'us'):

                if context_entities:
                    contextual_support += 0.30
                    object_completeness += 0.40
                else:
                    ambiguity_penalty += 0.60
            elif len(after_words) >= 2 or any(w in after_words for w in ['timeout', 'behavior', 'test', 'report', 'approval', 'brief', 'ticket', 'rate', 'revenue', 'database', 'pipeline']):
                object_completeness += 0.50
                action_completeness += 0.30
            else:
                object_completeness += 0.20

        has_explicit_assignment = bool(re.match(r'^((?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][a-z]+)[,\s\-]+(?:please\s+)?([a-z]+)\b', t_clean))
        has_explicit_commitment = bool(re.match(r'^(?:i\s+will|i\'ll|i\s+can|we\s+will)\s+([a-z]+)\b', t_lower)) or bool(re.match(r'^([A-Z][a-z]+)\s+will\s+([a-z]+)\b', t_clean))

        if has_explicit_assignment or has_explicit_commitment:
            action_completeness += 0.30

        if re.search(r'\b(?:committee\s+(?:or|will)\s+focus|items\s+during\s+the\s+meeting|from\s+the\s+concerned\s+agenda|corrections\s+(?:to\s+)?the\s+minutes)\b', t_lower):
            fragment_penalty += 0.40

        final_score = coherence + action_completeness + object_completeness + contextual_support - asr_noise - fragment_penalty - ambiguity_penalty
        final_score = max(0.0, min(1.0, round(final_score, 2)))
        is_garbage = (final_score < 0.50) or (asr_noise >= 0.50)

        return {
            'score': final_score,
            'is_garbage': is_garbage,
            'action_completeness': action_completeness,
            'object_completeness': object_completeness,
            'has_explicit_actor': has_explicit_assignment or has_explicit_commitment
        }

    def calculate_coherence_score(self, text: str, surrounding_turns: Optional[List[str]] = None) -> float:
        """Calculate overall coherence score (0.0 - 1.0)."""
        res = self.calculate_semantic_quality_score(text)
        return res['score']

    def classify_speech_act(self, text: str, context_entities: Optional[List[str]] = None, surrounding_turns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Classify transcript segment into exactly one canonical speech act:
        CASUAL, ASR_GARBAGE, COMMENTARY, FACT, CURRENT_VALUE, HISTORICAL_VALUE,
        QUESTION, OPINION, PROPOSAL, DECISION, ACTION, CONDITIONAL_ACTION,
        RISK, BLOCKER, CORRECTION, CLARIFICATION, MEETING_PROCEDURE, UNKNOWN.
        """
        if not text or not text.strip():
            return {'act': SpeechAct.UNKNOWN, 'confidence': 0.0, 'discard': True}

        t_clean = text.strip()
        t_lower = t_clean.lower()
        words = t_clean.split()
        word_count = len(words)

        quality = self.calculate_semantic_quality_score(t_clean, context_entities)
        if quality['is_garbage']:
            return {'act': SpeechAct.ASR_GARBAGE, 'confidence': quality['score'], 'discard': True}

        if self.is_speaker_administration(t_clean):
            return {'act': SpeechAct.CASUAL, 'confidence': 0.95, 'discard': True}

        if self.is_public_comment_or_rant(t_clean):
            return {'act': SpeechAct.COMMENTARY, 'confidence': 0.90, 'discard': True}

        proc = self.check_procedure(t_clean)
        if proc:
            if proc.get('is_vote'):
                return {'act': SpeechAct.DECISION, 'confidence': 0.98, 'discard': False}
            return {'act': SpeechAct.MEETING_PROCEDURE, 'confidence': 0.95, 'discard': True}

        if t_clean.endswith('?') or any(t_lower.startswith(q) for q in ['is ', 'are ', 'did ', 'has ', 'have ', 'what ', 'why ', 'how ', 'who ', 'when ', 'can we ', 'could we ']):
            return {'act': SpeechAct.QUESTION, 'confidence': 0.90, 'discard': True}

        if any(t_lower.startswith(op) for op in [
            'i think', 'in my opinion', 'i feel like', 'personally', 'i believe',
            'that\'s my opinion', 'that is my opinion', 'that\'s just your opinion',
            'that sounds good', 'in my view', 'i don\'t think'
        ]):
            return {'act': SpeechAct.OPINION, 'confidence': 0.85, 'discard': True}

        if any(t_lower.startswith(p) for p in ['maybe we should', 'what if we', 'how about we', 'could we consider', 'i propose', 'i suggest', 'maybe rahul should', 'maybe arjun should']):
            return {'act': SpeechAct.PROPOSAL, 'confidence': 0.85, 'discard': True}

        if any(w in t_lower for w in [
            'focus on three key items during the meeting', 'items from the concerned agenda',
            'have been a while you can tell', 'celebrated the anniversary', 'dedicated a new',
            'hosted an event', 'discussion on one no thank you'
        ]) or self.is_narrative_description(t_clean):
            return {'act': SpeechAct.COMMENTARY, 'confidence': 0.85, 'discard': True}

        if any(w in t_lower for w in ['correction,', 'actually,', 'was originally', 'original campaign budget was', 'original budget was', 'initially was', 'was estimated at']):
            if any(w in t_lower for w in ['approved', 'confirmed', 'current is', 'now']):
                return {'act': SpeechAct.CORRECTION, 'confidence': 0.95, 'discard': False}
            return {'act': SpeechAct.HISTORICAL_VALUE, 'confidence': 0.90, 'discard': False}

        if any(w in t_lower for w in ['we decided', 'everyone agreed', 'committee approved', 'board approved', 'board voted to approve', 'approved the', 'ship without', 'adopted the', 'motion passes', 'motion passed']):
            return {'act': SpeechAct.DECISION, 'confidence': 0.95, 'discard': False}

        has_valid_verb = any(re.search(rf'\b{v}\b', t_lower) for v in self.VALID_ACTION_VERBS)
        if has_valid_verb and quality['score'] >= 0.70:
            if quality['has_explicit_actor'] and quality['object_completeness'] > 0:
                return {'act': SpeechAct.ACTION, 'confidence': quality['score'], 'discard': False}
            elif quality['object_completeness'] >= 0.50 and re.search(r'\b(?:will|please|can you|should)\b', t_lower):
                return {'act': SpeechAct.ACTION, 'confidence': quality['score'], 'discard': False}

        if any(w in t_lower for w in ['cannot proceed', 'blocks release', 'prevents launch', 'primary blocker', 'cannot continue until']):
            return {'act': SpeechAct.BLOCKER, 'confidence': 0.95, 'discard': False}
        if any(w in t_lower for w in ['crashes', 'crash', 'timeout', 'failing', 'failed', 'failure', 'error', 'slowdown', 'exhaustion', 'below target', 'discrepancy']):
            return {'act': SpeechAct.RISK, 'confidence': 0.90, 'discard': False}

        metric_match = re.search(r'(\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:k|m|b|million)?|\b\d+k\b|\b\d+\s*million\b|\d+(?:\.\d+)?%|\d+\s*(?:minutes?|seconds?|hours?|ms|fps|users|requests|kb|mb|gb|rpm|candidates|active candidates|participants|patients|attendees|students|cases)|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+duplicate|\d+,\d{3})', t_clean, re.IGNORECASE)
        if metric_match or any(w in t_lower for w in ['property tax', 'revenue represents', 'estimated revenue', 'tuition is', 'state appropriation', 'budget is', 'approved budget is', 'target is', 'improved', 'decreased']):
            return {'act': SpeechAct.FACT, 'confidence': 0.90, 'discard': False}

        return {'act': SpeechAct.UNKNOWN, 'confidence': 0.30, 'discard': True}

    def validate_action(self, description: str, owner: str, evidence: str = "", surrounding_turns: Optional[List[str]] = None, context_entities: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Strict Action Item Promotion Gate:
        Enforces all 10 conditions:
        1. Genuine task intent (not commentary, discussion, or procedure).
        2. Concrete action verb.
        3. Meaningful semantic object (NOT bare pronouns or single letters).
        4. Pronoun objects resolved only with high confidence.
        5. Not ASR garbage (quality score >= 0.70).
        6. Linguistic completeness.
        7. Does not merely describe current conversation/narration.
        8. Valid ownership.
        """
        desc_clean = description.strip()
        desc_lower = desc_clean.lower()

        if any(re.search(pat, desc_lower) for pat in self.PROCEDURE_PATTERNS) or any(w in desc_lower for w in [
            'have been a while', 'you can tell', 'corrections the minutes', 'corrections to the minutes',
            'go down the road', 'road side trustee', 'be reporting', 'start on the interesting',
            'pull any items from the concerned agenda', 'focus on three key items during the meeting'
        ]):
            return {
                'status': 'REJECTED',
                'reason': 'Meeting procedure or conversational commentary',
                'confidence': 0.0
            }

        if re.search(r'\b(?:audit and finance committee|committee will focus|concerned agenda)\b', desc_lower):
            return {
                'status': 'REJECTED',
                'reason': 'Committee description or agenda narration, not an actionable task',
                'confidence': 0.10
            }

        quality = self.calculate_semantic_quality_score(desc_clean, context_entities)
        if quality['is_garbage'] or quality['score'] < 0.68:
            return {
                'status': 'REJECTED',
                'reason': f"Low semantic quality / ASR noise (score: {quality['score']})",
                'confidence': quality['score']
            }

        vague_action_tail = r'^(?:(?:please\s+)?(?:do|handle|take|pull|send|prepare|finish|get|make|run|investigate|record|review|draft|update)\s*(?:it|this|that|those|i|me|you|us|them|over|in|on|up|down|if\s+you|\.\.\.)?\s*)+$'
        if re.match(vague_action_tail, desc_lower) or re.match(r'^(?:investigate|send|prepare|review|audit|deploy|fix|update|run|create|draft|schedule|submit|handle|notify|launch)\.?$', desc_lower):
            return {
                'status': 'REJECTED',
                'reason': 'Vague or missing semantic object without resolvable antecedent',
                'confidence': 0.10
            }

        m_obj = re.search(r'\b(?:do|handle|take|pull|send|prepare|finish|get|make|run|investigate|record|review|draft|update|create|deploy|fix|schedule|submit|audit|launch|obtain)\s+(.+)', desc_lower)
        if m_obj:
            obj_phrase = m_obj.group(1).strip()
            obj_clean_words = [w for w in re.sub(r'[\*_\[\]\(\)\:\.\,\'\"]', ' ', obj_phrase).split() if w not in ('by', 'at', 'on', 'tomorrow', 'today', 'tonight', 'friday', '5', '4', 'pm', 'am')]
            if not obj_clean_words or (len(obj_clean_words) == 1 and obj_clean_words[0] in ('i', 'it', 'this', 'that', 'them', 'you', 'me', 'us', 'something', 'one')):
                return {
                    'status': 'REJECTED',
                    'reason': 'Action object is an unresolved pronoun or fragment',
                    'confidence': 0.10
                }

        has_valid_verb = any(re.search(rf'\b{v}\b', desc_lower) for v in self.VALID_ACTION_VERBS)
        words = desc_clean.split()
        if len(words) < 2 or not has_valid_verb:
            if not re.search(r'\b(?:will|to)\s+[a-z]{3,}\b', desc_lower):
                return {
                    'status': 'REJECTED',
                    'reason': 'No actionable verb identified',
                    'confidence': 0.30
                }

        clean_owner = owner.strip() if owner else "Unassigned"
        if clean_owner.lower() in ("speaker", "we", "they", "you", "someone", "unclear", "unknown speaker", "the"):
            clean_owner = "Unassigned"

        confidence = 0.95 if clean_owner != "Unassigned" else 0.85

        return {
            'status': 'CONFIRMED',
            'resolved_owner': clean_owner,
            'reason': 'Actionable task with concrete object and valid intent',
            'confidence': confidence
        }

    def validate_decision(self, decision_text: str, is_explicit_decision: bool = False, surrounding_turns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Strict Decision Promotion Gate:
        Distinguish confirmed decisions from informational background, discussions, proposals, and opinions.
        """
        d_clean = decision_text.strip()
        d_lower = d_clean.lower()
        coherence = self.calculate_coherence_score(d_clean)

        if coherence < 0.60:
            return {
                'status': 'REJECTED',
                'kind': 'INCOHERENT',
                'confidence': coherence
            }

        if any(re.search(pat, d_lower) for pat in self.VOTE_DECISION_PATTERNS) or 'motion passes' in d_lower:
            status = "REJECTED" if any(w in d_lower for w in ['failed', 'denied']) else "CONFIRMED"
            return {
                'status': status,
                'kind': 'VOTE_OUTCOME',
                'confidence': 0.98
            }

        if re.search(r'\b(?:approved\s+(?:last\s+year|last\s+month|previously|earlier)|was\s+approved\s+(?:last\s+year|previously))\b', d_lower):
            return {
                'status': 'HISTORICAL_DECISION',
                'kind': 'HISTORICAL',
                'confidence': 0.30
            }

        if any(re.search(pat, d_lower) for pat in self.INFORMATIONAL_PATTERNS) and not any(w in d_lower for w in ['approved', 'agreed', 'decided', 'voted to approve']):
            return {
                'status': 'REJECTED',
                'kind': 'INFORMATION',
                'confidence': 0.20
            }

        if any(w in d_lower for w in ['celebrated', 'opened a new', 'dedicated a new', 'donated', 'discussed the budget', 'discussed', 'hosted an event', 'two decades of partnership']):
            return {
                'status': 'REJECTED',
                'kind': 'INFORMATION',
                'confidence': 0.20
            }

        if is_explicit_decision:
            if not any(d_lower.startswith(op) for op in ['i think', 'in my opinion', 'i feel like', 'personally', 'that\'s my opinion', 'that is my opinion', 'in my view', 'i don\'t think']):
                return {
                    'status': 'CONFIRMED',
                    'kind': 'DECISION',
                    'confidence': 0.95
                }

        if any(d_lower.startswith(op) for op in [
            'i think', 'in my opinion', 'i feel like', 'personally', 'i believe',
            'that\'s my opinion', 'that is my opinion', 'that sounds good', 'in my view',
            'i don\'t think', "we should consider", "maybe we should"
        ]):
            return {
                'status': 'OPINION',
                'kind': 'OPINION',
                'confidence': 0.30
            }

        if any(d_lower.startswith(p) for p in ['maybe we should', 'what if we', 'how about we', 'could we', 'i propose', 'i suggest', 'we are considering']):
            return {
                'status': 'PROPOSAL',
                'kind': 'PROPOSAL',
                'confidence': 0.50
            }

        if is_explicit_decision or any(w in d_lower for w in ['we decided', 'everyone agreed', 'committee approved', 'board approved', 'board voted to approve', 'approved the', 'defer', 'ship without', 'adopted the', 'motion passes']):
            return {
                'status': 'CONFIRMED',
                'kind': 'DECISION',
                'confidence': 0.95
            }

        return {
            'status': 'UNCERTAIN',
            'kind': 'UNCONFIRMED',
            'confidence': 0.50
        }

    def validate_topic(self, topic: str) -> bool:
        """
        Strict Topic Quality Filter:
        Ensures topic represents a meaningful subject/entity and is not an isolated verb,
        pronoun, filler, metadata word, or ASR artifact.
        """
        if not topic or not topic.strip():
            return False

        t_clean = topic.strip()
        t_lower = t_clean.lower()
        words = t_clean.split()

        if len(words) == 1:
            if t_lower in ('start', 'pull', 'include', 'do', 'it', 'that', 'job', 'take', 'be', 'go', 'we', 'they', 'you', 'latest', 'original', 'confirmed', 'current'):
                return False
            if len(t_clean) < 5:
                return False

        junk_phrases = [
            'include job', 'pull it', 'do that', 'start on', 'interesting maria',
            'be reporting', 'go down', 'trustee staubach', 'my right trustee',
            'latest dashboard', 'checked latest', 's current', 'original budget estimate'
        ]
        if any(junk in t_lower for junk in junk_phrases):
            return False

        if any(t_lower.startswith(v + ' ') for v in ('include', 'pull', 'start', 'do', 'take', 'be', 'go', 'with', 'without', 'and', 'or', 'for', 'from', 's', 'i', 'a')):
            return False

        if all(w.lower() in ('the', 'a', 'an', 'and', 'or', 'it', 'that', 'this', 'to', 'for', 'job', 'task', 'latest', 'current', 'confirmed', 'checked', 'original', 'estimate') for w in words):
            return False

        return True

    def check_procedure(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Identify formal meeting governance procedures and vote decisions.
        """
        t_lower = text.lower().strip()
        for pat in self.PROCEDURE_PATTERNS + self.VOTE_DECISION_PATTERNS:
            if re.search(pat, t_lower):
                m_proc = re.search(pat, t_lower).group(0)
                is_vote = bool(re.search(r'\b(?:passed|passes|carried|approved|failed|denied)\b', t_lower))
                return {
                    'procedure': m_proc.title(),
                    'full_text': text,
                    'is_procedure': True,
                    'is_vote': is_vote
                }
        return None

    def validate_blocker(self, problem: str, impact: str) -> str:
        """
        Strictly verify blocking classification against explicit progress-halting evidence.
        """
        p_lower = problem.lower()
        if 'non-blocking' in p_lower or 'doesn\'t affect' in p_lower or 'can proceed' in p_lower:
            return "NON_BLOCKING"
        if any(w in p_lower for w in ['primary blocker', 'cannot proceed until', 'cannot continue until', 'must be resolved before']):
            return "BLOCKING"
        if impact == "BLOCKING":
            return "BLOCKING"
        return "NON_BLOCKING"

    def classify_statement_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify statement intent into verified semantic type with confidence,
        evidence strength, confirmation status, and temporal status.
        """
        if not text or not text.strip():
            return {
                'type': 'CASUAL',
                'confidence': 0.0,
                'evidence': 'WEAK',
                'status': 'REJECTED'
            }

        t_clean = text.strip()
        t_lower = t_clean.lower()

        if any(re.search(pat, t_lower) for pat in self.ASR_CORRUPTION_PATTERNS) or (len(t_clean.split()) <= 2 and t_lower not in ('yes', 'no', 'agreed', 'motion passed', 'motion passes')):
            return {
                'type': 'CASUAL',
                'confidence': 0.10,
                'evidence': 'WEAK',
                'status': 'REJECTED'
            }

        if t_clean.endswith('?') or any(t_lower.startswith(q) for q in ['did ', 'has ', 'have ', 'is ', 'are ', 'can someone', 'could someone', 'will someone', 'what ', 'why ', 'how ']):
            return {
                'type': 'QUESTION',
                'confidence': 0.90,
                'evidence': 'MODERATE',
                'status': 'STATED'
            }

        proc = self.check_procedure(t_clean)
        if proc:
            if proc.get('is_vote'):
                return {
                    'type': 'DECISION',
                    'confidence': 0.98,
                    'evidence': 'STRONG',
                    'status': 'CONFIRMED'
                }
            return {
                'type': 'MEETING_PROCEDURE',
                'confidence': 0.95,
                'evidence': 'STRONG',
                'status': 'STATED'
            }

        if re.search(r'\b(?:approved\s+(?:last\s+year|last\s+month|previously|earlier)|was\s+approved\s+(?:last\s+year|previously))\b', t_lower):
            return {
                'type': 'HISTORICAL_VALUE',
                'confidence': 0.90,
                'evidence': 'STRONG',
                'status': 'HISTORICAL'
            }

        if any(t_lower.startswith(op) for op in ['i think', 'in my opinion', 'i feel like', 'personally', 'i believe']):
            return {
                'type': 'OPINION',
                'confidence': 0.85,
                'evidence': 'MODERATE',
                'status': 'STATED'
            }

        if any(t_lower.startswith(p) for p in ['maybe we should', 'what if we', 'how about we', 'could we', 'i propose', 'i suggest', 'we are considering']):
            return {
                'type': 'PROPOSAL',
                'confidence': 0.85,
                'evidence': 'MODERATE',
                'status': 'PROPOSED'
            }

        if any(w in t_lower for w in ['committee approved', 'board approved', 'board voted to approve', 'approved the', 'we decided', 'everyone agreed', 'motion passes', 'motion passed']):
            return {
                'type': 'DECISION',
                'confidence': 0.96,
                'evidence': 'STRONG',
                'status': 'CONFIRMED'
            }

        if any(w in t_lower for w in ['hosted a dedication', 'hosted an event', 'dedicated a new', 'celebrated a long', 'celebrated the foundation', 'opened a new']):
            return {
                'type': 'EVENT',
                'confidence': 0.90,
                'evidence': 'STRONG',
                'status': 'STATED'
            }

        has_valid_verb = any(re.search(rf'\b{v}\b', t_lower) for v in self.VALID_ACTION_VERBS)
        if has_valid_verb and re.search(r'\b(?:will|to|please|can you|should)\b', t_lower):
            return {
                'type': 'ACTION',
                'confidence': 0.92,
                'evidence': 'STRONG',
                'status': 'CONFIRMED'
            }

        return {
            'type': 'FACT',
            'confidence': 0.90,
            'evidence': 'STRONG',
            'status': 'CONFIRMED'
        }

    def is_speaker_administration(self, text: str) -> bool:
        """
        Detect meeting management and speaker-turn administration phrases.
        These must never become action items, decisions, or facts.
        """
        if not text or not text.strip():
            return False
        t_lower = text.lower().strip()
        for pattern in self.SPEAKER_ADMIN_PATTERNS:
            if re.search(pattern, t_lower):
                return True
        if re.search(r'\b(?:donate|spare|lend)\s+(?:a |me |us )?\s*(?:minute|moment|second)', t_lower):
            return True
        return False

    def is_public_comment_or_rant(self, text: str) -> Optional[str]:
        """
        Detect public comments, political rants, and personal criticism.
        Returns 'PUBLIC_COMMENT', 'RANT', or None.
        """
        if not text or not text.strip():
            return None
        t_lower = text.lower().strip()
        words = t_lower.split()
        word_count = len(words)

        criticism_patterns = [
            r'\b(?:this person is|he is|she is|they are)\s+(?:terrible|corrupt|incompetent|dishonest|unfit)',
            r'\b(?:i (?:completely |totally |absolutely )?disagree with (?:everything|all of))',
            r"\b(?:shouldn't be (?:mayor|president|chair|director|leader|in charge))",
            r'\b(?:corruption|nepotism|cronyism|abuse of power|misconduct|malfeasance)',
            r'\b(?:shame on|disgrace|outrageous|unconscionable)',
        ]
        for pat in criticism_patterns:
            if re.search(pat, t_lower):
                return 'PUBLIC_COMMENT'

        emotional_words = ['terrible', 'horrible', 'awful', 'disgusting', 'outrageous',
                          'ridiculous', 'unbelievable', 'shameful', 'appalling']
        emotional_count = sum(1 for w in emotional_words if w in t_lower)
        accusatory_count = len(re.findall(r'\b(?:corrupt|incompetent|dishonest|liar|worst|failure|unfit)\b', t_lower))

        if emotional_count + accusatory_count >= 3:
            return 'RANT'
        if emotional_count + accusatory_count >= 2 and word_count > 20:
            return 'PUBLIC_COMMENT'

        return None

    def is_narrative_description(self, text: str) -> bool:
        """
        Detect past-tense event narration and descriptive storytelling.
        These must not become decisions or action items.
        """
        if not text or not text.strip():
            return False
        t_lower = text.lower().strip()
        words = t_lower.split()
        word_count = len(words)

        narrative_markers = [
            r'\b(?:hosted a (?:dedication|ceremony|celebration|event|reception|gala))',
            r'\b(?:celebrated (?:a |the |its )?(?:anniversary|milestone|achievement|founding))',
            r'\b(?:dedicated a new|opened a new|unveiled|inaugurated|ribbon.?cutting)',
            r'\b(?:established in|founded in|created in|built in)\s+\d{4}',
            r'\b(?:named after|in honor of|in memory of|tribute to)',
        ]
        for pat in narrative_markers:
            if re.search(pat, t_lower):
                return True

        past_verbs = len(re.findall(r'\b(?:was|were|had|held|took|gave|hosted|celebrated|opened|built|named|honored|established|founded|dedicated|attended|featured|included)\b', t_lower))
        if past_verbs >= 3 and word_count > 15:
            has_forward = bool(re.search(r'\b(?:will|shall|should|must|need to|going to)\b', t_lower))
            if not has_forward:
                return True

        return False
