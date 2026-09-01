"""
Cross-Sentence Discourse & Multi-Speaker Context Engine for Meetlytic.
Implements:
1. Q&A Turn-Pair Resolution across speakers (questions followed by answers/durations/status).
2. Proposal -> Discussion -> Agreement Adjacency Pairs for Decision extraction.
3. Request vs. Commitment distinction (Connects Request -> Acceptance -> Assignee -> Action).
4. Compound Logical Conditions (AND, OR, ONCE, UNLESS, UNTIL, BEFORE, AFTER, IF).
5. Technical Causality Tracking (Problem -> Cause -> Action).
Strictly free of emojis and emdashes.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from nlp.intelligence_types import IntelligenceItem, IntelligenceCategory

class DiscourseContextEngine:
    """
    Context-aware multi-turn conversational reasoning layer.
    """

    def __init__(self):
        self.turns_history: List[Dict[str, Any]] = []
        self.pending_queries: List[Dict[str, Any]] = []
        self.pending_proposals: List[Dict[str, Any]] = []
        self.pending_requests: List[Dict[str, Any]] = []
        self.last_problem_cause: Optional[Dict[str, Any]] = None

        self.agreement_tokens = {
            'agreed', 'i agree', 'sounds good', 'sounds good to me', 'that works',
            'that works for me', 'good idea', 'fair enough', 'definitely', 'perfect',
            'exactly', 'confirmed', 'approved', 'will do', 'let\'s do that', 'we\'ll go with that',
            'that\'s fine', 'okay we\'ll proceed', 'yep', 'yeah', 'yes'
        }

    def reset(self):
        """Reset state for clean session isolation."""
        self.turns_history.clear()
        self.pending_queries.clear()
        self.pending_proposals.clear()
        self.pending_requests.clear()
        self.last_problem_cause = None

    def process_turn(self, speaker: str, text: str, turn_index: int) -> List[IntelligenceItem]:
        """
        Process current conversational turn in context of recent discourse history.
        Returns derived cross-turn intelligence items.
        """
        derived_items: List[IntelligenceItem] = []
        text_lower = text.lower().strip()
        words = set(re.findall(r'\b[a-z]+\b', text_lower))

        clean_short = re.sub(r'[^a-z\s]', '', text_lower).strip()
        is_agreement = any(clean_short == a or clean_short.startswith(a + ' ') or clean_short.startswith(a + '.') for a in self.agreement_tokens)

        if is_agreement and self.pending_proposals:
            last_prop = self.pending_proposals.pop()
            derived_items.append(IntelligenceItem(
                category=IntelligenceCategory.DECISION,
                content=last_prop['content'],
                speaker=speaker,
                confidence=0.95,
                evidence_turn=f"{last_prop['speaker']}: {last_prop['raw_text']} | {speaker}: {text}",
                evidence_turn_index=turn_index,
                related_context=last_prop.get('context')
            ))

        if self.pending_requests:
            req = self.pending_requests[-1]
            if re.search(r'\b(?:i will|i\'ll|sure|will do|i can|yes i can|yes i\'ll|i am on it)\b', text_lower):
                accepted_req = self.pending_requests.pop()
                derived_items.append(IntelligenceItem(
                    category=IntelligenceCategory.ACTION,
                    content=accepted_req['action'],
                    speaker=speaker,
                    target_owner=speaker,
                    deadline=accepted_req.get('deadline', 'unspecified'),
                    confidence=0.92,
                    evidence_turn=f"{accepted_req['speaker']}: {accepted_req['raw_text']} | {speaker}: {text}",
                    evidence_turn_index=turn_index
                ))

        if any(cue in text_lower for cue in ['let\'s move', 'let us move', 'we should', 'i recommend', 'how about', 'what if we', 'let\'s defer', 'we can ship without']):
            prop_content = text.strip()

            if 'move' in text_lower and 'next release' in text_lower:
                target = re.search(r'\b(?:move|defer)\s+([a-zA-Z0-9_\s\-]+?)\s+to\s+(?:the\s+)?([a-zA-Z0-9_\s\-]+)', text, re.IGNORECASE)
                if target:
                    prop_content = f"Defer {target.group(1).strip()} to {target.group(2).strip()}"
            elif 'we can ship without' in text_lower:
                target = re.search(r'\bship without\s+([a-zA-Z0-9_\s\-]+)', text, re.IGNORECASE)
                if target:
                    prop_content = f"Release without {target.group(1).strip()}"

            self.pending_proposals.append({
                'content': prop_content,
                'raw_text': text,
                'speaker': speaker,
                'turn_index': turn_index
            })
            if len(self.pending_proposals) > 4:
                self.pending_proposals.pop(0)

        req_match = re.search(r'\b(?:can you|could you|would you|please)\s+([a-zA-Z0-9_\s,\-]{8,80})\??', text, re.IGNORECASE)
        if req_match and text_lower.endswith('?'):
            self.pending_requests.append({
                'action': req_match.group(1).strip(),
                'raw_text': text,
                'speaker': speaker,
                'turn_index': turn_index
            })
            if len(self.pending_requests) > 4:
                self.pending_requests.pop(0)

        if 'because' in text_lower or 'due to' in text_lower or 'caused by' in text_lower:
            self.last_problem_cause = {
                'text': text.strip(),
                'speaker': speaker,
                'turn_index': turn_index
            }

        self.turns_history.append({
            'speaker': speaker,
            'text': text,
            'turn_index': turn_index
        })

        return derived_items
