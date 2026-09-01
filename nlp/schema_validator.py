"""
Schema Validation and Output Guardrail Layer for Meetlytic.
Enforces:
1. Terminal punctuation & clause integrity (discards items terminating on incomplete clauses or commas).
2. Anti-dangling grammar checks (rejects dangling conjunctions/prepositions).
3. Politeness vs Decision filter and Status check vs Requirement filter.
4. Strict forward-looking temporal target validation for deadlines.
Strictly free of emojis and emdashes.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

class SchemaGuardrailValidator:
    """
    Validates that all extracted meeting entities are grammatically complete,
    correctly classified, and free of dangling artifacts.
    """

    def __init__(self):

        self.dangling_endings = re.compile(
            r'\b(?:and|or|but|with|for|to|at|in|on|of|the|a|an|as|by|is|are|was|were|are a|is a|remaini|remain|the|our|your|my|its)\s*[,\s\-:]*$',
            re.IGNORECASE
        )

        self.politeness_patterns = [
            re.compile(r'\b(?:thank you|thanks|alright thank you|good morning|have a good day|see you at lunch|see you tomorrow)\b', re.IGNORECASE),
            re.compile(r'\b(?:how are you|hope you are well|nice to meet you|pleasure meeting you)\b', re.IGNORECASE),
            re.compile(r'\b(?:alright|ok|okay|sure thing|sounds good see you|no problem|fine i\'ll behave)\b', re.IGNORECASE)
        ]

        self.status_phrases = [
            re.compile(r'\b(?:have enough resources|progressing as planned|making good progress|everything is on track|team is ready)\b', re.IGNORECASE),
            re.compile(r'\b(?:running smoothly|going well|no major blockers|on schedule)\b', re.IGNORECASE)
        ]

        self.duration_regex = re.compile(
            r'\b(?:running|spent|spending|took|taking|lasted|for|over)\s+(?:\d+|two|three|several)\s+(?:hours?|minutes?|days?)\b',
            re.IGNORECASE
        )

        self.deadline_anchor_regex = re.compile(
            r'\b(?:by|before|due|on|until|within|this|next)\s+(?:today|tomorrow|friday|monday|tuesday|wednesday|thursday|saturday|sunday|week|month|end of day|end of week|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+|\d{1,2}[/-]\d{1,2})\b|\b(?:today|tomorrow)\b',
            re.IGNORECASE
        )

        self.filler_starters = (
            'behave', 'saying', 'consider', 'bring', 'adapt', 'make a note', 'promise',
            'try to protect', 'sound', 'went out', 'spent', 'spending', 'eating', 'coffee',
            'lunch', 'movie', 'shopping', 'windows 98', 'complaint ticket', 'running over'
        )

    def validate_task(self, content: str, owner: str) -> Optional[Dict[str, str]]:
        """Validate an action item or deliverable statement."""
        if not content or len(content.strip()) < 8:
            return None

        clean_text = self._clean_statement(content)
        if not clean_text or len(clean_text) < 8:
            return None

        if self._is_dangling(clean_text):
            return None

        if any(clean_text.lower().startswith(f) for f in self.filler_starters):
            return None

        clean_owner = owner if owner and owner not in ('Speaker', 'Team', 'Unknown', 'None') else "Unassigned"

        return {
            'content': clean_text,
            'owner': clean_owner
        }

    def validate_decision(self, decision: str) -> Optional[str]:
        """
        Validate a finalized consensus agreement. Rejects pleasantries and small talk.
        """
        if not decision or len(decision.strip()) < 12:
            return None

        clean_d = self._clean_statement(decision)
        if not clean_d or len(clean_d) < 12:
            return None

        for pat in self.politeness_patterns:
            if pat.search(clean_d):
                return None

        if self._is_dangling(clean_d) or clean_d.endswith('?'):
            return None

        if any(w in clean_d.lower() for w in ['coffee', 'furniture', 'lunch', 'dinner', 'movie', 'match', 'windows 98', 'complaint ticket']):
            return None

        return clean_d

    def validate_deadline(self, task_desc: str, deadline_str: str) -> Optional[Dict[str, str]]:
        """Validate that a deadline represents a genuine temporal target."""
        if not deadline_str or not task_desc:
            return None

        if self.duration_regex.search(deadline_str) or self.duration_regex.search(task_desc):
            if not self.deadline_anchor_regex.search(deadline_str):
                return None

        if not self.deadline_anchor_regex.search(deadline_str):
            return None

        clean_task = self._clean_statement(task_desc)
        if self._is_dangling(clean_task) or len(clean_task) < 6:
            clean_task = "Project Deliverable"

        return {
            'task': clean_task,
            'deadline': deadline_str.strip()
        }

    def validate_requirement(self, req: str) -> Optional[str]:
        """Validate that a requirement represents a concrete technical/functional specification."""
        if not req or len(req.strip()) < 12:
            return None

        clean_req = self._clean_statement(req)
        req_lower = clean_req.lower()

        for pat in self.status_phrases:
            if pat.search(clean_req):
                return None

        for pat in self.politeness_patterns:
            if pat.search(clean_req):
                return None

        if self._is_dangling(clean_req):
            return None

        if not any(w in req_lower for w in ['must', 'shall', 'required to', 'needs to have', 'has to be', 'mandatory', 'specification']):
            return None

        return clean_req

    def _is_dangling(self, text: str) -> bool:
        """Detect if a text fragment ends in a dangling word or unfinished stem."""
        clean = text.strip()
        if self.dangling_endings.search(clean):
            return True
        words = clean.split()
        if words and words[-1].endswith(('-', '...')):
            return True
        return False

    def _clean_statement(self, text: str) -> str:
        """Strip markdown, timestamps, dangling punctuation, and enforce clean phrasing."""
        t = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', text).strip()
        t = re.sub(r'[\*_\[\]\(\)]', '', t).strip()
        t = re.sub(r'^(?:and|also|so|well|then|first|second|third|\d+\.)\s+', '', t, flags=re.IGNORECASE).strip()
        t = re.sub(r'^[,\s\-:]+', '', t).strip()
        t = re.sub(r'[,\s\-:]+$', '', t).strip()

        t = self.dangling_endings.sub('', t).strip()

        if t:
            t = t[0].upper() + t[1:]
        return t
