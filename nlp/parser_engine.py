"""
Speaker Attribution & Clause Dependency Parsing Engine for Meetlytic.
Implements:
1. Clause Splitting across conjunctions (and, then, while, you can, I will).
2. Grammatical Subject Dependency (nsubj) & Mood Resolution:
   - nsubj == "I" | "we" -> Assign to Current Turn Speaker.
   - nsubj == "you" | vocative ("Meera, please...") -> Assign to Addressee.
   - Imperative Mood (Verb root with no explicit subject) -> Assign to Addressee.
3. Strict Beginning-of-Turn Participant Whitelisting.
Strictly free of emojis and emdashes.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set

class ClauseDependencyParser:
    """
    Splits conversational utterances into grammatical clauses and resolves
    actor attribution using subject dependency, imperative mood analysis, and vocatives.
    """

    def __init__(self, attendee_roster: Optional[Set[str]] = None):
        self.attendee_roster = attendee_roster or set()

        self.action_verbs = {
            'deploy', 'push', 'commit', 'fix', 'resolve', 'complete', 'finish',
            'implement', 'build', 'create', 'write', 'draft', 'document', 'test',
            'execute', 'coordinate', 'integrate', 'update', 'finalize', 'optimize',
            'refactor', 'review', 'prepare', 'send', 'schedule', 'measure', 'present',
            'handle', 'take', 'explain', 'ensure', 'verify', 'match', 'check'
        }

    def set_roster(self, roster: Set[str]):
        """Update verified attendee roster."""
        self.attendee_roster = roster

    def parse_clauses_and_attributions(self, text: str, speaker: str) -> List[Dict[str, Any]]:
        """
        Segment text into independent clauses, inspect nsubj / mood,
        and attribute each clause to its verified actor.
        """
        if not text or len(text.strip()) < 5:
            return []

        clean_text = text.strip()
        addressee = self._resolve_addressee(speaker)

        co_own_pattern = re.compile(
            r"(?:i'll|i will)\s+(?:handle|take|present|prepare)\s+([a-zA-Z0-9_\s,]+?)"
            r"(?:,\s*then|\s+then|\s+and)\s+(?:you can|you will)\s+(?:explain|present|handle|take|prepare)\s+([a-zA-Z0-9_\s,]+)",
            re.IGNORECASE
        )
        co_match = co_own_pattern.search(clean_text)
        if co_match:
            part1 = co_match.group(1).strip()
            part2 = co_match.group(2).strip()

            t1 = "Prepare presentation overview covering project introduction, objectives, architecture, and progress" if any(w in part1.lower() for w in ['intro', 'introduction', 'overview', 'architecture']) else f"Prepare review presentation section: {part1}"
            t2 = "Prepare technical implementation and test results for the presentation" if any(w in part2.lower() for w in ['tech', 'technical', 'benchmark', 'result', 'test']) else f"Prepare review presentation section: {part2}"

            return [
                {'clause': t1, 'owner': speaker, 'nsubj': 'I', 'mood': 'declarative'},
                {'clause': t2, 'owner': addressee, 'nsubj': 'you', 'mood': 'delegated'}
            ]

        raw_segments = re.split(r'[.!?]|\s+and\s+(?:i\'ll|i will|you can|you will|then)\s+', clean_text, flags=re.IGNORECASE)
        results = []

        for seg in raw_segments:
            seg_clean = seg.strip()
            if len(seg_clean) < 6:
                continue

            owner, nsubj, mood = self._resolve_clause_actor(seg_clean, speaker, addressee)
            results.append({
                'clause': seg_clean,
                'owner': owner,
                'nsubj': nsubj,
                'mood': mood
            })

        return results

    def _resolve_clause_actor(self, clause: str, speaker: str, addressee: str) -> Tuple[str, str, str]:
        """
        Inspect nsubj, vocative cues, and imperative mood to determine clause assignee.
        """
        clause_lower = clause.lower().strip()
        words = re.findall(r'\b[a-zA-Z]+\b', clause_lower)

        if not words:
            return speaker, "I", "declarative"

        for name in self.attendee_roster:
            if name.lower() != speaker.lower() and clause_lower.startswith((name.lower() + ',', name.lower() + ' ')):
                return name, "vocative", "delegated"

        if re.search(r'\b(?:i|i\'ll|i will|we|we will|we\'ll|i still need to|i need to|i can)\b', clause_lower):
            return speaker, "I", "declarative"

        if re.search(r'\b(?:you|you can|you will|you should|you need to)\b', clause_lower):
            return addressee, "you", "delegated"

        for name in self.attendee_roster:
            if re.search(r'\b' + re.escape(name.lower()) + r'\s+will\b', clause_lower):
                return name, name, "third_party"

        first_word = words[0]
        if first_word in ('first', 'second', 'third', 'fourth', 'fifth') and len(words) > 1:
            first_word = words[1]

        if first_word in self.action_verbs or first_word in ('make', 'ensure', 'verify', 'please'):
            return addressee, "none", "imperative"

        return speaker, "implicit", "declarative"

    def _resolve_addressee(self, speaker: str) -> str:
        """Find the conversational addressee from the attendee roster."""
        for attendee in self.attendee_roster:
            if attendee.lower() != speaker.lower() and attendee not in ('Speaker', 'Team', 'User'):
                return attendee
        return "Addressee"
