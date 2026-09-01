"""
Dialogue Act & Deliverable Intent Filter for Meetlytic.
Distinguishes genuine operational work deliverables from conversational etiquette,
procedural remarks, promises to communicate, and conversational acknowledgments.
Strictly free of emojis and emdashes.
"""

import re
from typing import Dict, Any, Optional, List

class DeliverableIntentFilter:
    """
    Classifies conversational statements to identify genuine work commitments
    and reject procedural remarks, promises to communicate, and conversational etiquette.
    """

    def __init__(self):

        self.banned_intent_patterns = [
            re.compile(r'\b(?:keep you posted|keep you updated|stay in touch|let you know|inform you)\b', re.IGNORECASE),
            re.compile(r'\b(?:prioritize these|prioritize this|work on these priorities|look into these)\b', re.IGNORECASE),
            re.compile(r'\b(?:make a note|take note|write this down|mental complaint ticket)\b', re.IGNORECASE),
            re.compile(r'\b(?:try to protect|promise to|adapt to|consider your feedback|think about it)\b', re.IGNORECASE),
            re.compile(r'\b(?:understood|sounds good see you|will do see you|thanks for the update|no problem)\b', re.IGNORECASE),
            re.compile(r'\b(?:running on windows|coffee is responsible|start the day with)\b', re.IGNORECASE),
            re.compile(r'\b(?:see you at lunch|see you tomorrow|have a great day|have a good weekend)\b', re.IGNORECASE),
            re.compile(r'\b(?:behave myself|be good|keep it short|not start with)\b', re.IGNORECASE),
            re.compile(r'\b(?:spent three hours|going shopping|furniture shopping)\b', re.IGNORECASE)
        ]

        self.actionable_verbs = {
            'deploy', 'push', 'commit', 'fix', 'resolve', 'complete', 'finish',
            'implement', 'build', 'create', 'write', 'draft', 'document', 'test',
            'execute', 'coordinate', 'integrate', 'update', 'finalize', 'optimize',
            'refactor', 'review', 'prepare', 'send', 'schedule', 'measure', 'deliver'
        }

        self.deliverable_artifacts = {
            'fix', 'bug', 'patch', 'documentation', 'docs', 'section', 'guide',
            'api', 'endpoint', 'dashboard', 'filter', 'filters', 'module', 'branch',
            'presentation', 'deck', 'outline', 'report', 'code', 'repository', 'driver',
            'firmware', 'sensor', 'microcontroller', 'prototype', 'benchmark', 'suite',
            'metrics', 'dataset', 'classifier', 'model', 'tests', 'integration', 'hardware',
            'vendor', 'component', 'components', 'specs', 'specification', 'review'
        }

    def is_valid_deliverable(self, task_text: str) -> bool:
        """
        Verify that a candidate statement represents a concrete work deliverable
        rather than meeting etiquette or procedural communication.
        """
        if not task_text or len(task_text.strip()) < 8:
            return False

        text_lower = task_text.lower().strip()

        for pat in self.banned_intent_patterns:
            if pat.search(text_lower):
                return False

        words = re.findall(r'\b[a-z0-9_\-]+\b', text_lower)
        if len(words) < 3:
            return False

        has_action_verb = any(w in self.actionable_verbs for w in words)

        has_artifact = any(w in self.deliverable_artifacts for w in words)

        if has_action_verb and has_artifact:
            return True

        if has_artifact and len(words) >= 4:
            return True

        return False
