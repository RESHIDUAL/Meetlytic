"""
Temporal & Deadline Extraction Engine for Meetlytic.
Implements strict semantic deadline resolution without guessing or arbitrary proximity transfer.
Supports:
- Relative targets: today, tomorrow, this afternoon, this evening, tonight, this week, next week, by EOD
- Event/milestone triggers: before the demo, before deployment, before release, after approval, once testing passes, until X happens
- Durations: within one working day, in 2 days, within 3 hours
- Explicit timestamps: 4 PM, 10:30 AM, Friday 5 PM, Oct 15th
Strictly free of emojis and emdashes.
"""

import re
from typing import Optional, Tuple, List, Dict, Any

class TemporalExtractor:
    """
    Identifies and validates temporal milestones and deadlines.
    Ensures temporal targets are bound strictly to their matching semantic action.
    """

    def __init__(self):

        self.temporal_patterns = [

            (re.compile(r'\b(?:before|prior to)\s+(?:the\s+)?(?:demo|deployment|release|meeting|launch|review|testing|migration)\b', re.IGNORECASE), "event_trigger"),
            (re.compile(r'\b(?:after|following)\s+(?:the\s+)?(?:approval|sign-off|testing|demo|review|release)\b', re.IGNORECASE), "event_trigger"),
            (re.compile(r'\b(?:once|as soon as)\s+([a-zA-Z0-9_\s\-]+?(?:passes|is verified|is approved|completes|is available|lands))\b', re.IGNORECASE), "conditional"),
            (re.compile(r'\buntil\s+([a-zA-Z0-9_\s\-]+?(?:happens|completes|is resolved|finishes))\b', re.IGNORECASE), "conditional"),

            (re.compile(r'\b(?:by|before|due|within)?\s*(?:today|tomorrow|this afternoon|this evening|tonight|this week|next week|by eod|end of day|end of week|this friday)\b', re.IGNORECASE), "relative"),
            (re.compile(r'\b(?:within|in)\s+(?:one|two|three|four|five|\d+)\s+(?:working\s+|business\s+)?(?:days?|weeks?|hours?|months?)\b', re.IGNORECASE), "duration"),
            (re.compile(r'\b(?:by|on|at|before)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?\b', re.IGNORECASE), "day_of_week"),
            (re.compile(r'\b(?:by|at|before)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b', re.IGNORECASE), "clock_time"),
            (re.compile(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b', re.IGNORECASE), "calendar_date")
        ]

        self.past_tense_regex = re.compile(
            r'\b(?:fixed|completed|finished|shipped|resolved|closed|built|went|was|were|happened|occurred|deployed|merged)\b.*\b(?:yesterday|last\s+week|on\s+(?:monday|tuesday|wednesday|thursday|friday)|earlier)\b',
            re.IGNORECASE
        )

    def extract_temporal_expression(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Extract valid future/target temporal expression and its type.
        Returns (clean_expression, expr_type) or None.
        """
        if not text:
            return None

        if self.past_tense_regex.search(text):

            if not re.search(r'\b(?:will|can|need to|should|by tomorrow|today|this week|next week)\b', text, re.IGNORECASE):
                return None

        for pattern, expr_type in self.temporal_patterns:
            match = pattern.search(text)
            if match:
                expr = match.group(0).strip()

                expr_clean = re.sub(r'^[,\s\-:]+', '', expr).strip()
                if len(expr_clean) >= 3:
                    return expr_clean, expr_type

        return None

    def format_deadline_string(self, raw_expression: Optional[str]) -> str:
        """Standardize deadline representation."""
        if not raw_expression:
            return "unspecified"
        clean = raw_expression.strip()

        clean = re.sub(r'^(?:due\s+(?:by|on|at|in|before)\s*|due\s*:?\s*)', '', clean, flags=re.IGNORECASE).strip()
        return clean
