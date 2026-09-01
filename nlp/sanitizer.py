"""
Text Sanitization and ASR Normalization Module for Meetlytic.
Implements:
1. Complete timestamp, VTT/SRT, and bracket metadata stripping.
2. Robust markdown speaker parsing (**A:**, **David:**, [SPEAKER_01]:) without artifact bleeding.
3. Two-Stage Mixed Sentence Filtering: Strips emotional complaints / casual preambles
   while strictly preserving technical facts, metrics, issues, and action commitments.
Strictly free of emojis and emdashes.
"""

import re
from typing import List, Tuple, Dict, Any, Optional

class TextSanitizer:
    """
    Sanitizes raw meeting transcripts, audio ASR outputs, and subtitle formats.
    """

    def __init__(self):

        self.timestamp_patterns = [
            re.compile(r'\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*(?:-->|-)?\s*(?:\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\s*\]'),
            re.compile(r'\(\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*(?:-->|-)?\s*(?:\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\s*\)'),
            re.compile(r'<\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*>'),
            re.compile(r'\b\d{1,2}:\d{2}:\d{2}(?:[,\.]\d{1,3})?\s*(?:-->|-)\s*\d{1,2}:\d{2}:\d{2}(?:[,\.]\d{1,3})?\b'),
            re.compile(r'\b\d{1,2}:\d{2}:\d{2}\b'),
            re.compile(r'\b\d{1,2}:\d{2}\b(?=\s*[-:]|\s*$)')
        ]

        self.filler_patterns = [
            re.compile(r'\b(?:uh|um|uhm|er|ah|eh|erm|hmm|hm)\b', re.IGNORECASE),
            re.compile(r'\b(?:you know|i mean|sort of|kind of)\b(?=\s*[,.]|\s+[a-z])', re.IGNORECASE)
        ]

        self.stutter_pattern = re.compile(r'\b([a-zA-Z]{1,8})[---\s]+\1\b', re.IGNORECASE)

        self.preamble_patterns = [
            re.compile(r'^\s*[\*_\[\]\(\)]+\s*'),
            re.compile(r'^(?:(?:yes|no|yeah|yep|sure|ok|alright|good morning|good morning everyone)[,\s\-]+)+', re.IGNORECASE),
            re.compile(r'^(?:(?:arsenal|real madrid|barcelona|chelsea|liverpool|india|australia) (?:won|played well|played amazingly)[.!?,\s\-]+)+', re.IGNORECASE),
            re.compile(r'^(?:(?:it was (?:great|good|fun)|went hiking|had fun|great match|great game)[.!?,\s\-]+)+', re.IGNORECASE),
            re.compile(r'^(?:(?:but )?(?:speaking of|getting to|getting back to|talking about|regarding) [^.!?]+?[,\s\-]+)+', re.IGNORECASE),
            re.compile(r'^(?:(?:honestly|frankly|personally|to be honest|look|listen|man|dude|gosh|damn|ugh)[,\s\-]+)+', re.IGNORECASE),
            re.compile(r'^(?:i (?:really )?(?:hate|dislike|can\'t stand) (?:this|the|our|working with) [a-zA-Z0-9_\s\-]+?(?:,| but| yet|\.|\;)\s*)', re.IGNORECASE),
            re.compile(r'^(?:this (?:terrible|horrible|annoying|frustrating|broken|stupid|ridiculous) [a-zA-Z0-9_\s\-]+? is (?:so |really )?(?:bad|terrible|annoying|frustrating|ridiculous)[,\s\-]+)+', re.IGNORECASE),
            re.compile(r'^(?:(?:the |our )?(?:manager|lead|boss|management|finance|team|developer) (?:is|are) (?:being )?(?:useless|incompetent|ridiculous|impossible|lazy|annoying|terrible|stupid|awful)[^.!?]*?(?:because|since|as|\.|\;)\s*)', re.IGNORECASE),
            re.compile(r'^(?:(?:management|leadership|they) (?:doesn\'t|don\'t) understand (?:this|the) (?:project|work)[.!?,\s\-]+)+', re.IGNORECASE),
            re.compile(r'^(?:(?:everyone is|people are) (?:frustrated|upset|angry) with [^.!?]+?[.!?,\s\-]+)+', re.IGNORECASE),
            re.compile(r'^(?:(?:she|he) (?:never listens|is terrible at (?:her|his) job|never approves anything)[.!?,\s\-]+)+', re.IGNORECASE),
            re.compile(r'^(?:(?:anyway|anyways|on another note|by the way|excellent|perfect|great)[,\s\-:\.]+)+', re.IGNORECASE),
            re.compile(r'^(?:(?:this|the) (?:stupid|broken|terrible|annoying) [a-zA-Z0-9_\s\-]+? (?:keeps|is) [^.!?]+?(?:driving everyone crazy|driving us crazy|annoying everyone)[.!?,\s\-]+)', re.IGNORECASE),
            re.compile(r'^(?:after (?:lunch|dinner|coffee|the break|my break|work)[,\s\-]+)', re.IGNORECASE),
            re.compile(r'^(?:did you (?:watch|see|hear about) [^.!?]+?[.!?]\s*(?:yes|no|yeah)?[^.!?]*?[.!?]\s*(?:but|speaking of work|anyway)?[,\s\-]+)', re.IGNORECASE)
        ]

        self.casual_indicators = [
            'real madrid', 'barcelona', 'arsenal', 'chelsea', 'manchester', 'liverpool', 'football',
            'soccer', 'match yesterday', 'game yesterday', 'movie last night', 'weekend plans',
            'see you at lunch', 'have lunch', 'grab coffee', 'eat lunch', 'dinner tonight',
            'weather is nice', 'traffic was crazy', 'how was your weekend', 'good morning', 'good afternoon',
            'hey there', 'see you later', 'have a good one', 'see you at lunch!'
        ]

    def clean_text(self, text: str) -> str:
        """
        Sanitize and normalize text, removing timestamps, metadata, and speech disfluencies.
        """
        if not text:
            return ""

        cleaned = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').replace('-', ' - ')

        for pat in self.timestamp_patterns:
            cleaned = pat.sub(' ', cleaned)

        cleaned = re.sub(r'^\s*\d+\s*$', '', cleaned, flags=re.MULTILINE)

        for pat in self.filler_patterns:
            cleaned = pat.sub(' ', cleaned)

        cleaned = self.stutter_pattern.sub(r'\1', cleaned)

        cleaned = re.sub(r'[\*_\[\]\(\)]', '', cleaned)

        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\s+([,.:;!?])', r'\1', cleaned)
        cleaned = re.sub(r'([.!?]){2,}', r'\1', cleaned)

        return cleaned.strip()

    def strip_non_professional_preamble(self, text: str) -> str:
        """
        Stage 2 Filtering: Strips emotional complaints and casual transitions
        while strictly retaining core professional facts and deliverables.
        """
        if not text:
            return ""

        cleaned = self.clean_text(text)

        for pat in self.preamble_patterns:
            cleaned = pat.sub('', cleaned).strip()

        cleaned = re.sub(r'^(?:this\s+)?(?:terrible|horrible|annoying|frustrating|stupid|awful)\s+([a-zA-Z0-9_\-]+)', r'The \1', cleaned, flags=re.IGNORECASE).strip()

        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned.strip()

    def check_purely_casual_or_venting(self, text: str) -> Tuple[bool, str]:
        """
        Determine if utterance is 100% casual / social / venting with zero business or technical content.
        """
        text_lower = text.lower().strip()

        has_metrics = bool(re.search(r'\b(?:\d+(?:\.\d+)?%|\d+\s*(?:ms|fps|users|requests|units|kb|mb|gb|rpm|am|pm)|500|404|401|403|502|503)\b', text_lower))
        has_tech_substance = bool(re.search(r'\b(?:api|database|service|endpoint|server|auth|migration|deploy|pipeline|test|bug|error|failing|failed|timeout|latency|script|permission|security|dashboard|feature|export|release|version|build|pull request|branch|commit|fix|schedule|task|requirement|spec|upload|network|crash|listener|connection|credentials)\b', text_lower))

        if has_metrics or has_tech_substance:
            return False, "professional"

        if any(ind in text_lower for ind in self.casual_indicators) or re.search(r'\b(?:lunch|dinner|coffee break|heading out|see you|bye|goodbye)\b', text_lower):
            return True, "casual"

        if re.search(r'^(?:(?:hey|hi|hello)\s+(?:team|everyone|all|there)[,.\s!]*)+(?:let\'s start [^.!?]+)?$', text_lower):
            if not has_tech_substance:
                return True, "casual"

        if re.search(r'^(?:i (?:really )?(?:hate|dislike|am sick of|am tired of) [^.!?]+?|this is (?:so )?(?:terrible|annoying|frustrating|stupid)\.?)$', text_lower):
            return True, "venting"

        return False, "neutral"

    def parse_speaker_turns(self, raw_transcript: str) -> List[Tuple[str, str]]:
        """
        Parse multi-line transcript into structured (Speaker, Utterance) turns.
        Supports markdown bolding (**A:**, **David:**, [SPEAKER_01]:) without artifact bleeding.
        """
        if not raw_transcript:
            return []

        lines = raw_transcript.split('\n')
        turns: List[Tuple[str, str]] = []
        current_speaker = "Speaker"
        current_sentences: List[str] = []

        speaker_pattern = re.compile(
            r'^\s*(?:\[|\()?[\*\_]{0,2}([A-Za-z0-9_\s\.\-]{1,25}?)[\*\_]{0,2}\s*(?::\*\*|:\s*\*+|\*{1,2}:|:\s*|-|\]|\))\s*(.*)$',
            re.IGNORECASE
        )

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            for pat in self.timestamp_patterns:
                line_str = pat.sub('', line_str).strip()

            if not line_str:
                continue

            match = speaker_pattern.match(line_str)
            if match:
                if current_sentences:
                    full_turn = " ".join(current_sentences).strip()
                    cleaned_turn = self.clean_text(full_turn)
                    if cleaned_turn:
                        turns.append((current_speaker, cleaned_turn))
                    current_sentences = []

                raw_spk = match.group(1).strip()
                spk_clean = re.sub(r'[\*_\[\]\(\)\:]', '', raw_spk).strip()
                spk_clean = re.sub(r'^\d+\s*', '', spk_clean).strip()
                if not spk_clean or spk_clean.isdigit():
                    spk_clean = "Speaker"

                current_speaker = spk_clean
                utterance_body = match.group(2).strip()
                if utterance_body:
                    current_sentences.append(utterance_body)
            else:
                current_sentences.append(line_str)

        if current_sentences:
            full_turn = " ".join(current_sentences).strip()
            cleaned_turn = self.clean_text(full_turn)
            if cleaned_turn:
                turns.append((current_speaker, cleaned_turn))

        if len(turns) == 0 or (len(turns) == 1 and turns[0][0] == "Speaker"):
            single_text = self.clean_text(raw_transcript)
            sentences = self.reconstruct_sentences(single_text)
            turns = [("Speaker", s) for s in sentences if len(s.strip()) > 3]

        return turns

    def reconstruct_sentences(self, text: str) -> List[str]:
        """Reconstruct complete sentences across chunks."""
        if not text:
            return []

        raw_splits = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\*])', text)
        sentences = []

        for s in raw_splits:
            s_clean = self.clean_text(s)
            if len(s_clean) > 2:
                s_clean = s_clean[0].upper() + s_clean[1:]
                if not s_clean.endswith(('.', '!', '?')):
                    s_clean += '.'
                sentences.append(s_clean)

        return sentences if sentences else [text]
