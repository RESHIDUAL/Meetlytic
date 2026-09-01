"""
Verified Participant Roster and Pronoun Resolver for Meetlytic.
Performs:
1. Building a strict verified meeting attendee roster from verified speaker turns.
2. Banning adverbs, prepositions, generic adjectives, and numeric artifacts from becoming names.
3. Strict resolution of 'I', 'you', and third-party assignments to verified roster attendees.
Strictly free of emojis and emdashes.
"""

import re
from typing import List, Tuple, Dict, Any, Optional, Set

class ParticipantRoster:
    """
    Maintains a strict, validated list of meeting attendees and resolves pronouns.
    """

    BANNED_WORDS = {
        'currently', 'for', 'by', 'with', 'from', 'then', 'also', 'maybe', 'well',
        'just', 'very', 'really', 'mostly', 'especially', 'main', 'remaining',
        'primary', 'initial', 'completed', 'working', 'core', 'current', 'entire',
        'final', 'updated', 'major', 'individual', 'complete', 'previous', 'first',
        'second', 'third', 'fourth', 'fifth', 'good', 'morning', 'afternoon',
        'evening', 'hello', 'there', 'team', 'everyone', 'someone', 'anyone',
        'speaker', 'user', 'meeting', 'project', 'update', 'status', 'progress',
        'dashboard', 'database', 'api', 'sensor', 'firmware', 'hardware', 'vendor',
        'today', 'tomorrow', 'friday', 'monday', 'tuesday', 'wednesday', 'thursday',
        'saturday', 'sunday', 'provide', 'tell', 'give', 'explain', 'check', 'send',
        'share', 'show', 'make', 'take', 'see', 'look', 'know', 'think', 'have',
        'find', 'start', 'begin', 'help', 'discuss', 'review', 'verify', 'action',
        'decision', 'task', 'step', 'note', 'details', 'issues', 'thing', 'things',
        'all', 'both', 'each', 'more', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'can', 'will', 'don', 'should', 'now',
        'yes', 'no', 'yeah', 'yep', 'nope', 'alright', 'sure', 'certainly', 'perfect',
        'agreed', 'exactly', 'great', 'fine', 'cool', 'okay', 'windows', 'coffee'
    }

    def __init__(self):
        self.roster: Set[str] = set()
        self.speaker_turns_count: Dict[str, int] = {}

    def build_roster(self, turns: List[Tuple[str, str]]) -> Set[str]:
        """
        Build verified attendee roster from initial pass of conversational turns.
        """
        self.roster.clear()
        self.speaker_turns_count.clear()

        for spk, text in turns:
            clean_name = self.validate_candidate_name(spk)
            if clean_name:
                self.roster.add(clean_name)
                self.speaker_turns_count[clean_name] = self.speaker_turns_count.get(clean_name, 0) + 1

            intro_matches = re.findall(r"\b(?:i'm|i am|this is|my name is)\s+([A-Z][a-z]{2,15})\b", text, re.IGNORECASE)
            for m in intro_matches:
                intro_name = self.validate_candidate_name(m)
                if intro_name:
                    self.roster.add(intro_name)

        return self.roster

    def validate_candidate_name(self, candidate: str) -> Optional[str]:
        """
        Validate whether a string is a genuine human name or an invalid artifact/adverb/digit.
        """
        if not candidate:
            return None

        clean = re.sub(r'[\*_\[\]\(\)\d]', '', candidate).strip()
        clean = re.sub(r'^[,\s\-:]+', '', clean).strip()

        if not clean or len(clean) < 2 or len(clean) > 20:
            return None

        if clean.lower() in self.BANNED_WORDS:
            return None

        if clean.isdigit() or len(clean) <= 1:
            return None

        return clean.capitalize()

    def resolve_owner(self, claimed_owner: str, current_speaker: str, text: str) -> str:
        """
        Strictly resolve task owner to an attendee on the verified roster.
        Maps pronouns ("I", "you", "we") to actual participants.
        """
        text_lower = text.lower()
        speaker_valid = self.validate_candidate_name(current_speaker)

        if re.search(r"\b(?:i will|i'll|i can|i need to|i still need to|i am going to|i'm going to)\b", text_lower):
            if speaker_valid and speaker_valid in self.roster:
                return speaker_valid

        if re.search(r"\b(?:you will|you can|you should)\b", text_lower) or re.search(r"^(?:first|second|third|\d+\.)\s+", text_lower):
            addressee = self.get_other_participant(current_speaker)
            if addressee and addressee in self.roster:
                return addressee

        claimed_valid = self.validate_candidate_name(claimed_owner)
        if claimed_valid and claimed_valid in self.roster:
            return claimed_valid

        if speaker_valid and speaker_valid in self.roster:
            return speaker_valid

        return "Unassigned"

    def get_other_participant(self, current_speaker: str) -> Optional[str]:
        """Find the conversational partner in a dialogue."""
        spk_clean = self.validate_candidate_name(current_speaker)
        for attendee in self.roster:
            if attendee != spk_clean:
                return attendee
        return None
