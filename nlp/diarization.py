"""
Speaker Diarization and Entity Resolution Module for Meetlytic.
Performs:
1. Multi-speaker turn alignment and cluster tracking (e.g. SPEAKER_00, SPEAKER_01).
2. Name-to-Speaker Entity Linking strictly for anonymous clusters.
3. Strict validation using ParticipantRoster to prevent prepositions (For, By) from becoming names.
Strictly free of emojis and emdashes.
"""

import re
from typing import List, Tuple, Dict, Any, Optional, Set
from nlp.roster import ParticipantRoster

class SpeakerEntityResolver:
    """
    Tracks conversational turns and resolves anonymous speaker clusters into verified human entities.
    """

    def __init__(self):
        self.roster_manager = ParticipantRoster()
        self.cluster_to_name: Dict[str, str] = {}
        self.known_names: Set[str] = set()

    def resolve_turns(self, turns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Analyze conversational flow, resolve anonymous speaker clusters to real human names,
        and return disambiguated (SpeakerName, Utterance) turns.
        """
        if not turns:
            return []

        self._discover_names_and_alignments(turns)

        resolved_turns = []
        for raw_spk, text in turns:
            spk_clean = self._sanitize_speaker_label(raw_spk)

            if self.roster_manager.validate_candidate_name(spk_clean) and not spk_clean.upper().startswith("SPEAKER"):
                resolved_turns.append((spk_clean, text))
                continue

            resolved_name = self.cluster_to_name.get(spk_clean, spk_clean)

            if not self.roster_manager.validate_candidate_name(resolved_name):
                resolved_name = "Speaker"

            resolved_turns.append((resolved_name, text))

        return resolved_turns

    def _sanitize_speaker_label(self, raw_label: str) -> str:
        """Sanitize raw speaker label."""
        clean = re.sub(r'[\*_\[\]\(\)]', '', raw_label).strip()
        clean = re.sub(r'^\d+\s*', '', clean).strip()
        if not clean or clean.isdigit():
            return "Speaker"
        return clean

    def _discover_names_and_alignments(self, turns: List[Tuple[str, str]]):
        """Analyze conversational context to align anonymous speaker clusters with names."""
        intro_patterns = [
            re.compile(r"\b(?:i'm|i am|this is|my name is|here is)\s+([A-Z][a-z]{2,15})\b"),
            re.compile(r"\b([A-Z][a-z]{2,15})\s+speaking\b"),
            re.compile(r"\b([A-Z][a-z]{2,15})\s+here\b"),
        ]

        vocative_address_patterns = [
            re.compile(r"\b(?:hey|hi|hello|good morning|thanks|thank you)\s+([A-Z][a-z]{2,15})\b"),
            re.compile(r"\b(?:what do you think|can you update us|over to you|take it away)\s*,\s*([A-Z][a-z]{2,15})\b"),
            re.compile(r"\b([A-Z][a-z]{2,15})\s*,\s*(?:can you|could you|what's the status|do you agree)\b"),
            re.compile(r"\bpassing (?:this|over) to\s+([A-Z][a-z]{2,15})\b")
        ]

        for idx, (spk, text) in enumerate(turns):
            spk_clean = self._sanitize_speaker_label(spk)

            if self.roster_manager.validate_candidate_name(spk_clean) and not spk_clean.upper().startswith("SPEAKER"):
                self.known_names.add(spk_clean)

            for pat in intro_patterns:
                m = pat.search(text)
                if m:
                    candidate = self.roster_manager.validate_candidate_name(m.group(1))
                    if candidate:
                        if spk_clean.upper().startswith("SPEAKER") or spk_clean == "Speaker":
                            self.cluster_to_name[spk_clean] = candidate
                        self.known_names.add(candidate)

            for pat in vocative_address_patterns:
                for m in pat.finditer(text):
                    candidate = self.roster_manager.validate_candidate_name(m.group(1))
                    if candidate:
                        self.known_names.add(candidate)
                        if idx + 1 < len(turns):
                            next_spk = self._sanitize_speaker_label(turns[idx + 1][0])
                            if (next_spk.upper().startswith("SPEAKER") or next_spk == "Speaker") and next_spk not in self.cluster_to_name:
                                self.cluster_to_name[next_spk] = candidate

        anonymous_clusters = [s for s, _ in turns if s.upper().startswith("SPEAKER") or s == "Speaker"]
        unique_clusters = list(set([self._sanitize_speaker_label(s) for s in anonymous_clusters]))

        if len(unique_clusters) == 2 and len(self.known_names) == 2:
            names_list = list(self.known_names)
            c0, c1 = unique_clusters[0], unique_clusters[1]
            if c0 in self.cluster_to_name and c1 not in self.cluster_to_name:
                rem_name = [n for n in names_list if n != self.cluster_to_name[c0]][0]
                self.cluster_to_name[c1] = rem_name
            elif c1 in self.cluster_to_name and c0 not in self.cluster_to_name:
                rem_name = [n for n in names_list if n != self.cluster_to_name[c1]][0]
                self.cluster_to_name[c0] = rem_name
