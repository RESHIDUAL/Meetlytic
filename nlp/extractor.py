"""
Comprehensive Multi-Type Professional Intelligence Extractor for Meetlytic.
Classifies and extracts 18 distinct intelligence categories:
- ACTION, DECISION, AGREEMENT, STATUS, ISSUE, RISK, REQUIREMENT,
  DEPENDENCY, BLOCKER, CONDITION, CONSTRAINT, DEADLINE, MILESTONE,
  METRIC, SCOPE, STAKEHOLDER_REQUEST, TECHNICAL_FACT, DEFERRED_WORK.
Strictly free of emojis and emdashes.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

from nlp.intelligence_types import IntelligenceItem, IntelligenceCategory, MeetingSessionIntelligence
from nlp.sanitizer import TextSanitizer
from nlp.temporal import TemporalExtractor
from nlp.parser_engine import ClauseDependencyParser
from nlp.discourse import DiscourseContextEngine
from nlp.roster import ParticipantRoster
from nlp.synthesizer import AbstractiveSynthesizer

@dataclass
class ExtractedInfo:
    """Container for multi-type intelligence extracted from a single turn or session."""
    items: List[IntelligenceItem] = field(default_factory=list)
    raw_text: str = ""

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        return [{'content': it.content, 'owner': it.target_owner, 'deadline': it.deadline}
                for it in self.items if it.category == IntelligenceCategory.ACTION]

    @property
    def decisions(self) -> List[str]:
        return [it.content for it in self.items if it.category in (IntelligenceCategory.DECISION, IntelligenceCategory.AGREEMENT)]

    @property
    def deadlines(self) -> List[Dict[str, str]]:
        return [{'task': it.content, 'deadline': it.deadline or "unspecified"}
                for it in self.items if it.deadline and it.deadline != "unspecified"]

    @property
    def requirements(self) -> List[str]:
        return [it.content for it in self.items if it.category in (IntelligenceCategory.REQUIREMENT, IntelligenceCategory.DEPENDENCY, IntelligenceCategory.STAKEHOLDER_REQUEST)]

    @property
    def projects(self) -> List[str]:
        return [it.content for it in self.items if it.category in (IntelligenceCategory.STATUS, IntelligenceCategory.TECHNICAL_FACT)]

class ProfessionalInfoExtractor:
    """
    Context-aware, Multi-Type Professional Intelligence Extraction Engine.
    """

    def __init__(self):
        self.sanitizer = TextSanitizer()
        self.temporal_extractor = TemporalExtractor()
        self.parser_engine = ClauseDependencyParser()
        self.discourse_engine = DiscourseContextEngine()
        self.roster_manager = ParticipantRoster()
        self.synthesizer = AbstractiveSynthesizer()

        self.metric_regex = re.compile(
            r'\b(?:\d+(?:\.\d+)?%|\d+\s*(?:ms|fps|users|requests|units|kb|mb|gb|rpm|am|pm|working\s+days?|days?|hours?|minutes?)|'
            r'(?:from\s+)?\d+(?:\.\d+)?%\s+to\s+\d+(?:\.\d+)?%|'
            r'\$\s*\d+(?:\.\d+)?(?:\s*(?:k|m|million|billion))?|500|404|401|403|502|503)\b',
            re.IGNORECASE
        )

        self.speaker_profiles: Dict[str, Dict[str, Any]] = {}

    def reset_session(self):
        """Reset state for clean session isolation."""
        self.discourse_engine.reset()
        self.speaker_profiles.clear()
        self.roster_manager = ParticipantRoster()

    def register_participant(self, name: str):
        """Register attendee on roster and create isolated profile."""
        clean_name = self.roster_manager.validate_candidate_name(name)
        if clean_name and clean_name not in self.speaker_profiles:
            self.speaker_profiles[clean_name] = {
                'name': clean_name,
                'role': 'Participant',
                'focus': 'Project Discussion',
                'tasks': [],
                'speaking_turns': 0
            }

    def extract(self, text: str, speaker: str = "Speaker", turn_index: int = 0, is_professional: bool = True) -> ExtractedInfo:
        """
        Extract all relevant professional intelligence types from an utterance.
        """
        clean_text = self.sanitizer.clean_text(text)
        clean_speaker = self.roster_manager.validate_candidate_name(speaker) or "Speaker"
        if clean_speaker != "Speaker":
            self.register_participant(clean_speaker)

        info = ExtractedInfo(raw_text=clean_text)
        if not clean_text:
            return info

        if clean_speaker in self.speaker_profiles:
            self.speaker_profiles[clean_speaker]['speaking_turns'] += 1

        filtered_text = self.sanitizer.strip_non_professional_preamble(clean_text)
        text_lower = filtered_text.lower()

        self.parser_engine.set_roster(set(self.speaker_profiles.keys()))

        discourse_items = self.discourse_engine.process_turn(clean_speaker, filtered_text, turn_index)
        for d_item in discourse_items:
            info.items.append(d_item)

        metric_matches = self.metric_regex.findall(filtered_text)

        issue_cues = ['failing for', 'failing', 'failed', 'error', 'timeout', 'timed out', 'latency', 'memory leak',
                      'crash', 'bug', 'defect', 'broke', 'broken', 'slow', 'exhaustion', 'blocking release', 'blocker']
        if any(cue in text_lower for cue in issue_cues):
            cat = IntelligenceCategory.BLOCKER if 'blocking' in text_lower or 'blocker' in text_lower else IntelligenceCategory.ISSUE
            info.items.append(IntelligenceItem(
                category=cat,
                content=filtered_text,
                speaker=clean_speaker,
                confidence=0.92,
                evidence_turn=filtered_text,
                evidence_turn_index=turn_index,
                metrics=metric_matches
            ))

        condition_patterns = [
            r'\b(?:should wait until|must wait until|can proceed once|wait until|once|only after)\s+([a-zA-Z0-9_\s\-]+)',
            r'\b([a-zA-Z0-9_\s\-]+?)\s+(?:must pass|must be approved|is required before)\b'
        ]
        if any(re.search(pat, filtered_text, re.IGNORECASE) for pat in condition_patterns):
            info.items.append(IntelligenceItem(
                category=IntelligenceCategory.CONDITION,
                content=filtered_text,
                speaker=clean_speaker,
                confidence=0.90,
                evidence_turn=filtered_text,
                evidence_turn_index=turn_index
            ))

        if any(w in text_lower for w in ['needs the', 'needs updated', 'needs to verify', 'must approve', 'wants csv', 'wants excel', 'requires', 'requirement']):
            cat = IntelligenceCategory.STAKEHOLDER_REQUEST if 'wants' in text_lower or 'client' in text_lower else IntelligenceCategory.REQUIREMENT
            info.items.append(IntelligenceItem(
                category=cat,
                content=filtered_text,
                speaker=clean_speaker,
                confidence=0.90,
                evidence_turn=filtered_text,
                evidence_turn_index=turn_index
            ))

        if any(w in text_lower for w in ['is ready but', 'has not been tested', 'is functional', 'is implemented', 'dropped from', 'uptime', 'requests per second', 'is running']):
            info.items.append(IntelligenceItem(
                category=IntelligenceCategory.STATUS,
                content=filtered_text,
                speaker=clean_speaker,
                confidence=0.88,
                evidence_turn=filtered_text,
                evidence_turn_index=turn_index,
                metrics=metric_matches
            ))

        clause_attributions = self.parser_engine.parse_clauses_and_attributions(filtered_text, clean_speaker)
        temporal_match = self.temporal_extractor.extract_temporal_expression(filtered_text)
        turn_deadline = temporal_match[0] if temporal_match else None

        for clause_info in clause_attributions:
            c_text = clause_info['clause']
            c_owner = clause_info['owner']

            if any(w in c_text.lower() for w in ['i will', 'i\'ll', 'prepare', 'coordinate', 'deploy', 'complete', 'update', 'finish', 'push', 'test', 'write', 'handle']):
                syn_action = self.synthesizer.synthesize_task(c_text, c_owner)
                if syn_action and len(syn_action) > 6:
                    info.items.append(IntelligenceItem(
                        category=IntelligenceCategory.ACTION,
                        content=syn_action,
                        speaker=clean_speaker,
                        target_owner=c_owner,
                        deadline=turn_deadline,
                        confidence=0.92,
                        evidence_turn=filtered_text,
                        evidence_turn_index=turn_index
                    ))

                    if c_owner in self.speaker_profiles:
                        self.speaker_profiles[c_owner]['tasks'].append(syn_action)

        return info

    def get_speaker_context_profiles(self) -> List[Dict[str, Any]]:
        """Return formatted attendee context profiles."""
        profiles = []
        for name, data in self.speaker_profiles.items():
            if name in ('Speaker', 'Team', 'User', 'Unknown'):
                continue
            tasks = list(dict.fromkeys(data['tasks']))
            profiles.append({
                'name': name,
                'role': data['role'],
                'focus': data['focus'],
                'tasks_count': len(tasks),
                'tasks': tasks[:4],
                'turns': data['speaking_turns']
            })
        return profiles
