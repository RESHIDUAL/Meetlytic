"""
Core Data Models and Intelligence Classification Types for Meetlytic.
Supports multi-type professional intelligence extraction:
ACTION, DECISION, AGREEMENT, STATUS, ISSUE, RISK, REQUIREMENT,
DEPENDENCY, BLOCKER, CONDITION, CONSTRAINT, DEADLINE, MILESTONE,
METRIC, SCOPE, STAKEHOLDER_REQUEST, CONTEXT, TECHNICAL_FACT, DEFERRED_WORK.
Strictly free of emojis and emdashes.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

class IntelligenceCategory(str, Enum):
    ACTION = "ACTION"
    DECISION = "DECISION"
    AGREEMENT = "AGREEMENT"
    STATUS = "STATUS"
    ISSUE = "ISSUE"
    RISK = "RISK"
    REQUIREMENT = "REQUIREMENT"
    DEPENDENCY = "DEPENDENCY"
    BLOCKER = "BLOCKER"
    CONDITION = "CONDITION"
    CONSTRAINT = "CONSTRAINT"
    DEADLINE = "DEADLINE"
    MILESTONE = "MILESTONE"
    METRIC = "METRIC"
    SCOPE = "SCOPE"
    STAKEHOLDER_REQUEST = "STAKEHOLDER_REQUEST"
    CONTEXT = "CONTEXT"
    TECHNICAL_FACT = "TECHNICAL_FACT"
    DEFERRED_WORK = "DEFERRED_WORK"

@dataclass
class IntelligenceItem:
    """A discrete unit of extracted professional intelligence with grounding evidence."""
    category: IntelligenceCategory
    content: str
    speaker: str = "Unspecified"
    target_owner: str = "Unclear"
    deadline: Optional[str] = None
    confidence: float = 1.0
    evidence_turn: str = ""
    evidence_turn_index: int = -1
    conditions: List[str] = field(default_factory=list)
    related_context: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    causality_link: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category.value,
            'content': self.content,
            'speaker': self.speaker,
            'target_owner': self.target_owner,
            'deadline': self.deadline or "unspecified",
            'confidence': round(self.confidence, 3),
            'evidence_turn': self.evidence_turn,
            'evidence_turn_index': self.evidence_turn_index,
            'conditions': self.conditions,
            'related_context': self.related_context,
            'metrics': self.metrics,
            'causality_link': self.causality_link
        }

@dataclass
class MeetingSessionIntelligence:
    """Isolated intelligence state for a specific meeting analysis session."""
    session_id: str
    meeting_name: str
    items: List[IntelligenceItem] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    speaker_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_turns: int = 0
    professional_turns: int = 0
    casual_turns: int = 0
    negative_venting_turns: int = 0
    discarded_segments: int = 0
    retained_segments: int = 0
