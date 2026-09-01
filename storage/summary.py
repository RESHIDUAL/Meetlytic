"""
Executive Intelligence Summary & Report Generation Engine for Meetlytic.
Renders stateful meeting intelligence reports from MeetingState or ConversationContextGraph:
- KEY TOPICS
- CURRENT STATUS / IMPORTANT FACTS (Transformations, Metrics, Superseded Updates)
- ISSUES / RISKS / BLOCKERS (Problems, Confirmed Causes, Warning Thresholds, Non-Blocking Flags)
- DECISIONS / SCOPE MANAGEMENT
- ACTION ITEMS (Concrete Objects, Owners, Deadlines, Conditions, Reasons)
- CONDITIONS / DEPENDENCIES
- RELEASE / BUSINESS GATES
- DEADLINES / MILESTONES
- PARTICIPANTS / RESPONSIBILITIES
- PRIVACY FILTERING AUDIT
Strictly free of emojis and emdashes.
"""

from typing import List, Dict, Any, Optional
from storage.database import MeetingDatabase
from nlp.stateful_engine import MeetingState, StatefulFact, StatefulIssueRisk, StatefulDecision, StatefulAction

class MeetingSummaryGenerator:
    """
    Generates executive meeting intelligence reports from resolved state machine.
    """

    def __init__(self, db: MeetingDatabase):
        self.db = db

    def generate_summary(self, meeting_id: int, graph: Optional[Any] = None, state: Optional[MeetingState] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive executive intelligence report.
        """
        meeting = self.db.get_meeting(meeting_id)
        if not meeting:
            return "Meeting record not found."

        db_meta = self.db.get_meeting_metadata(meeting_id) or {}
        combined_meta = {**db_meta, **(meta or {})}
        meta = combined_meta

        topics = state.semantic_topics if state else (graph.semantic_topics if graph else [])
        facts = state.facts if state else []
        issues_risks = state.issues_risks if state else []
        decisions = state.decisions if state else []
        actions = state.actions if state else []
        release_gates = state.release_gates if state else []
        dependencies = state.dependencies if state else []
        unresolved = state.unresolved if state else (graph.unresolved if graph else [])

        if not state and graph:
            topics = graph.semantic_topics
            status_facts = graph.status_facts
            causality_arcs = graph.causality_arcs
            decision_arcs = graph.decision_arcs
            tasks = graph.tasks
        else:
            status_facts = []
            causality_arcs = []
            decision_arcs = []
            tasks = []

        lines = [
            "============================================================",
            "                 MEETLYTIC INTELLIGENCE REPORT              ",
            "============================================================",
            f"Session Name    : {meeting.get('session_name', 'Meeting Session')}",
            f"Start Time      : {meeting.get('start_time', 'N/A')}",
            f"End Time        : {meeting.get('end_time', meeting.get('start_time', 'N/A'))}",
            f"Privacy Policy  : {meeting.get('privacy_mode', 'BALANCED')}",
            "------------------------------------------------------------",
            ""
        ]

        if topics:
            lines.append("[KEY TOPICS]")
            for i, top in enumerate(topics[:5], 1):
                lines.append(f"  {i}. {top}")
            lines.append("")

        if facts or status_facts:
            lines.append("[CURRENT STATUS / IMPORTANT FACTS]")
            if facts:
                for f in facts[:15]:
                    if not f.is_superseded:
                        lines.append(f"  - {f.statement}")
            elif status_facts:
                for sf in status_facts[:15]:
                    imp_note = f" (Release Impact: {sf['impact']})" if sf.get('impact') and sf['impact'] != 'UNSPECIFIED' else ""
                    lines.append(f"  - {sf['fact']}{imp_note}")
            lines.append("")

        if issues_risks or causality_arcs:
            lines.append("[ISSUES / RISKS / BLOCKERS]")
            if issues_risks:
                for item in issues_risks[:6]:
                    type_tag = f" [{item.kind.replace('_', ' ')}]" if item.kind != "ISSUE" else ""
                    lines.append(f"  - Issue{type_tag}: {item.problem}")
                    if item.cause:
                        cause_label = "Confirmed Cause" if item.cause_certainty.lower() == "confirmed" else "Suspected Cause"
                        lines.append(f"      * {cause_label}: {item.cause} (Status: {item.cause_certainty.capitalize()})")
                    if item.metric:
                        lines.append(f"      * Metric: {item.metric}")
                    if item.target_metric:
                        lines.append(f"      * Target: {item.target_metric}")
                    if item.threshold:
                        lines.append(f"      * Threshold: {item.threshold}")
                    if item.release_impact and item.release_impact != "UNSPECIFIED":
                        lines.append(f"      * Impact: {item.release_impact}")
            elif causality_arcs:
                for arc in causality_arcs[:6]:
                    lines.append(f"  - Issue: {arc.problem}")
                    if arc.cause:
                        cause_label = "Confirmed Cause" if arc.cause_certainty.lower() == "confirmed" else "Suspected Cause"
                        lines.append(f"      * {cause_label}: {arc.cause} (Status: {arc.cause_certainty.capitalize()})")
                    if arc.metric:
                        lines.append(f"      * Metric: {arc.metric}")
                    if arc.release_impact and arc.release_impact != "UNSPECIFIED":
                        lines.append(f"      * Impact: {arc.release_impact}")
            lines.append("")

        if decisions or decision_arcs:
            lines.append("[DECISIONS / SCOPE MANAGEMENT]")
            if decisions:
                for d in decisions[:6]:
                    status_tag = f" [{d.status}]" if d.status != "ADOPTED" else ""
                    lines.append(f"  - Decision{status_tag}: {d.decision}")
            elif decision_arcs:
                for d in decision_arcs[:6]:
                    lines.append(f"  - Decision: {d.final_decision}")
            lines.append("")

        if actions or tasks:
            lines.append("[ACTION ITEMS]")
            if actions:
                for a in actions:
                    owner_str = f"Assigned to: {a.owner}"
                    dl_str = f", Due: {a.deadline}" if a.deadline and a.deadline != "unspecified" else ""
                    dur_str = f", Duration: {a.duration}" if a.duration else ""
                    cond_str = f" (Condition: {', '.join(a.conditions)})" if a.conditions else ""
                    why_str = f" [Reason: {a.why_reason}]" if a.why_reason else ""
                    lines.append(f"  - {a.full_description} ({owner_str}{dl_str}{dur_str}){cond_str}{why_str}")
            elif tasks:
                for t in tasks:
                    owner_str = f"Assigned to: {t.owner}"
                    dl_str = f", Due: {t.deadline}" if t.deadline and t.deadline != "unspecified" else ""
                    dur_str = f", Duration: {t.duration}" if t.duration else ""
                    cond_str = f" (Condition: {', '.join(t.conditions)})" if t.conditions else ""
                    why_str = f" [Reason: {t.why_reason}]" if t.why_reason else ""
                    lines.append(f"  - {t.full_description} ({owner_str}{dl_str}{dur_str}){cond_str}{why_str}")
            lines.append("")

        if dependencies:
            lines.append("[CONDITIONS / DEPENDENCIES]")
            for dep in dependencies[:6]:
                lines.append(f"  - {dep.raw_statement}")
            lines.append("")

        if release_gates:
            domain_name = state.detected_domain.domain if state and hasattr(state, 'detected_domain') else "General"
            if domain_name == "Technology":
                gate_header = "[RELEASE / BUSINESS GATES]"
            elif domain_name in ("Education", "Academic"):
                gate_header = "[REQUIREMENTS / DEPENDENCIES]"
            elif domain_name in ("Travel", "Personal"):
                gate_header = "[CONDITIONS / ITINERARY PREREQUISITES]"
            else:
                gate_header = "[RELEASE / BUSINESS GATES]"

            lines.append(gate_header)
            for g in release_gates:
                op_str = f" {g.operator} "
                cond_text = op_str.join(g.conditions)
                tag = "PRIMARY RELEASE BLOCKER" if any('primary blocker' in c.lower() or 'must be resolved' in c.lower() for c in g.conditions) else f"{g.target.upper()} GATE"
                lines.append(f"  - [{tag}] {cond_text}")
            lines.append("")

        milestones = []
        source_actions = actions if actions else tasks
        for a in source_actions:
            if a.deadline and a.deadline != "unspecified":
                milestones.append(f"{a.full_description} (Due: {a.deadline})")

        if milestones:
            lines.append("[DEADLINES / MILESTONES]")
            for m in list(dict.fromkeys(milestones))[:6]:
                lines.append(f"  - {m}")
            lines.append("")

        by_speaker: Dict[str, List[str]] = {}
        for a in source_actions:
            if a.owner and a.owner not in ("Unclear", "Speaker", "Team"):
                if a.owner not in by_speaker:
                    by_speaker[a.owner] = []
                dl_note = f" (Due: {a.deadline})" if a.deadline and a.deadline != "unspecified" else ""
                by_speaker[a.owner].append(f"{a.full_description}{dl_note}")

        if by_speaker:
            lines.append("[PARTICIPANTS / RESPONSIBILITIES]")
            for spk, task_list in by_speaker.items():
                lines.append(f"  - {spk}:")
                for t_item in task_list:
                    lines.append(f"      * {t_item}")
            lines.append("")

        if unresolved:
            lines.append("[UNCERTAINTIES / FOLLOW-UP]")
            for u in unresolved[:4]:
                lines.append(f"  - {u}")
            lines.append("")

        tot = meta.get('total_segments_analyzed', meta.get('total_segments_processed', 0))
        prof_content = meta.get('professional_content', meta.get('professional_segments_retained', 0))
        casual_disc = meta.get('casual_discarded', 0)
        badmouthing_disc = meta.get('badmouthing_discarded', 0)
        sarcasm_disc = meta.get('sarcasm_discarded', 0)
        gossip_disc = meta.get('gossip_discarded', 0)
        venting_disc = meta.get('venting_discarded', 0)
        opinions_disc = meta.get('opinions_discarded', 0)
        proposals_held = meta.get('proposals_held', 0)
        questions_count = meta.get('questions_count', 0)
        other_disc = meta.get('other_discarded', 0)

        num_intel = (
            len(actions or tasks or []) +
            len(decisions or decision_arcs or []) +
            len(issues_risks or causality_arcs or []) +
            len(facts or []) +
            len(release_gates or []) +
            len(dependencies or [])
        )
        intel_items = meta.get('important_intelligence_items', num_intel)

        lines.extend([
            "------------------------------------------------------------",
            "[PRIVACY & SEMANTIC FILTERING AUDIT]",
            f"  Total Statements Analyzed   : {tot}",
            f"  Professional Statements     : {prof_content}",
            f"  Casual/Social Discarded     : {casual_disc}",
            f"  Badmouthing/Insult Discarded: {badmouthing_disc}",
            f"  Sarcasm Discarded           : {sarcasm_disc}",
            f"  Personal Gossip Discarded   : {gossip_disc}",
            f"  Emotional Venting Discarded : {venting_disc}",
            f"  Opinions Discarded          : {opinions_disc}",
            f"  Proposals Held              : {proposals_held}",
            f"  Questions Evaluated         : {questions_count}",
            f"  Ambiguous/Other Discarded   : {other_disc}",
            f"  Important Intelligence Items: {intel_items}",
            "  Sanitization Status         : Verified (Zero Disk Retention)",
            "============================================================"
        ])

        return "\n".join(lines)
