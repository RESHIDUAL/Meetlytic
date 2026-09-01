"""
Sticky Note Synthesis & Generation Engine for Meetlytic.
Transforms discourse context graph into clean, categorized post-it notes.
Strictly free of emojis and emdashes.
"""

import random
from typing import List, Dict, Any, Optional
from nlp.context_graph import ConversationContextGraph
from nlp.stateful_engine import MeetingState

class StickyNoteGenerator:
    """
    Synthesizes context graph or meeting state into categorized Corkboard Sticky Notes.
    """

    NOTE_COLORS = {
        'action': {'bg': '#fff740', 'border': '#f5ee38', 'text': '#2b2b2b', 'tag': 'TASKS'},
        'decision': {'bg': '#b2f2bb', 'border': '#8ce99a', 'text': '#1b4324', 'tag': 'DECISIONS'},
        'deadline': {'bg': '#a5d8ff', 'border': '#74c0fc', 'text': '#183c59', 'tag': 'SCHEDULE'},
        'issue': {'bg': '#ffc9c9', 'border': '#ffa8a8', 'text': '#4a1515', 'tag': 'ISSUES & RISKS'},
        'status': {'bg': '#d0ebff', 'border': '#74c0fc', 'text': '#183c59', 'tag': 'STATUS & FACTS'},
        'spec': {'bg': '#e599f7', 'border': '#da77f2', 'text': '#3b1046', 'tag': 'RELEASE GATES'},
        'takeaway': {'bg': '#ffe066', 'border': '#fcc419', 'text': '#493700', 'tag': 'TOPICS'},
        'speaker': {'bg': '#ffec99', 'border': '#ffd43b', 'text': '#493700', 'tag': 'SPEAKER'}
    }

    PIN_COLORS = ['pin-red', 'pin-blue', 'pin-yellow', 'pin-wood', 'pin-green']

    def generate_from_state(self, state: MeetingState) -> List[Dict[str, Any]]:
        """
        Generate visual sticky notes directly from MeetingState.
        """
        notes: List[Dict[str, Any]] = []
        note_id = 1

        if state.actions:
            task_bullets = []
            for a in state.actions[:6]:
                owner_prefix = f"**{a.owner}**: " if a.owner and a.owner != "Unclear" else ""
                dl_suffix = f" (Due: {a.deadline})" if a.deadline and a.deadline != "unspecified" else ""
                cond_suffix = f" [Condition: {', '.join(a.conditions)}]" if a.conditions else ""
                why_suffix = f" (Why: {a.why_reason})" if a.why_reason else ""
                task_bullets.append(f"{owner_prefix}{a.full_description}{dl_suffix}{cond_suffix}{why_suffix}")

            notes.append(self._create_note(
                note_id=note_id,
                category='action',
                title='Action Items and Tasks',
                items=task_bullets,
                pin_color=random.choice(self.PIN_COLORS),
                rotation=random.uniform(-2.5, 2.5)
            ))
            note_id += 1

            for spk_name, spk_data in list(state.participants.items())[:3]:
                if spk_data.get('actions'):
                    card_bullets = [f"**Role**: Contributor", f"**Speaking Turns**: {spk_data['turns']}"]
                    for a_item in spk_data['actions'][:3]:
                        card_bullets.append(f"Task: {a_item}")

                    notes.append(self._create_note(
                        note_id=note_id,
                        category='speaker',
                        title=f"{spk_name}'s Deliverables",
                        items=card_bullets,
                        pin_color=random.choice(self.PIN_COLORS),
                        rotation=random.uniform(-2.5, 2.5)
                    ))
                    note_id += 1

        if state.decisions:
            dec_bullets = [d.decision for d in state.decisions[:5]]
            notes.append(self._create_note(
                note_id=note_id,
                category='decision',
                title='Key Decisions and Scope',
                items=dec_bullets,
                pin_color='pin-green',
                rotation=random.uniform(-2.0, 2.0)
            ))
            note_id += 1

        if state.issues_risks:
            issue_bullets = []
            for item in state.issues_risks[:5]:
                if item.cause:
                    c_type = "Confirmed Cause" if item.cause_certainty.lower() == "confirmed" else "Suspected Cause"
                    cause_txt = f" ({c_type}: {item.cause})"
                else:
                    cause_txt = ""
                metric_txt = f" [Metric: {item.metric}]" if item.metric else ""
                impact_txt = f" [{item.release_impact}]" if item.release_impact != "UNSPECIFIED" else ""
                issue_bullets.append(f"{item.problem}{cause_txt}{metric_txt}{impact_txt}")

            notes.append(self._create_note(
                note_id=note_id,
                category='issue',
                title='Issues, Risks & Blockers',
                items=issue_bullets,
                pin_color='pin-red',
                rotation=random.uniform(-2.0, 2.0)
            ))
            note_id += 1

        if state.facts:
            status_bullets = [f.statement for f in state.facts[:5] if not f.is_superseded]
            if status_bullets:
                notes.append(self._create_note(
                    note_id=note_id,
                    category='status',
                    title='Current Status & Metrics',
                    items=status_bullets,
                    pin_color='pin-blue',
                    rotation=random.uniform(-2.0, 2.0)
                ))
                note_id += 1

        gate_bullets = []
        for gate in state.release_gates:
            gate_bullets.append(f"[{gate.target.upper()}] {' AND '.join(gate.conditions)}")
        for dep in state.dependencies:
            gate_bullets.append(f"[DEPENDENCY] {dep.raw_statement}")

        if gate_bullets:
            notes.append(self._create_note(
                note_id=note_id,
                category='spec',
                title='Release Gates & Dependencies',
                items=gate_bullets[:5],
                pin_color='pin-wood',
                rotation=random.uniform(-2.5, 2.5)
            ))
            note_id += 1

        milestones = [f"{a.full_description} (Due: {a.deadline})" for a in state.actions if a.deadline and a.deadline != "unspecified"]
        if milestones:
            notes.append(self._create_note(
                note_id=note_id,
                category='deadline',
                title='Deadlines and Schedules',
                items=milestones[:5],
                pin_color='pin-blue',
                rotation=random.uniform(-2.5, 2.5)
            ))
            note_id += 1

        if state.semantic_topics:
            notes.append(self._create_note(
                note_id=note_id,
                category='takeaway',
                title='Key Domain Topics',
                items=state.semantic_topics[:4],
                pin_color='pin-yellow',
                rotation=random.uniform(-1.5, 1.5)
            ))
            note_id += 1

        return notes

    def generate_from_context_graph(self, graph: ConversationContextGraph) -> List[Dict[str, Any]]:
        """
        Generate visual sticky notes directly from ConversationContextGraph.
        """
        notes: List[Dict[str, Any]] = []
        note_id = 1

        if graph.tasks:
            task_bullets = []
            for t in graph.tasks[:6]:
                owner_prefix = f"**{t.owner}**: " if t.owner and t.owner != "Unclear" else ""
                dl_suffix = f" (Due: {t.deadline})" if t.deadline and t.deadline != "unspecified" else ""
                cond_suffix = f" [Condition: {', '.join(t.conditions)}]" if t.conditions else ""
                task_bullets.append(f"{owner_prefix}{t.full_description}{dl_suffix}{cond_suffix}")

            notes.append(self._create_note(
                note_id=note_id,
                category='action',
                title='Action Items and Tasks',
                items=task_bullets,
                pin_color=random.choice(self.PIN_COLORS),
                rotation=random.uniform(-2.5, 2.5)
            ))
            note_id += 1

            for spk_name, spk_data in list(graph.participants.items())[:3]:
                if spk_data.get('actions'):
                    card_bullets = [f"**Role**: Contributor", f"**Speaking Turns**: {spk_data['turns']}"]
                    for a in spk_data['actions'][:3]:
                        card_bullets.append(f"Task: {a}")

                    notes.append(self._create_note(
                        note_id=note_id,
                        category='speaker',
                        title=f"{spk_name}'s Deliverables",
                        items=card_bullets,
                        pin_color=random.choice(self.PIN_COLORS),
                        rotation=random.uniform(-2.5, 2.5)
                    ))
                    note_id += 1

        if graph.decision_arcs:
            dec_bullets = [d.final_decision for d in graph.decision_arcs[:5]]
            notes.append(self._create_note(
                note_id=note_id,
                category='decision',
                title='Key Decisions and Agreements',
                items=dec_bullets,
                pin_color='pin-green',
                rotation=random.uniform(-2.0, 2.0)
            ))
            note_id += 1

        if graph.causality_arcs:
            issue_bullets = []
            for arc in graph.causality_arcs[:5]:
                cause_txt = f" (Cause: {arc.cause})" if arc.cause else ""
                impact_txt = f" [{arc.release_impact}]" if arc.release_impact != "UNSPECIFIED" else ""
                issue_bullets.append(f"{arc.problem}{cause_txt}{impact_txt}")

            notes.append(self._create_note(
                note_id=note_id,
                category='issue',
                title='Issues, Risks & Blockers',
                items=issue_bullets,
                pin_color='pin-red',
                rotation=random.uniform(-2.0, 2.0)
            ))
            note_id += 1

        if graph.status_facts:
            status_bullets = [sf['fact'] for sf in graph.status_facts[:5]]
            notes.append(self._create_note(
                note_id=note_id,
                category='status',
                title='Status & Technical Facts',
                items=status_bullets,
                pin_color='pin-blue',
                rotation=random.uniform(-2.0, 2.0)
            ))
            note_id += 1

        gate_bullets = []
        for gate in graph.release_gates:
            gate_bullets.append(f"[{gate.target.upper()}] {' AND '.join(gate.conditions)}")
        for req in graph.requirements:
            if req.get('kind') != 'RELEASE_GATE':
                gate_bullets.append(f"[{req['kind']}] {req['statement']}")

        if gate_bullets:
            notes.append(self._create_note(
                note_id=note_id,
                category='spec',
                title='Release Gates & Conditions',
                items=gate_bullets[:5],
                pin_color='pin-wood',
                rotation=random.uniform(-2.5, 2.5)
            ))
            note_id += 1

        milestones = [f"{t.full_description} (Due: {t.deadline})" for t in graph.tasks if t.deadline and t.deadline != "unspecified"]
        if milestones:
            notes.append(self._create_note(
                note_id=note_id,
                category='deadline',
                title='Deadlines and Schedules',
                items=milestones[:5],
                pin_color='pin-blue',
                rotation=random.uniform(-2.5, 2.5)
            ))
            note_id += 1

        if graph.semantic_topics:
            notes.append(self._create_note(
                note_id=note_id,
                category='takeaway',
                title='Key Agenda Topics',
                items=graph.semantic_topics[:4],
                pin_color='pin-yellow',
                rotation=random.uniform(-1.5, 1.5)
            ))
            note_id += 1

        return notes

    def _create_note(self, note_id: int, category: str, title: str, items: List[str],
                     pin_color: str, rotation: float) -> Dict[str, Any]:
        """Format a single note card."""
        style = self.NOTE_COLORS.get(category, self.NOTE_COLORS['takeaway'])
        return {
            'id': note_id,
            'category': category,
            'tag': style['tag'],
            'title': title,
            'items': items,
            'bg_color': style['bg'],
            'border_color': style['border'],
            'text_color': style['text'],
            'pin_color': pin_color,
            'rotation': round(rotation, 1)
        }
