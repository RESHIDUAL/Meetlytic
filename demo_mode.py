"""
Interactive Demo Mode for Script / Transcript Evaluation.
Allows users to paste custom transcripts, test mixed conversation scripts,
verify zero-retention privacy filtering, and inspect structured meeting summaries
without requiring live physical microphones or hardware.
"""

import os
import sys
import time
from typing import List, Optional

from nlp.tokenizer import SimpleTokenizer
from nlp.classifier import ConversationClassifier
from nlp.context import ContextEngine
from nlp.extractor import ProfessionalInfoExtractor, ExtractedInfo
from privacy.modes import PrivacyConfig, PrivacyMode
from privacy.filter import PrivacyFilter
from storage.database import MeetingDatabase
from storage.summary import MeetingSummaryGenerator

class DemoMode:
    """
    Interactive Demo and Transcript Simulator.
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"

    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        self.classifier = ConversationClassifier()
        self.context_engine = ContextEngine(window_size=5)
        self.extractor = ProfessionalInfoExtractor()
        self.privacy_config = PrivacyConfig(mode=PrivacyMode.BALANCED)
        self.privacy_filter = PrivacyFilter(self.privacy_config)
        self.db = MeetingDatabase("demo_meeting_records.db")
        self.summary_generator = MeetingSummaryGenerator(self.db)

        model_path = "config/trained_model.json"
        if os.path.exists(model_path):
            self.classifier.load_model(model_path)

    def print_banner(self):
        print(f"\n{self.CYAN}{'=' * 65}{self.RESET}")
        print(f"{self.BOLD}  [MIC] PROFESSIONAL CONVERSATION FILTER - DEMO & SCRIPT MODE{self.RESET}")
        print(f"{self.GRAY}  Privacy-First Standalone Edge Intelligence Simulator{self.RESET}")
        print(f"{self.CYAN}{'=' * 65}{self.RESET}\n")

    def run(self):
        while True:
            self.print_banner()
            print(f"Current Privacy Mode: {self.GREEN}{self.BOLD}{self.privacy_config.mode}{self.RESET}\n")
            print("Select an option:")
            print("  1. Paste multi-line meeting transcript / conversation script")
            print("  2. Type individual sentences interactively (live turn-by-turn)")
            print("  3. Run built-in realistic mixed meeting simulation (15 turns)")
            print("  4. Change privacy mode (STRICT / BALANCED / CUSTOM)")
            print("  5. Run automated evaluation benchmark")
            print("  6. View latest meeting summary from database")
            print("  7. Exit demo")

            choice = input(f"\n{self.BOLD}Enter choice (1-7): {self.RESET}").strip()

            if choice == "1":
                self.handle_paste_transcript()
            elif choice == "2":
                self.handle_interactive_sentences()
            elif choice == "3":
                self.handle_builtin_simulation()
            elif choice == "4":
                self.handle_change_privacy_mode()
            elif choice == "5":
                from evaluation.benchmark import Benchmark
                print("\nRunning benchmark...\n")
                print(Benchmark().run_full_benchmark())
                input(f"\n{self.GRAY}Press Enter to return to menu...{self.RESET}")
            elif choice == "6":
                self.handle_view_summary()
            elif choice == "7":
                print("\nExiting Demo. Goodbye!")
                break
            else:
                print(f"{self.RED}Invalid selection. Please choose 1-7.{self.RESET}")
                time.sleep(1)

    def handle_paste_transcript(self):
        """Allows user to paste a multi-line conversation script."""
        print(f"\n{self.BOLD}--- PASTE TRANSCRIPT MODE ---{self.RESET}")
        print(f"{self.GRAY}Paste your conversation text below. When finished, press Enter on an empty line or type 'END':{self.RESET}\n")

        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "END" or (len(lines) > 0 and line == ""):
                    break
                lines.append(line)
            except EOFError:
                break

        full_text = "\n".join(lines).strip()
        if not full_text:
            print(f"{self.YELLOW}No text entered.{self.RESET}")
            time.sleep(1)
            return

        meeting_id = self.db.start_meeting_session(
            session_name="Pasted Transcript Analysis",
            privacy_mode=self.privacy_config.mode
        )
        self.context_engine.reset()

        sentences = self.tokenizer.split_sentences(full_text)
        print(f"\n{self.CYAN}Analyzing {len(sentences)} conversational segments...{self.RESET}\n")

        self._process_sentence_sequence(sentences, meeting_id)

    def handle_interactive_sentences(self):
        """Type turn-by-turn sentences."""
        print(f"\n{self.BOLD}--- LIVE INTERACTIVE SENTENCE MODE ---{self.RESET}")
        print(f"{self.GRAY}Type sentences one at a time. Watch context and classification adapt.{self.GRAY}")
        print(f"{self.GRAY}Type 'exit' or 'done' to return to menu and view summary.{self.RESET}\n")

        meeting_id = self.db.start_meeting_session(
            session_name="Interactive Turn Session",
            privacy_mode=self.privacy_config.mode
        )
        self.context_engine.reset()

        while True:
            sentence = input(f"\n{self.BOLD}Input Sentence: {self.RESET}").strip()
            if sentence.lower() in ('exit', 'done', 'q'):
                break
            if not sentence:
                continue

            self._process_single_turn(sentence, meeting_id, verbose=True)

        self._finalize_meeting(meeting_id)

    def handle_builtin_simulation(self):
        """Run pre-packaged realistic meeting simulation."""
        print(f"\n{self.BOLD}--- REALISTIC MIXED MEETING SIMULATION ---{self.RESET}")
        print(f"{self.GRAY}Simulating mixed project discussion with personal small-talk interludes...{self.RESET}\n")

        meeting_id = self.db.start_meeting_session(
            session_name="Simulated Standup Meeting",
            privacy_mode=self.privacy_config.mode
        )
        self.context_engine.reset()

        turns = [
            ("Lead", "Good morning team, let's start today's sprint review.", "casual"),
            ("Dev 1", "Did anyone watch the football match yesterday? It was incredible.", "casual"),
            ("Lead", "Let's focus on our sprint deliverables and architecture milestones.", "professional"),
            ("Dev 1", "The authentication module has been completed and merged to main.", "professional"),
            ("Dev 2", "I investigated the database connection pool timeout on staging.", "professional"),
            ("Dev 2", "The root cause was connection leak in the reporting service.", "professional"),
            ("Dev 1", "By the way, are we ordering pizza for lunch today?", "casual"),
            ("Lead", "Let's keep lunch plans for later. What is our status on the API gateway?", "professional"),
            ("Dev 1", "Rahul will handle the API gateway deployment by 5 PM.", "professional"),
            ("Dev 2", "The load test showed API response time reduced to 45 milliseconds.", "professional"),
            ("Lead", "We agreed to deploy the beta release to production on Friday.", "professional"),
            ("Dev 2", "Action item: Anjali will write the user documentation by tomorrow.", "professional"),
            ("Lead", "Great work team. That concludes today's sync.", "casual")
        ]
        for speaker, sentence, _ in turns:
            time.sleep(0.15)
            self._process_single_turn(sentence, meeting_id, speaker=speaker, verbose=True)

        self._finalize_meeting(meeting_id)

    def handle_change_privacy_mode(self):
        """Switch between STRICT, BALANCED, and CUSTOM."""
        print(f"\n{self.BOLD}--- CONFIGURE PRIVACY MODE ---{self.RESET}")
        print("1. STRICT   - Only very high confidence (>= 85%) professional data stored. Uncertain discarded.")
        print("2. BALANCED - Standard retention (>= 70%) for professional tasks/decisions.")
        print("3. CUSTOM   - User-defined confidence threshold.")

        c = input("\nSelect Mode (1-3): ").strip()
        if c == "1":
            self.privacy_config = PrivacyConfig(mode=PrivacyMode.STRICT)
        elif c == "2":
            self.privacy_config = PrivacyConfig(mode=PrivacyMode.BALANCED)
        elif c == "3":
            try:
                th = float(input("Enter custom professional threshold (0.50 - 0.95): "))
                self.privacy_config = PrivacyConfig(mode=PrivacyMode.CUSTOM, custom_threshold=th)
            except ValueError:
                print(f"{self.RED}Invalid input. Kept BALANCED.{self.RESET}")
                return
        self.privacy_filter = PrivacyFilter(self.privacy_config)
        print(f"\n{self.GREEN} Privacy mode updated to: {self.privacy_config.mode}{self.RESET}")
        time.sleep(1)

    def handle_view_summary(self):
        """Display generated summary."""
        meetings = self.db.get_meeting_items(1)

        summary_str = self.summary_generator.generate_summary(1)
        print("\n" + summary_str)
        input(f"\n{self.GRAY}Press Enter to continue...{self.RESET}")

    def _process_sentence_sequence(self, sentences: List[str], meeting_id: int):
        for s in sentences:
            self._process_single_turn(s, meeting_id, verbose=True)
        self._finalize_meeting(meeting_id)

    def _process_single_turn(self, sentence: str, meeting_id: int,
                             speaker: str = "Speaker A", verbose: bool = True):
        tokens = self.tokenizer.tokenize(sentence)
        boost = self.context_engine.get_context_boost(sentence, tokens)
        res = self.classifier.classify(sentence, context_boost=boost)

        classification = res['classification']
        confidence = res['confidence']
        prof_score = res['professional_score']

        filter_res = self.privacy_filter.filter_turn(
            text=sentence,
            classification=classification,
            confidence=confidence
        )

        self.context_engine.update(sentence, classification, prof_score, tokens)

        extracted = ExtractedInfo()
        if filter_res['retained']:
            extracted = self.extractor.extract(sentence)

            for p in extracted.projects:
                self.db.add_professional_item(meeting_id, 'project', p, speaker, confidence, tokens)
            for t in extracted.tasks:
                self.db.add_professional_item(meeting_id, 'task', t, speaker, confidence, tokens)
            for d in extracted.decisions:
                self.db.add_professional_item(meeting_id, 'decision', d, speaker, confidence, tokens)
            for dl in extracted.deadlines:
                self.db.add_professional_item(meeting_id, 'deadline', dl, speaker, confidence, tokens)
            for r in extracted.requirements:
                self.db.add_professional_item(meeting_id, 'requirement', r, speaker, confidence, tokens)

        if verbose:
            self._display_turn_result(speaker, sentence, res, filter_res, extracted)

    def _display_turn_result(self, speaker: str, sentence: str,
                             res: dict, filter_res: dict, extracted: ExtractedInfo):
        cls = res['classification'].upper()
        conf = res['confidence'] * 100

        if filter_res['retained']:
            badge = f"{self.GREEN}{self.BOLD}[RETAINED - {cls} {conf:4.1f}%]{self.RESET}"
        else:
            badge = f"{self.RED}{self.BOLD}[DISCARDED - {cls} {conf:4.1f}%]{self.RESET}"

        print(f"\n{self.CYAN}{speaker}:{self.RESET} \"{sentence}\"")
        print(f"  -> Decision: {badge}")

        if filter_res['retained'] and not extracted.is_empty():
            if extracted.tasks:
                print(f"    {self.GREEN}[TASK]{self.RESET} {', '.join(extracted.tasks)}")
            if extracted.decisions:
                print(f"    {self.GREEN}[DECISION]{self.RESET} {', '.join(extracted.decisions)}")
            if extracted.deadlines:
                print(f"    {self.GREEN}[DEADLINE]{self.RESET} {', '.join(extracted.deadlines)}")
            if extracted.projects:
                print(f"    {self.GREEN}[PROJECT]{self.RESET} {', '.join(extracted.projects)}")

    def _finalize_meeting(self, meeting_id: int):
        stats = self.privacy_filter.get_privacy_statistics()
        self.db.update_meeting_metrics(
            meeting_id=meeting_id,
            total_segments=stats['total_segments_processed'],
            retained_segments=stats['professional_segments_retained'],
            discarded_segments=stats['private_segments_discarded']
        )
        self.db.end_meeting_session(meeting_id)

        summary_text = self.summary_generator.generate_summary(meeting_id)
        print("\n" + summary_text)
        input(f"\n{self.GRAY}Press Enter to return to main menu...{self.RESET}")

if __name__ == "__main__":
    DemoMode().run()
