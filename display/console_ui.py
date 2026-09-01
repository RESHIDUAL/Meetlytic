"""
Console / Terminal User Interface.
Renders real-time meeting dashboard with color-coded status, signal VU-meter,
privacy indicators, and live extraction counters using ANSI escape codes.
"""

import sys
import time
from typing import Optional

class ConsoleUI:
    """
    Terminal OLED-style UI dashboard.
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
        self.status: str = "IDLE"
        self.signal_rms: float = 0.0
        self.prof_confidence: float = 0.0
        self.current_topic: str = "Idle / Waiting"
        self.action_items_count: int = 0
        self.decisions_count: int = 0
        self.deadlines_count: int = 0
        self.stored_items_count: int = 0
        self.discarded_count: int = 0
        self.privacy_mode: str = "BALANCED"
        self.active_speaker: str = "Speaker A"
        self.start_time: float = time.time()

    def update_metrics(self, **kwargs):
        """Update any UI state metrics."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _render_signal_bar(self) -> str:
        """Create visual signal intensity VU-meter."""
        bars = int(min(10, self.signal_rms * 120))
        filled = "#" * bars
        empty = "-" * (10 - bars)
        return f"{self.CYAN}{filled}{self.GRAY}{empty}{self.RESET}"

    def _status_badge(self) -> str:
        """Render colored status badge."""
        st = self.status.upper()
        if "PROFESSIONAL" in st:
            return f"{self.GREEN}{self.BOLD}[PROFESSIONAL DETECTED]{self.RESET}"
        elif "PERSONAL" in st or "DISCARD" in st:
            return f"{self.RED}{self.BOLD}[PRIVATE - DISCARDING]{self.RESET}"
        elif "ANALYZING" in st:
            return f"{self.YELLOW}{self.BOLD}[ANALYZING TURN]{self.RESET}"
        elif "SAVING" in st:
            return f"{self.BLUE}{self.BOLD}[SAVING INTELLIGENCE]{self.RESET}"
        elif "LISTENING" in st:
            return f"{self.GREEN}[LISTENING]{self.RESET}"
        return f"{self.GRAY}[IDLE]{self.RESET}"

    def render_dashboard(self):
        """Render complete terminal dashboard frame."""
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

        sys.stdout.write("\033[H\033[J")

        box = [
            f"{self.GRAY}+--------------------------------------------------------+{self.RESET}",
            f"{self.GRAY}|{self.RESET}  {self.BOLD}[MIC] STANDALONE MEETING RECORD & PRIVACY FILTER{self.RESET}      {self.GRAY}|{self.RESET}",
            f"{self.GRAY}+--------------------------------------------------------+{self.RESET}",
            f"{self.GRAY}|{self.RESET}  Status      : {self._status_badge():<44} {self.GRAY}|{self.RESET}",
            f"{self.GRAY}|{self.RESET}  Audio Signal: [{self._render_signal_bar()}]  Speaker: {self.CYAN}{self.active_speaker:<10}{self.RESET} {self.GRAY}|{self.RESET}",
            f"{self.GRAY}|{self.RESET}  Time Active : {time_str:<12} Privacy Mode: {self.BOLD}{self.privacy_mode:<10}{self.RESET} {self.GRAY}|{self.RESET}",
            f"{self.GRAY}+--------------------------------------------------------+{self.RESET}",
            f"{self.GRAY}|{self.RESET}  {self.BOLD}LIVE INTELLIGENCE METRICS:{self.RESET}                             {self.GRAY}|{self.RESET}",
            f"{self.GRAY}|{self.RESET}  Current Topic    : {self.CYAN}{self.current_topic[:30]:<30}{self.RESET}     {self.GRAY}|{self.RESET}",
            f"{self.GRAY}|{self.RESET}  Professional Prob: {self.GREEN}{self.prof_confidence * 100:5.1f}%{self.RESET}                               {self.GRAY}|{self.RESET}",
            f"{self.GRAY}|{self.RESET}  Action Items     : {self.BOLD}{self.action_items_count:<4}{self.RESET}        Decisions: {self.BOLD}{self.decisions_count:<4}{self.RESET}       {self.GRAY}|{self.RESET}",
            f"{self.GRAY}|{self.RESET}  Deadlines Logged : {self.BOLD}{self.deadlines_count:<4}{self.RESET}        Stored   : {self.GREEN}{self.stored_items_count:<4} items{self.RESET}  {self.GRAY}|{self.RESET}",
            f"{self.GRAY}|{self.RESET}  Private Discarded: {self.RED}{self.discarded_count:<4} units{self.RESET}  (Zero Retention)       {self.GRAY}|{self.RESET}",
            f"{self.GRAY}+--------------------------------------------------------+{self.RESET}",
            f"{self.GRAY}|{self.RESET}  {self.GRAY}Shortcuts: [Ctrl+C] Stop & Generate Summary            |{self.RESET}",
            f"{self.GRAY}+--------------------------------------------------------+{self.RESET}"
        ]

        print("\n".join(box))
        sys.stdout.flush()

    def print_log_event(self, text: str, classification: str, confidence: float):
        """Print a scrolling event log beneath the dashboard."""
        if classification == "professional":
            color = self.GREEN
            tag = "PROF"
        elif classification == "personal":
            color = self.RED
            tag = "PRIV"
        elif classification == "casual":
            color = self.YELLOW
            tag = "CASL"
        else:
            color = self.GRAY
            tag = "UNCT"

        print(f"[{color}{tag}{self.RESET}] ({confidence*100:4.1f}%) {text[:75]}")
        sys.stdout.flush()
