"""
Unified Application Entry Point.
Professional Conversation Filtering and Meeting Record Device.

Usage:
  python main.py                     # Start live meeting recording with default microphone
  python main.py --demo              # Interactive demo mode (paste transcripts/scripts)
  python main.py --wifi-esp          # Listen for streaming ESP8266 table hardware node
  python main.py --eval              # Run comprehensive accuracy and privacy benchmark
  python main.py --mode STRICT       # Set privacy policy (STRICT / BALANCED / CUSTOM)
  python main.py --record-template   # Enroll custom spoken keywords
  python main.py --train             # Train custom Naive Bayes classifier
"""

import os
import sys
import time
import argparse
import numpy as np

from audio.capture import AudioCapture
from audio.buffer import CircularAudioBuffer
from audio.vad import VoiceActivityDetector
from audio.segmenter import SpeechSegmenter, SpeechSegment
from speech.keyword_spotter import KeywordSpotter
from speaker.identifier import SpeakerIdentifier
from nlp.tokenizer import SimpleTokenizer
from nlp.classifier import ConversationClassifier
from nlp.context import ContextEngine
from nlp.extractor import ProfessionalInfoExtractor
from privacy.modes import PrivacyConfig, PrivacyMode
from privacy.filter import PrivacyFilter
from storage.database import MeetingDatabase
from storage.summary import MeetingSummaryGenerator
from display.console_ui import ConsoleUI

class MeetingEngine:
    """
    Main Real-Time Engine coordinating Audio -> VAD -> Speech -> NLP -> Privacy -> SQLite.
    """

    def __init__(self, privacy_mode: str = "BALANCED", use_esp_wifi: bool = False):
        self.privacy_mode = privacy_mode
        self.use_esp_wifi = use_esp_wifi

        self.privacy_config = PrivacyConfig(mode=privacy_mode)
        self.privacy_filter = PrivacyFilter(self.privacy_config)
        self.buffer = CircularAudioBuffer(duration_sec=30, sample_rate=16000)
        self.vad = VoiceActivityDetector(sample_rate=16000)
        self.segmenter = SpeechSegmenter(sample_rate=16000)
        self.keyword_spotter = KeywordSpotter(sample_rate=16000)
        self.speaker_id = SpeakerIdentifier(sample_rate=16000)
        self.tokenizer = SimpleTokenizer()
        self.classifier = ConversationClassifier()
        self.context_engine = ContextEngine(window_size=5)
        self.extractor = ProfessionalInfoExtractor()
        self.db = MeetingDatabase("meeting_records.db")
        self.summary_gen = MeetingSummaryGenerator(self.db)
        self.ui = ConsoleUI()

        model_path = "config/trained_model.json"
        if os.path.exists(model_path):
            self.classifier.load_model(model_path)

        if self.use_esp_wifi:
            from hardware.wifi_bridge import WiFiHardwareBridge
            self.audio_source = WiFiHardwareBridge()
        else:
            self.audio_source = AudioCapture(sample_rate=16000)

        self.meeting_id: Optional[int] = None
        self.is_running: bool = False

    def start(self):
        """Start meeting session."""
        self.meeting_id = self.db.start_meeting_session(
            session_name="Live Table Meeting",
            privacy_mode=self.privacy_mode
        )
        self.ui.update_metrics(privacy_mode=self.privacy_mode, status="LISTENING")

        if self.use_esp_wifi:
            print("[Engine] Starting ESP8266 WiFi Server on port 5555...")
            self.audio_source.start_server()
        else:
            print("[Engine] Starting local microphone stream...")
            self.audio_source.start()

        self.is_running = True
        self._run_loop()

    def _run_loop(self):
        """Main real-time audio and DSP processing loop."""
        stored_items = 0
        discarded_units = 0

        try:
            while self.is_running:
                if self.use_esp_wifi:
                    chunk = self.audio_source.get_audio_chunk(timeout=0.2)
                else:
                    chunk = self.audio_source.get_chunk(timeout=0.2)

                if chunk is None:
                    continue

                self.buffer.write(chunk)
                rms = AudioCapture.calculate_rms(chunk) if not self.use_esp_wifi else float(np.sqrt(np.mean((chunk / 32768.0)**2)))
                self.ui.update_metrics(signal_rms=rms)

                vad_decision = self.vad.is_frame_active(chunk.astype(np.float32) / 32768.0)
                self.segmenter.feed_audio_and_vad(chunk, is_speech=vad_decision)

                completed_segments = self.segmenter.get_completed_segments()
                for seg in completed_segments:
                    self.ui.update_metrics(status="ANALYZING")
                    self.ui.render_dashboard()

                    speaker = self.speaker_id.identify_speaker(seg.audio_data)
                    self.ui.update_metrics(active_speaker=speaker)

                    spotted = self.keyword_spotter.spot_keywords(seg.audio_data)
                    spotted_terms = [s['keyword'] for s in spotted]

                    if spotted_terms:
                        turn_text = " ".join(spotted_terms)
                    else:
                        turn_text = "meeting project discussion"

                    tokens = self.tokenizer.tokenize(turn_text)
                    boost = self.context_engine.get_context_boost(turn_text, tokens)
                    res = self.classifier.classify(turn_text, context_boost=boost)

                    classification = res['classification']
                    confidence = res['confidence']

                    filter_res = self.privacy_filter.filter_turn(
                        text=turn_text,
                        classification=classification,
                        confidence=confidence,
                        audio_buffer=self.buffer,
                        audio_duration_sec=seg.duration_sec
                    )

                    if filter_res['retained']:
                        stored_items += 1
                        self.ui.update_metrics(status="PROFESSIONAL", prof_confidence=confidence)
                        if self.use_esp_wifi:
                            self.audio_source.send_hardware_feedback("PROFESSIONAL", int(confidence * 100))

                        extracted = self.extractor.extract(turn_text)
                        for t in extracted.tasks:
                            self.db.add_professional_item(self.meeting_id, 'task', t, speaker, confidence, tokens)
                        for d in extracted.decisions:
                            self.db.add_professional_item(self.meeting_id, 'decision', d, speaker, confidence, tokens)
                        for dl in extracted.deadlines:
                            self.db.add_professional_item(self.meeting_id, 'deadline', dl, speaker, confidence, tokens)
                        for p in extracted.projects:
                            self.db.add_professional_item(self.meeting_id, 'project', p, speaker, confidence, tokens)

                    else:
                        discarded_units += 1
                        self.ui.update_metrics(status="PERSONAL / DISCARDED", prof_confidence=0.0)
                        if self.use_esp_wifi:
                            self.audio_source.send_hardware_feedback("DISCARD")

                    self.context_engine.update(turn_text, classification, res['professional_score'], tokens)
                    self.ui.update_metrics(
                        stored_items_count=stored_items,
                        discarded_count=discarded_units,
                        current_topic=self.context_engine.get_conversation_state()['current_topic']
                    )

                self.ui.render_dashboard()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop meeting, wipe buffer, and generate final summary."""
        self.is_running = False
        print("\n\n[Engine] Stopping meeting session and generating report...")

        if self.use_esp_wifi:
            self.audio_source.stop()
        else:
            self.audio_source.stop()

        self.buffer.secure_wipe_all()

        if self.meeting_id:
            stats = self.privacy_filter.get_privacy_statistics()
            self.db.update_meeting_metrics(
                meeting_id=self.meeting_id,
                total_segments=stats['total_segments_processed'],
                retained_segments=stats['professional_segments_retained'],
                discarded_segments=stats['private_segments_discarded']
            )
            self.db.end_meeting_session(self.meeting_id)

            summary = self.summary_gen.generate_summary(self.meeting_id)
            print("\n" + summary)

def main():
    parser = argparse.ArgumentParser(description="Professional Conversation Filtering & Meeting Record Device")
    parser.add_argument("--demo", action="store_true", help="Launch interactive demo mode for scripts and transcripts")
    parser.add_argument("--wifi-esp", action="store_true", help="Stream audio from ESP8266 WiFi hardware table node")
    parser.add_argument("--eval", action="store_true", help="Run comprehensive evaluation benchmark suite")
    parser.add_argument("--mode", choices=["STRICT", "BALANCED", "CUSTOM"], default="BALANCED", help="Privacy enforcement policy")
    parser.add_argument("--record-template", action="store_true", help="Record new spoken keyword templates")
    parser.add_argument("--train", action="store_true", help="Train custom Naive Bayes classifier")

    args = parser.parse_args()

    if args.demo:
        from demo_mode import DemoMode
        DemoMode().run()
    elif args.eval:
        from evaluation.benchmark import Benchmark
        print(Benchmark().run_full_benchmark())
    elif args.record_template:
        from tools.record_template import record_keyword_session
        record_keyword_session()
    elif args.train:
        from tools.train_classifier import main as train_main
        train_main()
    else:
        engine = MeetingEngine(privacy_mode=args.mode, use_esp_wifi=args.wifi_esp)
        engine.start()

if __name__ == "__main__":
    main()
