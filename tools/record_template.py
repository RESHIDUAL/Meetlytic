"""
Interactive CLI utility for recording and enrolling custom spoken keyword templates.
Extracts 13-dimensional MFCC vectors from microphone audio and saves them into speech/templates/.
Allows the user to teach the device new professional or personal vocabulary without cloud models.
"""

import os
import time
import numpy as np
from audio.capture import AudioCapture
from speech.features import MFCCExtractor

def record_keyword_session():
    print("=" * 60)
    print("   ACOUSTIC KEYWORD TEMPLATE ENROLLMENT TOOL")
    print("=" * 60)
    print("This utility records your voice to create acoustic templates")
    print("for Dynamic Time Warping (DTW) keyword recognition.\n")

    capture = AudioCapture(sample_rate=16000)
    extractor = MFCCExtractor(sample_rate=16000)
    templates_root = "speech/templates"
    os.makedirs(templates_root, exist_ok=True)

    while True:
        keyword = input("\nEnter keyword to enroll (or 'q' to quit): ").strip().lower()
        if keyword in ('q', 'quit', 'exit'):
            break

        if not keyword:
            continue

        keyword_dir = os.path.join(templates_root, keyword)
        os.makedirs(keyword_dir, exist_ok=True)

        print(f"\nWe will record 3 spoken samples for keyword: '{keyword}'.")
        print("Please speak clearly after the countdown.")

        for i in range(1, 4):
            input(f"\nPress Enter to record Sample #{i}...")
            print("Ready...", end=" ", flush=True)
            time.sleep(0.5)
            print("Speak NOW! [MIC]", flush=True)

            try:
                capture.start()
            except Exception as e:
                print(f"[Error starting mic: {e}]")
                break

            audio_chunks = []
            start_t = time.time()
            while time.time() - start_t < 1.5:
                chunk = capture.get_chunk(timeout=0.2)
                if chunk is not None:
                    audio_chunks.append(chunk)

            capture.stop()
            print("Done capturing.")

            if not audio_chunks:
                print("[Warning] No audio captured. Skipping.")
                continue

            full_audio = np.concatenate(audio_chunks)

            mfcc = extractor.extract(full_audio)

            if len(mfcc) < 5:
                print("[Warning] Audio too quiet or short. Please try again.")
                continue

            save_path = os.path.join(keyword_dir, f"template_{i}.npy")
            np.save(save_path, mfcc)
            print(f"[OK] Template #{i} saved -> {save_path} (Shape: {mfcc.shape})")

        print(f"\n[SUCCESS] Successfully enrolled keyword '{keyword}' with 3 acoustic templates!")

if __name__ == "__main__":
    record_keyword_session()
