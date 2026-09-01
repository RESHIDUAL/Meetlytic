"""
Classifier Training and Vocabulary Calibration Tool.
Trains the Multinomial Naive Bayes classification model from scratch using
either the built-in benchmark dataset or interactive user-provided examples.
Exports learned probabilities and vocabulary to JSON.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nlp.classifier import ConversationClassifier
from config.vocabulary import PROFESSIONAL_KEYWORDS, PERSONAL_KEYWORDS, CASUAL_KEYWORDS

def get_default_training_corpus():
    texts = []
    labels = []
    for w in PROFESSIONAL_KEYWORDS:
        texts.append(f"We need to discuss the {w} and project implementation.")
        labels.append("professional")
    for w in PERSONAL_KEYWORDS:
        texts.append(f"I was talking about my {w} and family vacation.")
        labels.append("personal")
    for w in CASUAL_KEYWORDS:
        texts.append(f"Hey there, {w} how are you doing today?")
        labels.append("casual")
    return texts, labels

def main():
    print("=" * 60)
    print("   CUSTOM NAIVE BAYES CLASSIFIER TRAINER")
    print("=" * 60)
    print("This tool fits statistical word likelihoods from scratch")
    print("without any external pre-trained models or cloud APIs.\n")

    classifier = ConversationClassifier()
    save_path = "config/trained_model.json"
    os.makedirs("config", exist_ok=True)

    print("Options:")
    print("  1. Train from default lexicon corpus")
    print("  2. Interactive training (provide custom sentences and labels)")

    choice = input("\nSelect option (1/2, default 1): ").strip()

    if choice == "2":
        texts = []
        labels = []
        print("\nEnter sentences with labels. Type 'done' when finished.")
        valid_labels = {'professional', 'personal', 'casual'}
        while True:
            sent = input("\nSentence: ").strip()
            if sent.lower() == 'done':
                break
            if not sent:
                continue
            lbl = input("Label (professional/personal/casual): ").strip().lower()
            if lbl not in valid_labels:
                print("Invalid label. Must be professional, personal, or casual.")
                continue
            texts.append(sent)
            labels.append(lbl)
            print(f"Added: [{lbl.upper()}] \"{sent}\"")

        if len(texts) < 5:
            print("[Warning] Too few examples provided. Falling back to default corpus.")
            texts, labels = get_default_training_corpus()
    else:
        print("\nLoading default training corpus...")
        texts, labels = get_default_training_corpus()

    print(f"Fitting Naive Bayes on {len(texts)} training samples...")
    classifier.train_on_corpus(texts, labels)

    classifier.save_model(save_path)
    print(f"\n[OK] Training complete! Model parameters saved to: {save_path}")
    print(f"   Vocabulary Size : {len(classifier.naive_bayes.vocabulary)} terms")
    print(f"   Classes Fitted  : {classifier.naive_bayes.classes}")

if __name__ == "__main__":
    main()
