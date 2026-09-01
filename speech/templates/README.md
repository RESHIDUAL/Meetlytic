# Keyword Audio Templates Directory

This directory stores enrolled acoustic templates for keyword spotting without pre-trained neural networks.

## Directory Structure

Each enrolled keyword has its own subfolder containing extracted MFCC feature arrays stored in NumPy binary format (`.npy`):

```
speech/templates/
|-- meeting/
|   |-- template_1.npy
|   |-- template_2.npy
|   \-- template_3.npy
|-- project/
|   |-- template_1.npy
|   \-- template_2.npy
|-- deadline/
|   |-- template_1.npy
|   \-- template_2.npy
\-- prototype/
    \-- template_1.npy
```

## Enrolling New Keywords

To enroll new professional or personal keywords, use the interactive enrollment tool:

```bash
python -m tools.record_template
```

Each keyword should have 3 to 5 spoken utterances recorded in quiet conditions to provide acoustic variation for Dynamic Time Warping (DTW) matching.
