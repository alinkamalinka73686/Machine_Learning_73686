# Individual Project Proposal

**Student:** Alina Muradova  **ID:** 73686

Title: Keyword Spotting on a Small Budget: Compressing a Speech Command Recognizer for On-Device Use

## Problem

Voice assistants usually send your audio to a server, which is slow and not great for privacy. I want to see how small and fast a keyword spotting model can get while still being accurate enough to be useful on a phone. The task is to recognize 10 spoken commands (yes, no, up, down, left, right, on, off, stop, go) from 1-second clips, plus "silence" and "unknown", so 12 classes in total. What interests me most is the trade-off: a big model is easy to make accurate, but the real challenge is keeping that accuracy after shrinking it.

## Dataset

I will use Google Speech Commands v2 (about 105,000 clips from roughly 2,600 speakers), loaded through torchaudio. I will use the official validation and test lists, because they split the data by speaker. This means nobody who is in the training set shows up in the test set, which avoids the kind of leakage we talked about in Lab 4. "Silence" examples are cut from the background noise files, and I will downsample "unknown" so it doesn't dominate. Before training I will run validation checks (clip length, corrupted files, class balance) and report any problems instead of silently dropping data. The whole pipeline runs from a YAML config with seed 42, and I will save a SHA256 hash of the processed data and write a data card.

## Success metric

Test accuracy of at least 90%, with macro-F1 reported as well.
After compression, accuracy should drop by no more than 1 percentage point compared to the FP32 baseline.
I will also test every compressed model on clips with added background noise since a model that only works in a quiet room is not very useful.


##Deployment constraints

Latency of 50 ms or less per clip on CPU with batch size 1. This includes computing the mel spectrogram, not just the model itself.
Model size of 5 MB or less on disk.

## Planned optimization

I will start with a larger CNN (about 1M parameters) as the teacher and baseline. Then I will try three compression methods and compare them.
Pruning: global and per layer magnitude pruning at 0 to 95% sparsity, and structured filter pruning at 10, 25 and 50%, with short fine tuning after each.
Distillation: a small DS-CNN student (about 100k parameters) trained with and without the teacher (T=4, α=0.5).
Quantization: FP32 vs INT8, comparing post-training quantization and quantization-aware training.

## Plan
Build and validate the data pipeline, then train the teacher.
Train the student with and without distillation.
Apply pruning and quantization.
Measure accuracy, size, latency and noise robustness, and put them in one table to check whether the targets are met.
