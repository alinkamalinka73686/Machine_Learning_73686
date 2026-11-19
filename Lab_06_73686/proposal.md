# Individual Project Proposal

**Student:** Alina Muradova  **ID:** 73686

## Problem
Classify clothing items from small grayscale images so the model can run on-device (for example in a mobile app that sorts a photo gallery or a wardrobe) without sending images to a server.

## Dataset
Fashion-MNIST: 70,000 grayscale 28x28 images, 10 classes (55,000 train / 5,000 validation / 10,000 test split, fixed seed 42).

## Success metric
- Test accuracy >= 90% on the 10,000-image test set (baseline from Lab 3/Lab 6: CNN, ~5 epochs).
- Accuracy drop after optimization <= 1 percentage point compared with the FP32 baseline.

## Deployment constraints
- Inference latency <= 50 ms per image (CPU, batch size 1).
- Model size <= 5 MB on disk.

## Planned optimization
Post-training INT8 quantization of the CNN (and, if accuracy allows, magnitude pruning before quantization). I will report size, latency and accuracy for FP32 vs. optimized model, tracked in MLflow.

## Plan
1. Reproducible baseline with the YAML pipeline (Lab 6).
2. Measure baseline size, latency, accuracy.
3. Apply quantization (+ pruning), re-measure, compare against the constraints.
