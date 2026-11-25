# Individual Project Proposal

**Student:** Alina Muradova  **ID:** 73686

## Problem
I want to classify small color photos into 10 classes (like cars, birds, cats) with a model that is small and fast enough to run on a phone, so images don't have to be sent to a server.

## Dataset
CIFAR-10. It has 60,000 color images of size 32x32 in 10 classes (50,000 for training, 10,000 for testing).

## Success metric
- Test accuracy of at least 75% on the 10,000 test images.
- After compression, accuracy should drop by no more than 1 percentage point compared to the baseline.

## Deployment constraints
- Latency of 50 ms or less per image (CPU, batch size 1).
- Model size of 5 MB or less.

## Planned optimization
I will use magnitude pruning and knowledge distillation. For distillation, a big baseline CNN is the teacher and a much smaller CNN is the student. I will compare accuracy, size and latency for the baseline, the pruned model and the distilled model.

## Plan
1. Train a baseline CNN and measure accuracy, size and latency.
2. Apply pruning and distillation, then measure again.
3. Check the results against the constraints above.
