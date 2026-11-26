# Lab 6: Training pipelines and project proposal

Alina Muradova, 73686

For this lab I used the CNN from Lab 3 on Fashion-MNIST. Everything ran on my laptop CPU, so the timings are slower than they would be on a GPU. The code is in `src/lab6.py` and all settings are in `src/config.yaml`.

## 1. YAML pipeline and MLflow

I moved my Lab 3 training code into one script that reads everything from a single YAML file (seed, batch size, optimizer, learning rate, scheduler, epochs, and so on). It sets the seed to 42 and saves a checkpoint after every epoch, plus the best one by validation accuracy. For each run MLflow logs the parameters, the metrics per epoch (loss, validation accuracy, epoch time, throughput, learning rate) and the artifacts (the config and the best checkpoint).

The baseline (CNN, Adam, lr 0.001, batch 128, 5 epochs) got a test accuracy of 0.8959 and needed about 7 seconds per epoch. All 21 runs are visible in `results/mlflow_runs.png`.

## 2. Profiling one epoch

| Stage | Time (s) | Share |
|---|---|---|
| Data loading | 0.39 | 5.5% |
| Forward | 3.10 | 43.9% |
| Backward | 3.10 | 43.9% |
| Optimizer step | 0.47 | 6.7% |

Almost 90% of the time goes into the forward and backward passes. Inside those, convolution backward (about 31%) and max pooling (about 24%) are the biggest parts. Data loading is small, so on this setup the training is limited by computation and not by the data pipeline.

## 3. Number of workers

I ran 3 epochs for each setting and averaged epochs 2 and 3, because the first epoch includes worker start-up time (on Windows every worker is a new process).

| num_workers | Epoch time (s) | Samples/s | First epoch (s) |
|---|---|---|---|
| 0 | 12.0 | 4583 | 12.0 |
| 1 | 8.6 | 6433 | 13.9 |
| 2 | 6.6 | 8397 | 16.0 |
| 4 | 6.4 | 8594 | 24.7 |
| 8 | 6.9 | 8029 | 50.2 |

With 0 workers, data loading takes about 40% of the epoch (4.8 s of 12 s), so training is limited by data loading. With 1 worker the epoch drops to about 8.6 s, and with 2 workers to about 6.5 s. After that the time stays flat (4 workers 6.4 s, 8 workers 6.9 s), so the bottleneck moves to computation and more workers do not help. I would use 2 workers.

The first epoch gets slower as workers are added (12.0 s with 0 workers, 50.2 s with 8). This is the start-up cost of the worker processes, and it is the reason my first measurement (one epoch only) wrongly suggested that 0 workers is fastest. The profile in section 2 was taken with 2 workers, where prefetching hides most of the loading time, so its 5.5% is only the waiting time.


## 4. Batch size

| Batch | Samples/s | Peak memory (MB) | Val acc | Test acc |
|---|---|---|---|---|
| 32 | 5915 | 528 | 0.8998 | 0.9010 |
| 64 | 6494 | 535 | 0.9036 | 0.8975 |
| 128 | 7542 | 551 | 0.8986 | 0.8959 |
| 256 | 7816 | 584 | 0.8886 | 0.8864 |
| 512 | 6609 | 631 | 0.8744 | 0.8718 |

Throughput goes up until batch 256 and then drops at 512. Memory grows with batch size, as expected (this is the memory of the whole process on CPU, not GPU memory). Accuracy after 5 epochs gets worse for big batches, most likely because there are fewer weight updates per epoch. I would pick batch 128 as a compromise: it is close to the fastest and the accuracy is still about 0.90. The differences between 32, 64 and 128 are small and I only used one seed, so I would not read too much into them.

## 5. Schedulers and optimizers (10 epochs each)

| Optimizer | Scheduler | Val acc | Test acc | Final train loss |
|---|---|---|---|---|
| SGD + momentum (lr 0.05) | StepLR | 0.9100 | 0.9073 | 0.1920 |
| SGD + momentum (lr 0.05) | CosineAnnealingLR | 0.9140 | 0.9147 | 0.1571 |
| SGD + momentum (lr 0.05) | ReduceLROnPlateau | 0.9086 | 0.9094 | 0.1715 |
| Adam (lr 0.001) | StepLR | 0.8972 | 0.8935 | 0.2513 |
| Adam (lr 0.001) | CosineAnnealingLR | 0.9078 | 0.9022 | 0.2144 |
| Adam (lr 0.001) | ReduceLROnPlateau | 0.9094 | 0.9062 | 0.2000 |

The best combination was SGD with cosine annealing (test accuracy 0.9147). StepLR cut the learning rate early and after that the loss barely moved. ReduceLROnPlateau worked, but the validation accuracy jumped up and down because it only reacts after the accuracy has already dropped. SGD ended up a bit ahead of Adam here, but I did not tune the learning rates and I only ran one seed, so I would not say SGD is better in general. The curves are in `results/sched_plot.png`.

## 6. Gradient accumulation

| Setup | Final train loss | Val acc | Test acc | Epoch time |
|---|---|---|---|---|
| Batch 256 | 0.2382 | 0.9010 | 0.8933 | about 6.2 s |
| Batch 64, accumulate 4 | 0.2383 | 0.9014 | 0.8945 | about 7.4 s |

The two runs behave almost exactly the same. The loss after epoch 1 is 0.6159 and 0.6160, and at epoch 10 it is 0.2382 and 0.2383. The small differences are probably from floating point rounding. Accumulation is about 20% slower per epoch, but it only keeps 64 samples in memory at a time, which is useful when memory is limited. See `results/accum_plot.png`.

## 7. Stopping and resuming

I stopped the training after epoch 3 and continued from the saved checkpoint (it stores the model, the optimizer, the scheduler and the history).

| Epoch | Train loss | Val acc |
|---|---|---|
| 1 | 0.5346 | 0.8648 |
| 2 | 0.3507 | 0.8794 |
| 3 (stopped) | 0.3076 | 0.8902 |
| 4 (resumed) | 0.2828 | 0.8888 |
| 10 | 0.2066 | 0.9020 |

The loss keeps going down smoothly and there is no jump when training restarts (`results/resume_plot.png`). Epoch 4 is slightly different from the uninterrupted run (0.2828 against 0.2837). I think this is because the random state is not saved in the checkpoint, so the batches come in a different order after resuming.

## 8. Project proposal

The one-page proposal is in `proposal.md`.

## Files

- `src/lab6.py`, `src/config.yaml`
- `results/`: the tables (`profile_table.md`, `workers_table.md`, `batch_table.md`, `sched_table.md`), the plots (`workers_plot.png`, `sched_plot.png`, `accum_plot.png`, `resume_plot.png`) and `mlflow_runs.png`
- `proposal.md`
