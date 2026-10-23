\# Lab 1: Environment and first system measurements



\## 1. Goal

Set up a reproducible environment, train two baseline models on the Breast Cancer Wisconsin dataset, measure their system cost, and decide which deployment targets (Cloud, Edge, Mobile, TinyML) each model fits.



\## 2. Method

\- Python 3.11.9 in a venv with pinned requirements (versions in `results/versions.txt`). Note: the manual asks for 3.11.8; I used 3.11.9, the same minor version. `pyarrow` is pinned to 15.0.2 because `mlflow==2.14.1` requires `pyarrow<16`, which conflicts with the 16.1.0 in the manual.

\- Because two models were measured, they are saved as `results/model\_LogisticRegression.joblib` and `results/model\_RandomForest.joblib` instead of a single `results/model.joblib`.

\- Data: `load\_breast\_cancer`, 70/30 stratified split, `random\_state=42`. Seeds set to 42.

\- Models: LogisticRegression (max\_iter=1000) and RandomForest (100 trees).

\- Training time: median of 5 runs after 1 warm-up. Inference: median of 100 single-sample predictions after a warm-up. Model size: `joblib.dump` file size. Peak memory: `memory\_profiler` during training and during 100 inferences.

\- Code is in `src/`, raw numbers in `results/`.

\## 3. Results



| Metric | LogisticRegression | RandomForest |

|---|---|---|

| Test accuracy | 0.9415 | 0.9357 |

| Training time (median, s) | 0.4118 | 0.1103 |

| Inference latency (median, ms) | 0.0431 | 1.4899 |

| Model size (bytes) | 1055 | 290889 |

| Model size (KB) | 1.03 | 284.07 |

| Peak memory, training (MiB) | 143.4 | 145.3 |

| Peak memory, inference (MiB) | 143.5 | 145.3 |



Note: peak memory is that of the whole Python process (Python + libraries), not of the model alone. That is why both models show almost the same value.



\### Deployment fit



| Target | Budget | LogisticRegression | RandomForest |

|---|---|---|---|

| Cloud | mem ≥ 1 GB, ≤ 100 ms, ≤ 500 MB | Fits | Fits |

| Edge | 256–1024 MB, ≤ 50 ms, ≤ 50 MB | Fits (143 MiB, 0.04 ms, 1 KB) | Fits (145 MiB, 1.5 ms, 284 KB) |

| Mobile | 64–256 MB, ≤ 20 ms, ≤ 10 MB | Fits | Fits |

| TinyML | ≤ 256 KB, ≤ 10 ms, ≤ 100 KB | Model size and latency fit, but measured memory (143 MiB) does not; possible only if reimplemented without the Python stack | Does not fit (284 KB > 100 KB) |



Caveat: latency was measured on a desktop CPU, not on the target hardware, so Mobile and TinyML latency would be higher in reality.



\## 4. Conclusions

1\. The accuracy is almost the same (0.9415 / 0.9357), but Random Forest is much larger (284 KB / 1 KB) and has slower inference (1.49 ms / 0.04 ms).

2\. Memory usage is approximately 145 MiB for both because what is measured is the memory usage of the entire Python process, not just the model.

3.Logistic Regression is closer to TinyML (1 KB, within the 100 KB limit), while Random Forest is far from it (284 KB).

