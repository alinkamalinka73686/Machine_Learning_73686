\# Lab 5: Frameworks and Automatic Differentiation



Name: Alina Muradova, Student ID: 73686



Code is in `src/`; run the scripts from inside `src/` (e.g. `cd src` then `python verify.py`).

Screenshots of all outputs are in `screenshots/`.



\## 1. Autodiff engine (src/engine.py)

A scalar `Value` node stores `data`, `grad`, parent nodes (`\_prev`), the op name (`\_op`)

and a `\_backward` closure holding the local derivative rule. Addition, multiplication,

powers, ReLU (plus sigmoid and log for BCE) build the computation graph during the forward pass.

`backward()` topologically sorts the graph, sets the output gradient to 1, visits nodes in

reverse order and accumulates gradients with `+=` (checked with `y = x\*x`, giving `x.grad = 6`).



\## 2. Verification (src/verify.py)

Function: `((a\*b + c)\*\*2).relu().sum()`, 6 elements, float64.

\- Max |engine - PyTorch autograd|: 0.0

\- Max |engine - numerical (central difference)|: 1.5e-10

\- Threshold 1e-6: PASS



\## 3. Training on make\_moons (src/train\_moons.py)

Architecture 2 -> 8 -> 1 (ReLU, sigmoid), BCE loss, 200 samples, 100 epochs, full-batch, lr 0.5,

identical initial weights. Median of 5 runs.



| Implementation | Median time | Final loss | Accuracy |

|---|---|---|---|

| My engine | 33.49 s | 0.2350 | 0.890 |

| PyTorch | 0.0368 s | 0.2350 | 0.890 |



The engine is about 910x slower. Every scalar operation is a separate Python object and

closure, while PyTorch runs a few vectorised C++ kernels. Loss and accuracy are identical.



\## 4. PyTorch graph, eager vs compiled, contiguity (src/torch\_inspect.py, src/test6.py)

Graph: `SigmoidBackward0 -> AddmmBackward0 -> (AccumulateGrad, ReluBackward0 -> AddmmBackward0, TBackward0)`.

`AccumulateGrad` plays the same role as `+=` in our engine.



Eager vs compiled (MLP 256-512-512-1, batch 1024, after warm-up):

\- eager: 3.591 ms

\- torch.jit.trace: 3.813 ms (0.94x). `torch.compile` failed with an InductorError on this

&#x20; Windows machine (no C++ compiler toolchain), so tracing was used as the fallback.



Contiguity (2048x2048 float32):

\- `A` stride (2048, 1), contiguous; `A.t()` stride (1, 2048), non-contiguous (a view, no copy).

\- `A + B` both contiguous: 1.129 ms; `At + B` mixed layout: 14.777 ms (13.09x slower);

&#x20; `At.contiguous() + B`: 1.196 ms.

\- `At + 1` was not slower (0.96x) and `At.sum(0)` was faster (0.36x): non-contiguous is only

&#x20; slow when the access pattern defeats cache and vectorisation.

\- `At.contiguous()` itself costs 9.574 ms, so copying pays off only if the tensor is reused.



\## 5. When does compilation help?

Compilation helps when the model has many small element-wise operations that can be fused

(fewer memory passes), when tensors are small and Python/dispatch overhead dominates, and

when the same fixed-shape code runs many times so the one-off compile cost is amortised.

It did not help here: the MLP is dominated by large matrix multiplications that already run

in optimised BLAS, tracing only removes Python overhead and does not fuse operations,

and the first-call compile cost is wasted for code that runs rarely.

