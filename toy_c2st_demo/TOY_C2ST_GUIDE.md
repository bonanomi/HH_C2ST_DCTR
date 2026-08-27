# Toy Classifier Two-Sample Test (C2ST) Walkthrough

## 0. Before we start

In this toy example and in the rest of the entire project, we will refer often to two terms:
- C2ST: stands for "CLassifier Two-Sample Test", i.e. a test that aims at assessing whether two samples are drawn from the same distribution. This is nowadays done with Neural Networks (NN) as described in [this reference paper](https://arxiv.org/abs/1610.06545).
- DCTR: stands for "Deep neural networks using Classification for Tuning and Reweighting". It is a multidimensional reweighting approach that uses NNs to determine the reweighting factors that can be used to improve the agreement between two samples (for example Data and Monte Carlo (MC) simulation). This is described in more detail in [this reference paper](https://arxiv.org/pdf/1907.08209).

The example contained in this folder helps you familarizing with both concepts in a simple use case where all the aspects can be controlled explicitly.

## 1. Purpose of this example

This is the recommended entry point before working with the full DY validation pipeline.

The real analysis contains many complications:

- millions of events;
- physical event weights;
- DY-specific corrections;
- negative MC weights;
- many correlated observables;
- train/validation/test bookkeeping;
- data loading from Parquet files;
- GPU execution;
- DCTR reweighting;
- cross-validation and closure tests.

Those ingredients are important in the real study, but they can hide the central idea of a classifier two-sample test.

This toy example removes almost all of that complexity and asks only:

> **Can a neural network tell two weighted event samples apart?**

The script creates a situation where we know the answer by construction.

It demonstrates two cases:

1. **Residual Data/MC disagreement**  
   MC is reweighted with an intentionally incomplete correction.  
   A neural network should still distinguish Data from MC, giving an AUC clearly above 0.5.

2. **Near-perfect Data/MC closure**  
   MC is reweighted with the exact toy density ratio.  
   The weighted MC distribution should match Data, and an independent neural network should lose its discrimination power, giving an AUC close to 0.5.

The main script is:

```bash
python toy_c2st_demo.py
```

---

# 2. What is a classifier two-sample test?

Suppose we have two samples:

- Data, distributed according to $p_\mathrm{Data}(x)$;
- Monte Carlo, distributed according to $p_\mathrm{MC}(x)$.

Here $x$ can be one variable or a high-dimensional vector of observables.

We assign labels

$$
y =
\begin{cases}
1 & \text{Data},\\
0 & \text{MC}.
\end{cases}
$$

Then we train a binary classifier.

If the two feature distributions are different,

$$
p_\mathrm{Data}(x) \neq p_\mathrm{MC}(x),
$$

the classifier can learn where Data-like and MC-like events tend to appear.

Its held-out AUC will then be above 0.5.

If the distributions are identical,

$$
p_\mathrm{Data}(x) = p_\mathrm{MC}(x),
$$

there is no feature information that can reveal which sample an event came from.

The classifier becomes equivalent to random ranking and

$$
\mathrm{AUC} \rightarrow 0.5.
$$

This is the core idea behind the C2ST.

A very important interpretation rule is therefore:

> **For an ordinary classification problem, larger AUC is better.  
> For a Data/MC closure C2ST, an AUC closer to 0.5 means better agreement.**

---

# 3. The toy distributions

The script uses two input variables,

```text
x1
x2
```

so that the problem remains easy to visualize.

Both Data and MC are generated from correlated two-dimensional Gaussian distributions.

The Data mean is

$$
\mu_\mathrm{Data} = (0,0),
$$

while raw MC is displaced:

$$
\mu_\mathrm{MC} = (0.9,-0.7).
$$

They use the same covariance matrix,

$$
\Sigma =
\begin{pmatrix}
1 & 0.45\\
0.45 & 1
\end{pmatrix}.
$$

So raw Data and raw MC visibly disagree.

In code:

```python
DATA_MEAN = np.array([0.0, 0.0])
MC_MEAN = np.array([0.9, -0.7])
```

and samples are generated with

```python
rng.multivariate_normal(...)
```

---

# 4. Why event weights can change a distribution

Suppose MC events are sampled from

$$
p_\mathrm{MC}(x).
$$

If every MC event is assigned a weight

$$
w(x)=
\frac{p_\mathrm{target}(x)}
     {p_\mathrm{MC}(x)},
$$

then weighted MC behaves like the target distribution:

$$
w(x)p_\mathrm{MC}(x)=
p_\mathrm{target}(x).
$$

This density-ratio identity is the conceptual basis of the toy correction.

It is also closely related to why a classifier can be used for multidimensional reweighting in DCTR-like methods.

In a real analysis we do not know the exact Data density. In this toy example we *do* know it, because we generated the distributions ourselves.

That lets us construct a known perfect answer.

---

# 5. The deliberately incomplete nominal correction

We first pretend that the analysis already has a nominal MC correction.

Instead of correcting MC all the way to the Data distribution, it corrects MC only toward an intermediate Gaussian:

$$
\mu_\mathrm{nominal}=
(0.45,-0.35).
$$

This is halfway between raw MC and Data.

The script computes

$$
w_\mathrm{nominal}(x)=
\frac{
p(x\mid\mu_\mathrm{nominal})
}{
p(x\mid\mu_\mathrm{MC})
}.
$$

Weighted MC is therefore improved, but residual disagreement remains.

This is intentionally exaggerated so that the C2ST effect is easy to see.

The first classifier compares

```text
Data
vs.
MC weighted by w_nominal
```

and should obtain an AUC noticeably above 0.5.

---

# 6. The closure correction

The second correction uses the exact density ratio


$$
w_\mathrm{closure}(x)=
\frac{
p_\mathrm{Data}(x)
}{
p_\mathrm{MC}(x)
}.
$$

Therefore,

$$
w_\mathrm{closure}(x)
p_\mathrm{MC}(x)=
p_\mathrm{Data}(x).
$$

Apart from finite Monte Carlo statistics, weighted MC now follows the same distribution as Data.

The second classifier compares

```text
Data
vs.
MC weighted by w_closure
```

and should obtain

$$
\mathrm{AUC}\approx0.5.
$$

This is the central demonstration.

---

# 7. Why the MC weight is not an NN input

The input matrix contains only

```text
x1
x2
```

for each event.

The classifier is **not** given the event weight as a feature.

The event weight instead enters through

```python
model.fit(..., sample_weight=...)
```

This distinction is important.

The network learns a function of event kinematics,

$$
f(x_1,x_2),
$$

while the sample weights determine how strongly each MC event contributes to the loss.

This is the same basic distinction used in the real analysis.

---

# 8. Why the Data and MC classes are globally normalized for training

The script retains the event-to-event MC correction weights, but additionally multiplies the entire MC class by one global factor.

On the training sample,

$$
C =
\frac{N_\mathrm{Data}}
{\sum_\mathrm{MC}w_i}.
$$

The classifier therefore uses

$$
w_i^\mathrm{C2ST}=
C\,w_i^\mathrm{physical}
$$

for MC.

Data events use weight 1.

The purpose is to make the total effective Data and MC training weights equal.

Why is this useful?

Imagine Data and MC have exactly the same shape but different total normalizations.

If we leave the class totals unequal, binary cross entropy can learn that unequal class prior.

That is not the discrepancy we want the C2ST to focus on.

By balancing the classes, we ask:

> **Given the event features, can the classifier distinguish the shape of Data from the shape of weighted MC?**

The relative MC weights are still fully present.

Only one common multiplicative factor is introduced.

Weighted AUC is invariant to such a global rescaling of all MC weights.

---

# 9. Why the same split is used for both tests

The script makes one split into

```text
training
validation
test
```

and uses exactly the same event indices for both weighting scenarios.

This is important because then

```text
AUC_nominal
AUC_closure
```

are evaluated on the same physical Data/MC events.

The only difference between the two experiments is the MC weight.

That makes the comparison much cleaner.

The test set is never used to optimize the network.

---

# 10. The neural network

The network is deliberately small:

```text
2 inputs
  ↓
Dense(32, ReLU)
  ↓
Dense(32, ReLU)
  ↓
Dense(1, sigmoid)
```

The final sigmoid output can be interpreted as a classifier score approximately related to

$$
P(\mathrm{Data}\mid x).
$$

For this toy problem, a much larger architecture is unnecessary.

The point is not to optimize an industrial-strength classifier.

The point is to demonstrate that a flexible classifier can expose residual Data/MC differences.

---

# 11. Early stopping

The script uses

```python
tf.keras.callbacks.EarlyStopping(...)
```

with restoration of the best validation weights.

Training therefore stops once the validation loss ceases to improve.

This reduces unnecessary epochs and protects against obvious overtraining.

---

# 12. What AUC measures

The ROC AUC is based on event ranking.

Informally, it asks:

> If I randomly choose one Data event and one MC event, how often does the classifier assign the Data event the more Data-like score?

For perfect random ranking,

$$
\mathrm{AUC}=0.5.
$$

For increasingly distinguishable distributions,

$$
\mathrm{AUC}>0.5.
$$

In this example we expect

```text
AUC nominal  > 0.5
AUC closure ~= 0.5
```

Exact values depend on random fluctuations and neural-network optimization.

The closure AUC should therefore be interpreted as *close to* 0.5 rather than required to equal 0.500000 exactly.

---

# 13. Output plots

By default, the script writes outputs to

```text
toy_c2st_outputs/
```

## 13.1 `input_x1.png`

Shows Data and MC shapes for `x1`:

- raw MC;
- nominally weighted MC;
- closure-weighted MC.

The lower panel shows MC/Data ratios.

## 13.2 `input_x2.png`

Same idea for `x2`.

These two figures make the weighting effect visible without involving a neural network.

## 13.3 `roc_comparison.png`

Contains both C2ST ROC curves.

The nominal curve should visibly depart from the diagonal.

The closure curve should lie close to the diagonal.

## 13.4 `classifier_scores_mc_nominal.png`

Shows the NN output distribution for Data and nominally weighted MC.

If the classifier has learned a real residual discrepancy, these distributions differ.

## 13.5 `classifier_scores_mc_closure.png`

Shows Data and closure-weighted MC classifier outputs.

With good closure, the two distributions should nearly overlap.

## 13.6 `mc_weight_distributions.png`

Shows the nominal and closure multiplicative MC corrections themselves.

This is useful because reweighting quality should never be judged only from AUC.

One should also inspect whether the correction creates very large or pathological event weights.

## 13.7 `summary.json`

Stores the central numerical results in machine-readable form, including:

```text
auc_nominal
auc_closure
class-balancing factors
Kish effective sample sizes
```

---

# 14. Effective sample size

Weighted samples do not generally have the same statistical power as an unweighted sample with the same number of rows.

The script reports the Kish effective sample size,

$$
N_\mathrm{eff}=
\frac{(\sum_iw_i)^2}
{\sum_iw_i^2}.
$$

If all weights are identical,

$$
N_\mathrm{eff}=N.
$$

If a small number of events carry huge weights,

$$
N_\mathrm{eff}\ll N.
$$

This is an important diagnostic for real reweighting methods such as DCTR.

A reweighting that produces beautiful closure only by assigning enormous weights to a handful of events may be statistically unstable.

---

# 15. Running the example

Activate the same environment used for the C2ST project:

```bash
source ~/miniforge3/bin/activate
conda activate c2st
```

Then run

```bash
python toy_c2st_demo.py
```

A smaller development run is:

```bash
python toy_c2st_demo.py \
    --events 30000 \
    --epochs 20
```

A larger run can be made with:

```bash
python toy_c2st_demo.py \
    --events 200000 \
    --epochs 50
```

---

# 16. GPU use

TensorFlow will automatically use an available GPU.

Check with:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

For this tiny toy example, GPU acceleration is not essential.

The example is intentionally small enough to run comfortably on a normal machine.

The GPU becomes much more useful for the multi-million-event real analysis.

---

# 17. Exercises and questions

The toy is designed to be modified: play around with it!

## Exercise 1 — remove the residual discrepancy

Change

```python
NOMINAL_TARGET_MEAN
```

to exactly

```python
DATA_MEAN
```

Then nominal and closure weights become equivalent.

What happens to the two AUCs?

---

## Exercise 2 — make the mismatch worse

Move

```python
NOMINAL_TARGET_MEAN
```

closer to

```python
MC_MEAN
```

so that the nominal correction does less.

Does the nominal C2ST AUC increase?

---

## Exercise 3 — use only one NN feature

Change

```python
X = np.vstack([data, mc])
```

so the classifier sees only `x1`.

Compare the AUC with the two-feature classifier.

This illustrates that a C2ST can exploit multivariate information that may not be visible in one variable alone.

---

## Exercise 4 — create a correlation mismatch

Keep Data and MC means equal but give them different covariance matrices.

For example, modify MC to have a different `x1-x2` correlation.

Then inspect the one-dimensional `x1` and `x2` histograms.

They may look fairly similar even when the joint distribution differs.

Can the two-feature NN still find the mismatch?

This is one of the major motivations for multivariate C2STs.

---

## Exercise 5 — deliberately damage the closure weights

Replace

$$
w_\mathrm{closure}
$$

with

$$
w_\mathrm{closure}^{0.8}.
$$

This produces an intentionally incomplete density-ratio correction.

What happens to the closure AUC?

---

## Exercise 6 — inspect weight tails

Increase the Data/MC separation.

What happens to the maximum density-ratio weight and Kish effective sample size?

This illustrates the overlap problem: if Data and MC occupy very different regions, reweighting may require very large event weights.

---

# 18. Connection to the real DY study

The toy correspondence is:

| Toy example | Real analysis |
|---|---|
| `x1`, `x2` | physics features such as `mli_ll_pt`, `mli_n_jet`, MET, masses, angular variables |
| raw MC Gaussian | simulated background events |
| nominal toy weight | nominal DY correction |
| exact closure density ratio | idealized target that DCTR tries to estimate |
| Data vs nominal MC C2ST | before/after DY-correction validation |
| Data vs closure MC C2ST | final DCTR closure classifier |
| exact analytic density ratio | unknown in real data |
| small NN | full C2ST NN |

The crucial difference is that in the toy example we know the true generating probability densities.

In the real analysis, we only have finite Data and MC samples.

Therefore:

- the true density ratio is unknown;
- DCTR estimates it with a classifier;
- cross-validation is needed to avoid evaluating a reweighter on events it trained on;
- negative NLO weights require additional care;
- physical selections and systematic uncertainties matter;
- a residual AUC must be interpreted in terms of actual analysis impact.

---

# 19. What this example does *not* prove

This toy is a conceptual demonstration.

It does not establish that every AUC close to 0.5 means a real analysis model is correct.

A classifier can fail to see a discrepancy because of:

- insufficient capacity;
- poor preprocessing;
- too little training;
- too few events;
- inappropriate features.

Likewise, an AUC above 0.5 does not automatically mean a discrepancy is physically important.

With very large event samples, tiny differences can be statistically detectable.

The full analysis therefore needs both:

1. **detection** of residual discrepancies;
2. **interpretation** of where they occur and whether they affect the physics result.

---

# 20. The main lesson

The entire demonstration can be summarized in three lines:

$$
\text{Data/MC mismatch}
\Rightarrow
\text{classifier can distinguish them}
\Rightarrow
\mathrm{AUC}>0.5
$$

while

$$
\text{Data/MC closure}
\Rightarrow
\text{classifier loses discrimination}
\Rightarrow
\mathrm{AUC}\approx0.5.
$$

That is the basic logic underlying the much larger C2ST/DCTR pipeline used in the DY validation study.
