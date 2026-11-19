# APS DFD 2026 Interact session: story (draft v1, 19 Nov)

_For Krishna. A working document, edited together. It covers a 2-minute talk plus one large e-poster slide that people visit for 40 minutes._

## Main message

**A machine-learning closure, constrained by two pieces of physics, makes a cheap two-equation film model reproduce the DNS waves far better than the classical Lavalle model, including on parameter cases it never saw.**
- **Physics 1, flat film exact:** the learned closure is zero for a flat film, by construction.
- **Physics 2, linear growth:** linear waves grow at the rate Orr–Sommerfeld predicts.

The work is not finished: the model still fails in some conditions, and the poster shows how we are fixing that.

## 2-minute talk (3 slides, about 40 s each)

**Slide 1: the problem.**
- "Thin films falling down an incline form large solitary-like waves. A two-dimensional DNS of one case costs about 460 core-hours. Depth-averaged models are about 1,000 times cheaper, but the closures they rely on are derived under assumptions that break down: the classical Lavalle model gets the waves too low and the wrong shape."
- Visual: DNS against Lavalle h(x) at t = 2000 (`broadband_waves.png`, top rows).

**Slide 2: the idea.**
- "We extract the exact closure terms from the DNS, and show that inserting them into the ROM fixes it. So the closure is the bottleneck. We then learn the correction with a neural network.
- A plain network fits the data well but drifts when it runs inside the model. Two physics constraints fix that: the correction vanishes for a flat film, and linear waves must grow at the Orr–Sommerfeld rate."
- Visual: one equation, S = N(state) − N(flat film), plus a schematic: DNS → closure → network → ROM.

**Slide 3: the result and the invitation.**
- "On seven held-out cases, the distribution of film thickness from the ML model is 2.5–5 times closer to the DNS than Lavalle's, and crest heights are within 5%. It still fails when the film gets thinner than anything in the training data. Come to the poster to see where and how we are fixing it."
- Visual: the per-case metrics figure (`broadband_metrics.png`).

## E-poster: one large slide

**Layout:** three columns, read left to right. A title strip with the main message. A take-home box at the bottom right.

### Column 1: problem and diagnosis

**Panel 1: setup.**
- The film schematic and a DNS interface snapshot.
- The (Re, Ca, β) parameter map: training cases and the 7 held-out cases marked differently.
- Broadband inlet noise.
- A cost line: DNS about 460 core-h per case; Lavalle ROM about 0.3 core-h; ML ROM about 14 core-h.

**Panel 2: the closure is the bottleneck.**
- The Lavalle equations, with the unclosed terms highlighted.
- The oracle test: the ROM run with the exact DNS closure terms against Lavalle, from the same DNS state. h error 0.19 against 0.59 at t = 50.
- One finding about the roles of the terms: ΔM_x and ΔT_w largely cancel and must be applied together.

### Column 2: learning the closure

**Panel 3: naive MLP, ablation and where it fails.**
- **A priori:** a 1×16 MLP on local inputs removes 75–80% of the Lavalle error in the nonlinear wave region on held-out cases. The input ablation shows (h, h_x) is about as good as (h, h_x, q). Predicting the sum ΔM_x + ΔT_w as one target is harder than either term alone, but works better in the ROM.
- **A posteriori, oracle case:** the model matches the exact closure for about 10 time units. It then drifts:
  - crests grow too high (2.50 against 2.08);
  - the h error reaches 1.26 at t = 100, worse than Lavalle;
  - separately trained models for each term fail sooner.
- **Diagnosis:** the network adds a spurious source to a flat film, and gets the growth rate of small waves wrong.
- Visual: a small table, plus an error-against-time curve.

**Panel 4: physics-constrained closure.**
- The equation S = N(state) − N(h, h³, 0, 0, 0): zero for a flat film by construction.
- The Orr–Sommerfeld constraint in the loss.
- Network: 2×64, tanh, inputs h, q, h_x, q_x, h_xx, Re, Ca, β.
- Linear-wave test (`comb_growth_mlp.png`): growth within about 7–11% of OS at Re 120–200 (Lavalle 27–39%). Without the constraint, the waves are damped at Re 200.

### Column 3: results and what is next

**Panel 5: from the DNS state, short times.**
- The oracle case and two dense restarts (`dense_error_mlp.png`).
- Lowest error of all learned closures: Re 20 0.11 against Lavalle 0.28; Re 120 0.25 against 0.59, at t = 10. Oracle 0.39 at t = 50 against 0.47 for the naive MLP.

**Panel 6 (main panel): held-out broadband cases, long times.**
- `broadband_waves.png`, plus `broadband_metrics.png` (or one of them, if space is short).
- Distribution distance 0.04–0.07 against 0.11–0.29 for Lavalle on 4 cases; crests within 5% (Lavalle 6–28% too low).

**Panel 7: where it fails, and the fixes in progress.**
- 2 of 14 runs blow up when the film thins below the training range (h < 0.21–0.28) and the network extrapolates (S about 60× its training size).
- **Safeguard:** clamp the inputs to the training range, and fade S out in near-dry film. Results if ready.
- **Also:** Re 40 Ca 0.04 crests 16% too high; waves too long at Re 200 β 6.4°.
- **Next:** a clean validation/test split, an architecture and input ablation, a seed ensemble, and rollout training.

**Take-home (3 bullets):**
1. The closure, not the depth-averaged equations, limits the ROM.
2. A learned closure must be exact for a flat film and match linear theory, or it drifts in closed loop.
3. With both constraints, the ML ROM beats Lavalle on held-out cases; robustness outside the training range is the open problem.

## Open questions

_Resolved 19 Nov: the abstract (written in July) does not constrain the story. Panel 6 layout and safeguard results: later._

1. **Cost line:** DNS about 460 core-h (Re 120 Ca 0.01 β 45, t = 2000, 16 threads, 28.9 h wall); Lavalle ROM about 0.3 core-h (3.5 ms per step, 1 core); ML ROM about 14 core-h (about 42 ms per step on 4 threads). The ML ROM cost is the per-cell network evaluation (see below). It can be cut by an order of magnitude, so the poster can say "currently about 30× cheaper than DNS; the network evaluation is not yet optimised".

**Why the ML ROM is slow.**
- **Work per step:** at every one of the 4 RK stages, each of the 4000 cells evaluates the 8 → 64 → 64 → 1 network. That is about 4,700 multiply-adds and 128 tanh per cell, about 75 M multiply-adds and 2 M tanh per time step.
- **How it is coded:** the evaluation is a scalar, one-cell-at-a-time loop (`MLP::eval`), so it neither vectorises nor uses BLAS.
- **The time step** (0.005) is set by the capillary term, the same for Lavalle, so it is not the network's fault.
- **Fixes, largest first:**
  - batch all cells into one matrix-matrix product (Eigen/BLAS, float32), probably 10–30× faster;
  - a fast tanh approximation;
  - evaluating S once per time step instead of once per RK stage, which saves 4× but changes the time accuracy.
