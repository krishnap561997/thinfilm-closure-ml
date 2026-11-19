# 18–19 Nov 2026: flat-film-exact MLP, all seed-0 results; faster ROM; broadband test; safeguard

_For Krishna. Continues `2026-11-17-structured-closure-explainer.md` (Part 6). Numbers and tables: `docs/results/structured-closure.md`, sections dated 18 Nov._

## 1. Where we stand in one paragraph

The flat-film-exact MLP, S = N(h, q, h_x, q_x, h_xx; p) − N(h, h³, 0, 0, 0; p) trained with the Orr–Sommerfeld (OS) constraint, is now the best learned closure in closed loop. On the oracle case (Re 120, Ca 0.01, β 45°, 10 Hz) the σ 0.5, λ_lin 10 model has h error 0.39 at t = 50 and 0.62 at t = 100. For comparison, the previous MLP (`ml_sum_hhx`) has 0.47 and 1.26, and the exact closure 0.19 and 0.55. The closed-loop gap to the exact closure is now about 0.2 at t = 50, down from about 0.3. Your new priority (draft by mid-December, best ML model first, no closed forms) means the next work is to:
- confirm this with seeds and with held-out broadband cases;
- then shrink the remaining gap.

## 2. What the four seed-0 runs show

**The OS constraint is necessary.** Linear waves were tested in the ROM with the comb inlet.
- With the constraint (λ_lin 1 or 10), the growth rate is within 7–11% of OS (median over the forced band) at Re 120 and 200. Lavalle is off by 27–39%.
- Without it (λ_lin 0), the Re 200 waves are damped: almost nothing grows, and the error is 47%.
- So the network does not learn linear growth from the data alone. The constraint is what fixes it, and the fix carries over to parameters far from the oracle case.

**The ROM does what the model says.** The ROM's measured growth matches each model's own analytic prediction to 3–5%. Any remaining error is in the model, not in how the solver uses it.

**σ 0.5 beats σ 0.25 in closed loop.** This is the opposite of the a priori ranking, where σ 0.25 fitted better. σ is the width of the derivative filter. A wider filter gives smoother gradients, which may matter more once the closure is driven by the ROM's own state. Seed 1 confirms the ranking (section 4).

**What is still wrong with the best model:**
- crests are 6–10% too high from t ≈ 50;
- the shape error on the wave backs, shared by every learned closure, remains (Part 6 of the 17 Nov explainer).

## 3. Why the ROM is now 6.8× faster

The closure is S = N(state) − N(h, h³, 0, 0, 0). The ROM used to evaluate the network twice per cell at each of the four RK stages.
- **Table:** the second term depends only on h for a given case (Re, Ca and β are fixed in a run). So the code now evaluates it once at startup on a fine grid in h and interpolates. h = 1 is exactly a grid point, so a flat film stays exactly flat.
- **Threads:** the remaining per-cell network evaluation is split over CPU cores with OpenMP.

**Check:** results agree with the old code to 1e-9 in h, and 1 and 4 threads give identical numbers. One time step went from 0.164 s to 0.024 s. A 2000-time-unit run now takes about 3 hours instead of about 18. That is what makes the broadband test affordable.

## 4. The broadband held-out test (seed 0 done, 19 Nov)

The oracle case is one monochromatic case, while the closures were trained on broadband data. The plan's real closed-loop test (gate GD) uses the 7 held-out broadband cases.

**Why statistics and not point-by-point errors.** The held-out DNS runs saved h only every 10 time units. That is too coarse to feed the DNS inlet into the ROM, as the oracle case does. So the ROM starts from a flat film with the same type of inlet noise as the DNS (same amplitude, number of modes and cut-off frequency) but its own random phases. The waves are therefore a different realisation, and we compare statistics over t = 1000–2000:
- the distribution of h (std, skewness, and a distance between the ROM and DNS distributions);
- the height of the main crests;
- how the wave amplitude grows and saturates along x (envelope error);
- the mean wavelength.

**Runs:** Lavalle, the previous MLP, structured V1 and the two new MLPs, in all 7 cases. The seed-1 runs of the two new MLPs are still running.

**What the seed-0 runs show** (table in the results doc; figures `broadband_metrics.png`, `broadband_waves.png`):
- **The gain on the oracle case carries over.** The σ 0.5 MLP is the best closure on 4 of the 6 cases it completed and ties with V1 on Re 20. On four of them its h distribution is about 2.5–5 times closer to the DNS than Lavalle's, and its crest heights are within 5%. Lavalle's crests are 6–28% too low in every case. The previous MLP overshoots, by a factor of 2 at Re 200.
- **But two runs blew up.** The σ 0.25 run on Re 40 Ca 0.04 and the σ 0.5 run on Re 80 β 90° stopped with loss of hyperbolicity: the equations stopped being well-posed, which the network does not prevent. No other closure failed. A model that fails on 1 of 7 cases is not yet usable as a general closure, so this is now the main open problem.
- **Where it is still wrong:** on Re 40 Ca 0.04, the most capillary case, the σ 0.5 MLP is worse than Lavalle (crests 16% too high). At Re 200 β 6.4° its waves are about twice too long, even though the distribution of h is close.

**Seed 1 (ROM tests):** the linear growth is as good as seed 0, and σ 0.5 beats σ 0.25 again on the oracle case (h error 0.37 at t = 50 against 0.48). So the σ ranking holds across both seeds. The t = 100 error differs by about 0.15 between seeds (0.62 and 0.75).

## 5. Why the two runs blew up, and the safeguard (19 Nov)

**What happened.** In both failed runs the film first got thinner than anything in the training data, and the network then extrapolated.
- h dropped below 0.28 (the lowest the DNS training data goes) from t ≈ 110 in the Re 80 β 90° run and t ≈ 420 in the Re 40 Ca 0.04 run. The DNS minima of these cases are 0.18 and 0.29.
- By the end, 5–9% of the domain was a near-dry film, down to h ≈ 0.01.
- There, the closure S reached 10–15. In training its typical size is 0.17, so the network was producing values about 60 times larger than anything it was fitted to.

**The "loss of hyperbolicity" message was a symptom, not the cause.** For this closure the hyperbolicity check reduces to 4·dP/dh, which is always positive. The code computed it as a difference of two large numbers, and rounding made it negative once h was tiny. That is now computed exactly, but on its own this would only let the film thin further.

**The safeguard,** part of the closure, off unless switched on:
- **Clamp:** each network input is limited to the range of the training data (0.01–99.99 percentiles). A narrower range (0.1–99.9) already touched real wave crests and troughs on the oracle case.
- **Taper:** S is faded smoothly to zero for h below 0.25, and is zero below 0.15, so in near-dry film the ROM falls back to Lavalle. Lavalle survives these cases.
- Both thresholds come from the training data and the DNS minima, not from tuning on the test cases.
- A flat film stays exactly flat.

**Status:** the oracle, comb and all 7 broadband cases are being rerun with the safeguard. It passes if every case completes and the results that were already good do not change by more than about 0.01.

## 6. Making it journal quality (plan, not started)

The full plan is `docs/plans/flat-mlp-journal-plan.md`. In short:
- **No new architecture.** Training takes a minute and the held-out error stops improving by epoch 60, so the model is converged. Every learned closure shares the same wave-back error, which points to missing input information, not to too small a network.
- **What a reviewer will ask for, and the plan provides:**
  - a learning-rate schedule with validation stopping, and loss curves;
  - an ablation of width/depth, inputs (adding q_xx, h_xxx or a second filter width) and σ, with 3 seeds each;
  - a seed spread on every result.
- **Fair model selection.** So far the 7 broadband test cases were also looked at when choosing σ. From now on, choices are made on a separate validation set: the oracle case, the Re 40 Ca 0.002 dense restart, and a held-out time window of training cases. The broadband test is then run once on the final model.
- **Possibly a seed ensemble** as the final model (average S over 3–5 networks).
- **Later, only if needed:** non-local or memory inputs, or short-rollout fine-tuning (Phase F). Closed forms (Phase E) are not planned, per your 18 Nov decision.
