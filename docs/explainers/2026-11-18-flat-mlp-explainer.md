# 18 Nov 2026: flat-film-exact MLP, all seed-0 results; faster ROM; broadband test

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

## 5. What comes after (proposed, not started)

The broadband results confirm the MLP but add a robustness problem. In rough order of cost:
0. **Hyperbolicity safeguard (new, 19 Nov).** Limit the closure in the ROM where the system gets close to losing hyperbolicity, so no run fails. This comes first, since a closure that crashes cannot be reported as the model.
1. **Ensembles.** Average 3–5 seeds in the ROM. This is cheap, and averaging usually reduces closed-loop drift.
2. **Wider inputs.** Add q_xx and h_xxx, or a second filter width, to target the wave-back shape.
3. **Short-rollout fine-tuning (plan Phase F).** A differentiable ROM, trained on 2–5-time-unit rollouts from DNS states. This is the most direct way to train for closed-loop accuracy, but it is the most work. Ask before starting.

**Not planned now:** closed forms and symbolic regression (Phase E), per your 18 Nov decision.
