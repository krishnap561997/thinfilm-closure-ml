# APS DFD 2026 Interact session: e-poster story (draft v2, 19 Nov)

_For Krishna. A working document, edited together. It covers the e-poster only: one 16:9 slide shown on an LCD monitor from your laptop for the 40-minute discussion (DFD-Interact I or II, Sunday 22 Nov, Orlando). You are preparing the 2-minute talk slide yourself._

**Changes from v1:**
- talk section removed;
- less detail: one ML model on the poster, few numbers, no model variants (σ 0.25, structured V1) and no run-by-run accounting;
- authors added;
- figure list added for your decisions (bottom).

## Title and authors

**Machine learning-augmented reduced-order models for thin film flow**

Krishnaprasad K.A., S. Moghaddam, S. Balachandar. Department of Mechanical and Aerospace Engineering, University of Florida.

Acknowledgement line: ONR, grant number [Krishna fills in].

## Main message (title strip)

**A neural-network closure, constrained by two pieces of physics, lets a cheap two-equation film model reproduce DNS waves much better than the classical Lavalle closure, including at parameters it was not trained on.**

The two constraints:
1. **Flat film exact:** the learned correction is zero for a uniform film.
2. **Linear stability:** small waves grow at the Orr–Sommerfeld rate.

## Format

- **Canvas:** 16:9, built at 36 × 20.25 in so the labmate poster's type sizes carry over unchanged: title 60 pt, headings 28 pt, body 20 pt, captions 18 pt.
- **Style:** Georgia, UF blue title, orange section headings, black-bordered columns, UF logo top-left (as in Jungyun Kim's poster).
- **Layout:** three columns read left to right, two panels each. A take-home box at the bottom of column 3.
- **Text:** much lighter than a printed poster, because it is read on a screen. Each panel gets one claim sentence as its heading, a figure, and at most three short bullets.

## Column 1: the problem

**Panel 1: falling films and why we need a cheap model.**
- Film on an incline forced by broadband inlet noise. Small disturbances grow into large solitary-like waves.
- Interface-resolved DNS (Basilisk) over a range of Re, Ca and inclination β. One DNS case costs about 460 core-hours.
- A depth-averaged (two-equation) model is far cheaper, but only as good as its closure.

**Panel 2: the closure is the bottleneck.**
- The Lavalle model: depth-averaged mass and momentum equations for h and q, with the terms that need closure highlighted (layout follows `Presentations/Meeting_09292026.pptx`, slides 2–3).
- Oracle test: run the ROM with the exact closure extracted from the DNS. From the same DNS state, its error stays about three times smaller than Lavalle's (one sentence; the curve is in Panel 3).
- So the depth-averaged equations are fine; the closure is what limits the ROM.

## Column 2: learning the closure

**Panel 3: a plain neural network fits the data but drifts in the model.**
- A small MLP trained on DNS closure terms with local inputs (h, q and their derivatives) removes most of Lavalle's error a priori, including on held-out cases.
- Run inside the ROM from a DNS state, it first tracks the constrained model, then drifts after t ≈ 50: crests grow 25% too tall and the error passes Lavalle's at t ≈ 80.
- Cause: it adds a spurious source to a flat film and gets the growth of small waves wrong.

**Panel 4: two physics constraints fix the drift.**
- Flat film exact by construction: S = N(state) − N(flat film at the same h).
- Orr–Sommerfeld growth rates enforced in the training loss.
- Derivatives are taken with a Gaussian filter of width σ, the same in training and in the ROM. (The finite-difference comparison stays off the poster.)
- Linear-wave test (Re 20 / 120 / 200): the constrained model follows Orr–Sommerfeld and the DNS near the most unstable frequency, where Lavalle underpredicts growth by up to 2× at Re 120–200. It overpredicts growth at the highest frequencies (an open point if asked).

## Column 3: results and what is next

**Panel 5 (main panel): held-out cases, long times.**
- The ROM runs from a flat film with its own inlet noise to t = 2000 and is compared statistically with the DNS.
- On most held-out cases, the distribution of film thickness is several times closer to the DNS than Lavalle's, and crest heights are close to the DNS, where Lavalle's are too low.
- Shown: two cases where it works (Re 20 Ca 0.01 β 45, Re 120 Ca 0.02 β 90) and one where it fails (Re 200 Ca 0.01 β 6.4: ML waves about twice the DNS wavelength). The plain MLP is far off at Re 200.

**Panel 6: open problems.**
- Robustness: in a few runs, the film thins below anything in the training data and the network extrapolates. A fix (limiting the inputs to the training range and switching the correction off in very thin film) is being tested.
- Cost: the ML ROM is already about 30× cheaper than DNS; the network evaluation is not yet optimised.
- Next: a separate validation set, an ensemble of networks, and training on short model rollouts.

**Take-home (3 bullets):**
1. The closure, not the depth-averaged equations, limits the ROM.
2. A learned closure must be exact for a flat film and match linear theory, or it drifts in the model.
3. With both, the ML ROM beats Lavalle at parameters it has not seen. Robustness outside the training range is the open problem.

## Figures: decide here

**Visual conventions for all poster figures:**
- the same four labels and colours everywhere: DNS / Orr–Sommerfeld (dark navy), Lavalle (grey), unconstrained MLP (orange), physics-constrained MLP (teal);
- σ 0.5 only, with no σ 0.25 or structured V1;
- larger fonts than the report versions.

| # | Panel | Figure | Source | Decision (19 Nov) |
|---|---|---|---|---|
| F1 | 1 | Film schematic and DNS interface snapshot | your ONR report figures (to upload) | **yes** |
| F1b | 1 | `anim01_h_full_domain.mp4` looping in Panel 1 | P1 folder | **yes** |
| F2 | 1 | (Re, Ca, β) case map | new | **dropped** |
| F3 | 2 | Depth-averaged equations with colour-coded unclosed terms M, P_x, T_w, D_x; Lavalle's closures; S_ML | `Presentations/Meeting_09292026.pptx`, slides 2–3 | **yes, retypeset**; Panel 2 has no other figure |
| F4 | 2 | Oracle error curve | — | **dropped**: the exact-closure curve moves into F5 |
| F5 | 3 | Oracle case, h error against time: Lavalle, exact closure, plain MLP, constrained MLP; small inset of h(x) at a late time with the plain MLP's crests too tall | oracle runs (`ml_sum_hhx`, `mlp_s0.5_L10_seed0`) | **yes, curve + inset** |
| F6 | 4 | Method schematic: DNS → closure terms → constrained network → ROM, with S = N(state) − N(flat) | new | **yes**, stays in Panel 4 |
| F7 | 4 | Growth rate against frequency, Re 20 / 120 / 200: OS, comb DNS, Lavalle, plain MLP, constrained MLP | regenerate after job 52044322 | **yes, all three Re**; adds the plain MLP |
| F8 | 5 | h(x) at t = 2000 and h PDFs: Re 20 Ca 0.01 β 45, Re 120 Ca 0.02 β 90, Re 200 Ca 0.01 β 6.4; DNS, Lavalle, plain MLP, constrained MLP | regenerate | **yes, 4 curves** |
| F9 | 5 | Per-case metrics | — | **dropped** |
| F10 | 5 | Dense-restart short-time error | — | **dropped** |

**Labels and colours (all figures):** DNS / Orr–Sommerfeld (dark navy), Lavalle (grey), plain MLP (orange; `ml_sum_hhx`, local inputs, no constraints), physics-constrained MLP (teal; `mlp_s0.5_L10_seed0`). One "plain MLP" label means the same model in Panels 3, 4 and 5.

**Figures are generated only after your approval.**

## Figure files (generated 19 Nov, my decisions)

Built with `scripts/meetings/aps_dfd_2026_figs.py` into `docs/meetings/2026-11-aps-dfd/poster_figs/`, at poster size (11 in wide = one column, 18–20 pt text). The ONR figures are vector PDFs in `docs/meetings/onr figures/`: place and crop them in PowerPoint (no PDF rasteriser on HiPerGator).

| Panel | Figure | File |
|---|---|---|
| 1 | Film schematic + inlet noise | ONR `Figure1.pdf`, crop panel D |
| 1 | Looping DNS video | `anim01_h_full_domain.mp4` |
| 2 | Equations, unclosed terms in colour; Lavalle and ML ROM lines | `p2_equations.png` |
| 1 | Lavalle against DNS, monochromatic: Re 19.3 β 6.4 (agrees) and Re 120 β 45 (crests far too low), side by side; ONR Figure 2 redrawn | `p1_lavalle_mono.png` (`aps_dfd_2026_figs.py lavalle_mono`) |
| 3 | A priori: MLP prediction against DNS closure terms | ONR `Figure5.pdf`, crop two rows (e.g. Re 120 Ca 0.005 β 90 and Re 200) |
| 3 | A priori: DNS ΔM_x + ΔT_w against the plain MLP, held-out Re 120 Ca 0.01 β 45 at t = 2000 (snapshot NRMSE 0.15; replaces the "figure to come" placeholder) | `p3_apriori_fit.png` (`aps_dfd_2026_figs.py apriori`) |
| 3 | Oracle case: h error against time (Lavalle, exact closure, plain MLP, constrained MLP) and h(x) at t = 100 (plain MLP crests too tall) | `p3_oracle_error.png` |
| 3 | Animation, oracle case t = 0–100: Lavalle, plain MLP, constrained MLP, each over the DNS (36 s loop); still of the last frame | `p3_oracle_race.mp4`, `p3_oracle_race_last.png` (`scripts/meetings/aps_dfd_2026_anim.py`) |
| 4 | Method schematic with the two constraints | `p4_method.png` |
| 4 | Growth rate against frequency, Re 20 / 120 / 200: OS, DNS, Lavalle, constrained MLP | `p4_linear_growth.png` |
| 5 | Broadband case Re 120 Ca 0.02 β 90: h(x) at t = 2000 | `p5_heldout_snapshot.png` |
| 5 | h PDFs, Re 20 β 45, Re 120 β 90, Re 200 β 6.4 (linear axis) | `p5_heldout_pdf.png` |

**Choices made without you (change any):**
- Panel 2 adds ONR Figure 2B: one look shows why the closure matters.
- Panel 3 uses ONR Figure 5 for "fits the data" and the oracle curve for "drifts in the model". The oracle curve is honest: plain and constrained MLP track each other to t ≈ 40, then the plain MLP drifts.
- Panel 4's growth figure has no plain-MLP points (its comb runs were not ready); Panels 3 and 5 carry the plain-MLP failure.
- ONR Figures 1A/B/C/E, 3, 4 and 6 are left for questions.

## Open items

1. **ONR report figures:** upload them, and say which ones go in Panel 1.
2. **Safeguard validation (`mlp_s0.5_L10_seed0_sg`):** oracle and comb results are unchanged with the safeguard on (19 Nov). The broadband runs (job 52043204) decide it: if all 7 complete before Sunday, Panel 6 can say "fixed"; otherwise it stays "being tested".
4. **Finite-difference MLP (jobs 52043498, 52043502):** a priori 40% worse than σ 0.5. Kept off the poster unless the a posteriori runs change the picture.
3. **Cost numbers:**
   - DNS about 460 core-h (Re 120 Ca 0.01 β 45, t = 2000);
   - Lavalle ROM about 0.3 core-h;
   - ML ROM about 10–15 core-h after the 18 Nov speed-up (0.024 s per step on 4 threads);
   - "about 30×" is the conservative figure.
