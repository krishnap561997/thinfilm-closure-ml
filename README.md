# thinfilm-closure-ml

Machine-learning-augmented reduced-order model for thin liquid films down an inclined plane.
Basilisk DNS generates closure data for a depth-integrated two-equation model (Lavalle et al. 2015). ML learns corrections to its unclosed terms, and those corrections are embedded back into the ROM solver.

| Folder | Contents |
|---|---|
| `dns/` | Basilisk DNS: production sources, per-case `params.in`, HDF5 headers, tests |
| `rom/` | Thin-film ROM solvers (C++ `lavalle-cpp`, original MATLAB) |
| `pipeline/` | DNS → closure-term extraction |
| `analysis/` | Statistics of closure terms |
| `ml/` | ML closure models and training scripts |
| `notebooks/` | Exploratory and figure notebooks (outputs stripped) |
| `datasets/` | Manifests of DNS and ROM runs (data lives on /blue) |
| `docs/` | Equations, project status, data layout, HiPerGator workflow, decisions |

Start with `CLAUDE.md` for orientation and `docs/hipergator.md` for setup.
Pre-restructure history is tagged `pre-restructure-2026-10`.
