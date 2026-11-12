# Figures for the results write-ups

One folder per write-up in `docs/results/`, with the same name:

| Folder | Write-up |
|---|---|
| `budget-check/` | `../budget-check.md` |
| `dpx-closure/` | `../dpx-closure.md` |
| `functional-form-structure/` | `../functional-form-structure.md` |
| `mesh-sensitivity/` | `../mesh-sensitivity.md` |
| `mlp-heldout-eval/` | `../mlp-heldout-eval.md` |
| `oracle-a-posteriori/` | `../oracle-a-posteriori.md` |

Conventions

- Name files `fig<NN>_<what>.png` for results that work and `limit<NN>_<what>.png` for figures that show where something fails (and `.pdf` for vector figures); keep each PNG under about 400 KB so the repo stays small. Do not commit anything larger or any raw data.
- Put the script that makes a figure in `scripts/figures/<result>/` and name it after the figure, so each figure can be regenerated. Scripts read data from `$THINFILM_STORE`, never from this folder.
- Reference each figure from its write-up with a one-line caption saying what to look at and the case, time window and commit it was made from.
- `.gitignore` ignores `*.png` and `figures/` everywhere except `docs/figures/` and `docs/results/figures/`.
