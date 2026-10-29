# Working on HiPerGator (and keeping the laptop in sync)

## One-time migration (October 2026 restructure)

The old checkout at `/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml` holds all the DNS output, closure data, ML caches and the Python env. **Keep it exactly where it is** and treat it as the *data store*. Do **not** `git pull` the restructure into it. Removing the binaries and logs from git would make that pull delete them from the store as well.

```bash
# 1. Freeze the old checkout as the data store (no more commits from there)
cd /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml
git status          # commit/push anything pending to main FIRST, then stop using it for git

# 2. Fresh clone of the code
mkdir -p /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code
cd /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code
git clone https://github.com/krishnap561997/thinfilm-closure-ml.git
cd thinfilm-closure-ml

# 3. Environment variables (add to ~/.bashrc)
echo 'export THINFILM_ROOT=/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml' >> ~/.bashrc
echo 'source $THINFILM_ROOT/env.sh' >> ~/.bashrc

# 4. Commit identity (commits currently show krishnap.kalivel@loginN.ufhpc)
git config --global user.name  "Krishnaprasad K.A."
git config --global user.email "121191287+krishnap561997@users.noreply.github.com"

# 5. Notebook hygiene (once per clone)
$THINFILM_ENV/bin/pip install nbstripout && $THINFILM_ENV/bin/nbstripout --install
```

On the laptop, clone into a folder **outside OneDrive** (for example `C:\dev\thinfilm-closure-ml`). The OneDrive copy can then be archived or deleted.

## Day-to-day

- **Before starting:** `git pull`.
- **After a session:** `git add -p`, commit with a descriptive message, then `git push`.
- **Moving the science forward:** if a finding, decision or figure moved the science forward, update `docs/project-status.md` or `docs/decisions.md` in the same commit.
- **Keep data out of git:** new data goes in `$THINFILM_STORE` and gets a manifest row, never a commit.
- **Tag deliverables:** for example `git tag onr-2026-10`, `git tag paper1-submitted`, then `git push --tags`.

## Running a new DNS case

```bash
case=Re080_Ca0_020_theta90
dst=$THINFILM_DATA/BasC/param_space/$case
mkdir -p $dst && cd $dst
cp $THINFILM_ROOT/dns/production/{filmflow0.c,filmflow_restart.c,Makefile,run.sh} .
cp $THINFILM_ROOT/dns/production/cases/$case/params.in .   # or write a new one
make && make run
```

Then refresh the manifest and commit:

```bash
python $THINFILM_ROOT/datasets/update_manifest.py   # fills counts / t_last / status / closure_extracted
# then edit split + notes by hand, and commit datasets/MANIFEST_dns.csv
```

`dns/production/Makefile` builds against `~/basilisk-old/src`. That is the only Basilisk compatible with the HDF5 headers, and all production cases use it.

## Claude Code in VS Code

- Claude Code reads `CLAUDE.md` automatically, and the `docs/` files carry the rest of the context.
- Put personal preferences that are not specific to this repo (editor habits, module aliases) in `~/.claude/CLAUDE.md` on HiPerGator.
- The claude.ai project mirrors `docs/project-status.md`. When the two disagree, the repo wins.
