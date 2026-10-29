# Source this file:  source $THINFILM_ROOT/env.sh
# Sets the paths used by the build files, SLURM scripts and docs. Override any of them before sourcing.
export THINFILM_ROOT="${THINFILM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export THINFILM_STORE="${THINFILM_STORE:-/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml}"
export THINFILM_DATA="${THINFILM_DATA:-$THINFILM_STORE/datasets}"
export THINFILM_ENV="${THINFILM_ENV:-$THINFILM_STORE/software/envs/thinfilm-ml-py311}"
