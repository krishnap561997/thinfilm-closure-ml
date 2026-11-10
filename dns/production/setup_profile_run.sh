#!/bin/bash
# Set up a broadband restart with in-situ 1D column profiles (filmflow_restart_profiles.c).
#
#   setup_profile_run.sh CASE T_START DURATION RUN_DIR [T_PROF] [T_OUT] [BUILD_DIR]
#
# CASE      case name under $THINFILM_DATA/BasC/DNS/param_space (its params.in is copied)
# T_START   restart time; dump-<T_START, %06d> of the case is linked as restart.dump
# DURATION  run length (T_END = T_START + DURATION)
# RUN_DIR   new run directory (must not exist)
# T_PROF    profile interval (default 0.01)
# T_OUT     2D snapshot interval (default DURATION, i.e. one snapshot at T_END)
# BUILD_DIR directory with the compiled filmflow_restart_profiles (default
#           $THINFILM_DATA/BasC/DNS/dense/build); source and Makefile are copied for provenance.
# Dumps are disabled (T_DUMP = 100000). Nothing is submitted.
set -euo pipefail
ROOT="${THINFILM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${ROOT}/env.sh"
CASE=$1; T0=$2; DUR=$3; RUN=$4; TPROF=${5:-0.01}; TOUT=${6:-$DUR}
BUILD=${7:-${THINFILM_DATA}/BasC/DNS/dense/build}
SRC="${THINFILM_DATA}/BasC/DNS/param_space/${CASE}"
DUMP="${SRC}/$(printf 'dump-%06d' "${T0%.*}")"
[[ -f "${SRC}/params.in" ]] || { echo "no params.in in ${SRC}" >&2; exit 1; }
[[ -f "${DUMP}" ]] || { echo "no ${DUMP}" >&2; exit 1; }
[[ -x "${BUILD}/filmflow_restart_profiles" ]] || { echo "build ${BUILD}/filmflow_restart_profiles first (make profiles)" >&2; exit 1; }
[[ -e "${RUN}" ]] && { echo "${RUN} exists" >&2; exit 1; }
mkdir -p "${RUN}"
TEND=$(python3 -c "print(${T0} + ${DUR})")
# params.in without the keys we set, then our overrides
grep -v -E '^\s*(T_END|T_OUT|T_DUMP|T_PROF|RESTART_FILE)\s*=' "${SRC}/params.in" > "${RUN}/params.in"
cat >> "${RUN}/params.in" <<EOF
T_END = ${TEND}
T_OUT = ${TOUT}
T_DUMP = 100000
T_PROF = ${TPROF}
RESTART_FILE = restart.dump
EOF
ln -s "${DUMP}" "${RUN}/restart.dump"
cp "${BUILD}/filmflow_restart_profiles" "${RUN}/"
cp "${BUILD}/filmflow_restart_profiles.c" "${BUILD}/Makefile" "${RUN}/"
{
  echo "case        : ${CASE}"
  echo "restart     : ${DUMP} (t = ${T0})"
  echo "t_end       : ${TEND}, T_PROF = ${TPROF}, T_OUT = ${TOUT}"
  echo "git commit  : $(git -C "${ROOT}" rev-parse HEAD)$(git -C "${ROOT}" diff --quiet -- dns || echo ' (dns/ modified)')"
  echo "created     : $(date)"
} > "${RUN}/RUN_INFO.txt"
echo "set up ${RUN} (t ${T0} -> ${TEND})"
