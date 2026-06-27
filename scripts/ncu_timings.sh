#!/usr/bin/env bash
#
# ncu_timings.sh — profile HeddleMD's per-step kernels with NVIDIA Nsight
# Compute (ncu).
#
# Run this on BARE METAL (the host), not inside the rootless podman
# container: ncu needs GPU performance counters, which a rootless
# container cannot obtain (ERR_NVGPUCTRPERM). If you must keep the
# host's default `NVreg_RestrictProfilingToAdminUsers=1`, run as a host
# admin; otherwise set it to 0 and reload the driver / reboot.
#
# Usage (run from the repository's top-level directory):
#   ./scripts/ncu_timings.sh [size] [kernel-regex] [ncu-set]
#
#   size          8192 (default) or 65536 — selects examples/spc-water-<size>
#   kernel-regex  ncu -k regex (default: the packed pair-force kernel,
#                 "heddle_jit_composed_pair_force_f", which matches the _f
#                 and _fev variants). Examples:
#                   "class_accumulator_memset"          the memset anomaly
#                   "heddle_jit_composed_pair_force"    all pair-force passes
#                   "shake|rattle"                      constraints
#                   "."                                 every kernel (slow)
#   ncu-set       ncu --set (default: detailed). Valid: default (fast: SOL +
#                 occupancy + launch), detailed (adds warp-stall + memory
#                 analysis), full (everything, slowest), roofline, source.
#
# Env overrides:
#   NCU_SKIP   matched launches to skip before profiling (default 20 —
#              steps past minimization + the early transient)
#   NCU_COUNT  matched launches to profile (default 5)
#   NCU        path to the ncu binary (default: ncu from PATH)
#
# Artifacts land in profile_out/ncu/:
#   ncu-<size>-<stamp>.ncu-rep    open in `ncu-ui` for the full analysis
#   ncu-<size>-<stamp>.details.txt   per-kernel text report
#   ncu-<size>-<stamp>.log           raw profiler output

set -euo pipefail

SIZE="${1:-8192}"
KREGEX="${2:-heddle_jit_composed_pair_force_f}"
NCU_SET="${3:-detailed}"
SKIP="${NCU_SKIP:-20}"
COUNT="${NCU_COUNT:-5}"
NCU="${NCU:-ncu}"

# Repository root is the parent of this script's scripts/ directory, so the
# script works invoked as ./scripts/ncu_timings.sh from the top level.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLE="$REPO/examples/spc-water-$SIZE"
BIN="$REPO/target/release/heddlemd"
OUTDIR="$REPO/profile_out/ncu"
STAMP="$(date +%Y%m%d-%H%M%S)"
REPORT="$OUTDIR/ncu-${SIZE}-${STAMP}"
CFG="$OUTDIR/water_ncu_${SIZE}.in.toml"

# ---- preflight ----------------------------------------------------------
# Resolve ncu even under `sudo`, where root's PATH often omits the CUDA dir.
if ! command -v "$NCU" >/dev/null 2>&1; then
  for cand in /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu \
              /usr/local/NVIDIA-Nsight-Compute*/ncu; do
    [ -x "$cand" ] && { NCU="$cand"; break; }
  done
fi
command -v "$NCU" >/dev/null 2>&1 || [ -x "$NCU" ] || {
  echo "ERROR: ncu not found. Install NVIDIA Nsight Compute (CUDA toolkit), or pass its" >&2
  echo "       full path: NCU=/usr/local/cuda/bin/ncu ./scripts/ncu_timings.sh ..." >&2
  exit 1
}
[ -d "$EXAMPLE" ] || { echo "ERROR: example dir not found: $EXAMPLE" >&2; exit 1; }
[ -f "$EXAMPLE/water.in.toml" ] || { echo "ERROR: $EXAMPLE/water.in.toml not found." >&2; exit 1; }

# Warn early if profiling is admin-restricted and we lack the capability
# (this is exactly the rootless-container situation).
if grep -q '^RmProfilingAdminOnly: 1' /proc/driver/nvidia/params 2>/dev/null; then
  if ! grep -qiE 'cap_sys_admin' < <(grep CapEff /proc/self/status 2>/dev/null | xargs -r -n1 2>/dev/null; capsh --print 2>/dev/null) ; then
    echo "WARNING: GPU profiling is admin-restricted (RmProfilingAdminOnly=1) and this"
    echo "         process may lack CAP_SYS_ADMIN. If ncu fails with ERR_NVGPUCTRPERM,"
    echo "         run on the host as admin, or set NVreg_RestrictProfilingToAdminUsers=0"
    echo "         on the host and reboot. Continuing..."
    echo
  fi
fi

if [ ! -x "$BIN" ]; then
  if [ "$(id -u)" -eq 0 ]; then
    # Avoid running cargo as root (would create a root-owned target dir
    # and use root's cargo home). Build as the normal user first.
    echo "ERROR: $BIN is missing. Build it as your normal user, then re-run:" >&2
    echo "       (cd '$REPO' && cargo build --release)" >&2
    exit 1
  fi
  echo "[ncu] release binary missing; building..."
  ( cd "$REPO" && cargo build --release )
fi
mkdir -p "$OUTDIR"

# ---- generate a graphs-disabled config ----------------------------------
# Derived from the tracked water.in.toml: force cuda_graphs_disable=true so
# each kernel is a separate launch ncu can target, and rewrite init/topology
# to absolute paths so the config can live in OUTDIR.
python3 - "$EXAMPLE" "$CFG" <<'PY'
import sys, re, pathlib
ex, out = sys.argv[1], sys.argv[2]
src = pathlib.Path(ex, "water.in.toml").read_text()
src = re.sub(r'(init|topology)\s*=\s*"([^"]+)"',
             lambda m: f'{m.group(1)} = "{pathlib.Path(ex, m.group(2))}"', src)
if re.search(r'cuda_graphs_disable\s*=', src):
    src = re.sub(r'cuda_graphs_disable\s*=\s*\w+', 'cuda_graphs_disable = true', src)
else:
    src = re.sub(r'\[simulation\]', '[simulation]\ncuda_graphs_disable = true', src, count=1)
pathlib.Path(out).write_text(src)
PY

# ---- profile ------------------------------------------------------------
echo "[ncu] system=spc-water-$SIZE  kernels=/$KREGEX/  set=$NCU_SET  skip=$SKIP  count=$COUNT"
rm -f "$OUTDIR"/water_ncu_"${SIZE}".out.*

set +e
( cd "$OUTDIR" && "$NCU" \
    --target-processes all \
    --set "$NCU_SET" \
    --kernel-name-base demangled \
    -k "regex:$KREGEX" \
    -s "$SKIP" -c "$COUNT" \
    --kill yes \
    -f -o "$REPORT" \
    "$BIN" run "$CFG" ) > "$REPORT.log" 2>&1
rc=$?
set -e
rm -f "$OUTDIR"/water_ncu_"${SIZE}".out.*

if grep -q ERR_NVGPUCTRPERM "$REPORT.log"; then
  cat >&2 <<EOF

ERROR: ERR_NVGPUCTRPERM — no permission for GPU performance counters.

Quickest fix on bare metal: run the profiler as root (real root has the
CAP_SYS_ADMIN the driver requires under RmProfilingAdminOnly=1):

  cargo build --release            # once, as your normal user
  sudo -E ./scripts/ncu_timings.sh $SIZE "$KREGEX" $NCU_SET
  # If sudo's PATH can't find ncu, pass it explicitly:
  #   sudo NCU="$NCU" ./scripts/ncu_timings.sh $SIZE "$KREGEX" $NCU_SET

Or lift the restriction host-wide so no sudo is needed (requires a reboot):

  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  sudo reboot
  # (reload instead of reboot, GPU must be idle:)
  # sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia

Full log: $REPORT.log
EOF
  exit 1
fi

if [ ! -f "$REPORT.ncu-rep" ]; then
  echo "ERROR: no report produced (ncu rc=$rc). Tail of log:" >&2
  tail -n 30 "$REPORT.log" >&2
  exit 1
fi

# ---- summary ------------------------------------------------------------
"$NCU" --import "$REPORT.ncu-rep" --page details > "$REPORT.details.txt" 2>"$REPORT.import.err" || true

echo
echo "===================== per-kernel headline ====================="
if [ -s "$REPORT.details.txt" ]; then
  # Loose match: kernel headers + the Speed-of-Light / occupancy / launch
  # metric lines. The full per-kernel report is in the .details.txt file.
  grep -E 'Throughput|Duration|Occupancy|Registers Per Thread|Block Limit|Waves Per SM|^ *[a-zA-Z].*\(.*\)' \
    "$REPORT.details.txt" || cat "$REPORT.details.txt"
else
  echo "WARNING: 'ncu --import --page details' produced no output. Error was:"
  sed 's/^/  /' "$REPORT.import.err" >&2
  echo "Open the report directly instead:  ncu-ui '$REPORT.ncu-rep'"
fi

echo
echo "Artifacts:"
echo "  GUI report : $REPORT.ncu-rep      (open with: ncu-ui '$REPORT.ncu-rep')"
echo "  Text report: $REPORT.details.txt"
echo "  Raw log    : $REPORT.log"
