#!/usr/bin/env bash
# Phase-2 capture driver for Halogen-vs-MiMo runs on the Strix Halo
# (gfx1151) target box.  STATIC-ANALYSIS PHASE NEVER RUNS THIS: it is
# executed only once hardware arrives.  Collects, per scenario:
#
#   1. rocprofv3 system trace   - timelines: HIP/HSA API, kernel dispatches,
#                                 memory copies/allocations, scratch (sync +
#                                 allocation evidence; kernel_trace.csv also
#                                 carries per-dispatch VGPR/SGPR/LDS usage)
#   2. rocprofv3 PMC passes     - multi-pass counters (waves/occupancy,
#                                 memory throughput); availability-filtered
#                                 via `rocprofv3 --list-avail`
#   3. amd-smi metric poller    - SCLK/MCLK, power, temperatures (1 Hz)
#   4. host CPU / RSS poller    - /proc sampler (1 Hz)
#
# Exact usage (recorded for the Phase-2 handoff):
#   ./capture_rocprofv3.sh -- <application> [args...]        # all passes
#   OUT_DIR=runs/p1 ./capture_rocprofv3.sh -- ./bench --cfg matrix.yaml
#   TRACE_ONLY=1 ./capture_rocprofv3.sh -- ./app             # skip PMC
#   PMC_ONLY=1   ./capture_rocprofv3.sh -- ./app             # skip trace
#
# Prereqs on the target box (recorded exact commands):
#   export PATH=$PATH:/opt/rocm/bin
#   sudo /opt/rocm/bin/amd-smi set --perf-level STABLE_STD   # PMC on RDNA
#   # alternative without sudo rights on amd-smi:
#   #   sudo sh -c 'echo profile_standard > \
#   #     /sys/class/drm/card0/device/power_dpm_force_performance_level'
#
# Every counter below is filtered against `rocprofv3 --list-avail` output;
# unavailable counters are recorded as notes (never silently dropped).

set -euo pipefail

ROCPROF="${ROCPROF:-rocprofv3}"
AMDSMI="${AMDSMI:-amd-smi}"
ROCMSMI="${ROCMSMI:-rocm-smi}"
OUT_DIR="${OUT_DIR:-capture_$(date +%Y%m%d_%H%M%S)}"
PMC_INPUT="${PMC_INPUT:-$(dirname "$0")/pmc_counters.txt}"
POLL_HZ="${POLL_HZ:-1}"

# Desired counter set (phases split into <=4-counter groups for multi-pass;
# per-GPU-block limits vary on gfx1151, list-avail decides inclusion).
PMC_GROUPS=(
  "GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES SQ_WAIT_ANY"
  "SQ_WAVE_CYCLES SQ_INSTS_VALU SQ_INSTS_SCALAR SQ_INSTS_VMEM_RD"
  "FETCH_SIZE WRITE_SIZE"
  "OccupancyPercent"
)

usage() { grep '^#' "$0" | sed -n '2,25p'; }

if [ "${1:-}" = "--help" ] || [ $# -eq 0 ]; then usage; exit 0; fi
if [ "${1:-}" != "--" ]; then echo "expected '--' then application" >&2; exit 2; fi
shift

mkdir -p "$OUT_DIR"
RUN_META="$OUT_DIR/run_meta.txt"
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "host: $(uname -a)"
  echo "app: $*"
  echo "rocprofv3: $($ROCPROF --version 2>&1 | head -1 || echo unavailable)"
  echo "amd-smi: $($AMDSMI --version 2>&1 | head -1 || echo unavailable)"
} > "$RUN_META"

# ---- counter availability gate -----------------------------------------
AVAIL="$OUT_DIR/pmc_available.txt"
"$ROCPROF" --list-avail > "$AVAIL" 2>&1 || true
selected=()
notes="$OUT_DIR/pmc_unavailable_notes.txt"
: > "$notes"
for group in "${PMC_GROUPS[@]}"; do
  for c in $group; do
    if grep -qw "$c" "$AVAIL"; then selected+=("$c")
    else echo "counter unavailable on this agent: $c" >> "$notes"; fi
  done
done
echo "selected_counters: ${selected[*]:-none}" >> "$RUN_META"
cat "$notes" >> "$RUN_META" || true

# ---- perf level for RDNA PMC -------------------------------------------
perf_note() {
  if "$AMDSMI" set --perf-level STABLE_STD > "$OUT_DIR/perf_level.txt" 2>&1; then
    echo "perf_level: STABLE_STD via amd-smi" >> "$RUN_META"
  else
    echo "perf_level: amd-smi set failed (see perf_level.txt); PMC blocks may be clock-gated" >> "$RUN_META"
  fi
}

# ---- background pollers -------------------------------------------------
start_pollers() {
  # GPU: SCLK/MCLK/power/temps at POLL_HZ (amd-smi primary; rocm-smi fallback)
  (
    echo "epoch,sclk_mhz,mclk_mhz,power_w,temp_c,tempedge_c" \
      > "$OUT_DIR/gpu_metrics.csv"
    while :; do
      line="$("$AMDSMI" metric --gpu 0 --clock --mem-clock --power --temp 2>/dev/null \
        | tr '\n' ' ' | tr -s ' ')"
      if [ -n "$line" ]; then
        echo "$(date +%s),$line" >> "$OUT_DIR/gpu_metrics.csv"
      else
        echo "$(date +%s),$("$ROCMSMI" -i 0 --showclocks --showpower --showtemp \
          --csv 2>/dev/null | tail -1 | tr -d '\r')" >> "$OUT_DIR/gpu_metrics.csv"
      fi
      sleep "$POLL_HZ"
    done
  ) &
  GPU_POLLER=$!

  # host: CPU + process RSS from /proc at POLL_HZ
  (
    echo "epoch,host_busy_pct,proc_rss_kb,proc_cpu_pct" > "$OUT_DIR/host_metrics.csv"
    prev_total=0; prev_busy=0; prev_cpu=0
    pid=""
    while :; do
      [ -z "$pid" ] && pid=$(pgrep -n -f "$(basename "$1")" || true)
      read -r _ user nice system idle iowait irq softirq _ < \
        /proc/stat 2>/dev/null || break
      busy=$((user + nice + system + irq + softirq))
      total=$((busy + idle + iowait))
      cpu_pct=0
      if [ "$prev_total" -gt 0 ]; then
        cpu_pct=$((100 * (busy - prev_busy) / (total - prev_total + 1)))
      fi
      prev_busy=$busy; prev_total=$total
      rss=0; cput=0
      if [ -n "$pid" ] && [ -r "/proc/$pid/statm" ]; then
        rss=$(awk '{print $2 * 4}' "/proc/$pid/statm" 2>/dev/null || echo 0)
        cput=$(awk '{print $14 + $15}' "/proc/$pid/stat" 2>/dev/null || echo 0)
      fi
      echo "$(date +%s),$cpu_pct,$rss,$cput" >> "$OUT_DIR/host_metrics.csv"
      sleep "$POLL_HZ"
    done
  ) &
  HOST_POLLER=$!
}

stop_pollers() {
  kill "${GPU_POLLER:-}" "${HOST_POLLER:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}

# ---- passes -------------------------------------------------------------
if [ -z "${PMC_ONLY:-}" ]; then
  perf_note
  start_pollers
  # Pass T: full trace (timelines + allocations + sync + per-dispatch regs)
  #   outputs: <pid>_kernel_trace.csv <pid>_hip_api_trace.csv
  #            <pid>_hsa_api_trace.csv <pid>_memory_copy_trace.csv
  #            <pid>_memory_allocation_trace.csv <pid>_scratch_memory_trace.csv
  ( cd "$OUT_DIR" && "$ROCPROF" --sys-trace --stats --output-format csv \
      --prefix trace -- "$@" ) 2>&1 | tee "$OUT_DIR/trace.log"
  stop_pollers
fi

if [ -z "${TRACE_ONLY:-}" ] && [ "${#selected[@]}" -gt 0 ]; then
  perf_note
  # Pass P: counter collection; one --pmc flag per hardware pass.
  pmc_flags=()
  for group in "${PMC_GROUPS[@]}"; do
    keep=()
    for c in $group; do
      case " ${selected[*]} " in *" $c "*) keep+=("$c");; esac
    done
    [ "${#keep[@]}" -gt 0 ] && pmc_flags+=(--pmc "${keep[@]}")
  done
  ( cd "$OUT_DIR" && "$ROCPROF" "${pmc_flags[@]}" --output-format csv \
      --prefix pmc -- "$@" ) 2>&1 | tee "$OUT_DIR/pmc.log"
fi

# PC sampling for symbol->address correlation (best effort; separate pass)
if [ "${PC_SAMPLING:-1}" = "1" ] && [ -z "${TRACE_ONLY:-}" ]; then
  ( cd "$OUT_DIR" && "$ROCPROF" --pc-sampling --pc-sampling-interval 1000 \
      --output-format csv --prefix pcs -- "$@" ) \
    > "$OUT_DIR/pc_sampling.log" 2>&1 || \
    echo "pc-sampling pass unavailable on this agent" >> "$RUN_META"
fi

{
  echo "outputs:"
  ls -1 "$OUT_DIR"
} >> "$RUN_META"
echo "capture complete: $OUT_DIR"
