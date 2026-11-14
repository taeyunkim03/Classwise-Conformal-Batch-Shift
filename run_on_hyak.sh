#!/usr/bin/env bash
# Staged Hyak setup + pilot runner.
# Each stage adds ONE dependency, so if something breaks you know exactly where.
#
#   bash run_on_hyak.sh setup     # once: create env, install packages
#   conda activate cpsc           # <-- YOU must do this in your own shell.
#                                 #     The script runs in a subshell, so any
#                                 #     activation inside it dies on exit.
#   bash run_on_hyak.sh test      # stage 1: engine correctness (seconds)
#   bash run_on_hyak.sh smoke     # stage 2: pipeline runs on fake data
#   bash run_on_hyak.sh inspect   # stage 3: download real data, show its columns
#   bash run_on_hyak.sh pilot     # stage 4: the actual experiment
#
# Run them in order. Do not skip to `pilot`.

set -euo pipefail
ENV_NAME=cpsc
STAGE="${1:-help}"

# Fail loudly if the env is not active, rather than silently using system
# Python (which may have different package versions, or none at all).
require_env() {
  if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    echo "ERROR: conda env '$ENV_NAME' is not active (current: '${CONDA_DEFAULT_ENV:-none}')."
    echo
    echo "  Run this in YOUR shell first:"
    echo "      conda activate $ENV_NAME"
    echo
    echo "  (Activating inside this script would not persist -- it runs in a"
    echo "   subshell. You also need to re-activate after any new login or srun.)"
    exit 1
  fi
  echo "env OK: $CONDA_DEFAULT_ENV   python: $(command -v python)"
}

case "$STAGE" in

setup)
  echo ">>> creating conda env '$ENV_NAME'"
  conda create -n "$ENV_NAME" python=3.11 -y
  # Load conda's shell hook so `conda activate` works in a non-interactive
  # script; plain `conda activate` often fails without this on HPC systems.
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  pip install numpy scipy scikit-learn matplotlib scanpy anndata
  echo
  echo ">>> Packages installed. NOW RUN THIS YOURSELF:"
  echo
  echo "      conda activate $ENV_NAME"
  echo
  echo ">>> (The activation above happened inside this script's subshell and"
  echo ">>>  does NOT carry over to your shell.) Then:"
  echo "      bash run_on_hyak.sh test"
  ;;

test)
  require_env
  echo ">>> STAGE 1: engine correctness (needs only numpy + sklearn)"
  python tests_conformal.py
  echo
  echo ">>> If you saw 'ALL 11 CHECKS PASS', the engine is good."
  echo ">>> Next:  bash run_on_hyak.sh smoke"
  ;;

smoke)
  require_env
  echo ">>> STAGE 2: pipeline runs end-to-end on FAKE data (no download)"
  echo ">>> The NUMBERS here are meaningless. Only 'SMOKE TEST PASSED' matters."
  python smoke_test.py
  echo
  echo ">>> Next:  bash run_on_hyak.sh inspect"
  ;;

inspect)
  require_env
  echo ">>> STAGE 3: download the pancreas benchmark and print its structure"
  echo ">>> This is the first step that needs internet + disk."
  python pancreas_gonogo.py --inspect
  echo
  echo ">>> CHECK THREE THINGS in the output above:"
  echo "      1. it downloaded at all (if not, the figshare URL moved --"
  echo "         get the scIB pancreas h5ad elsewhere and use --data PATH)"
  echo "      2. there is a cell-type column ('celltype') and a technology"
  echo "         column ('tech'), and 'smartseq2' is one of the technologies"
  echo "      3. a donor-like column exists -- note its name, you need it below"
  echo
  echo ">>> Next:  bash run_on_hyak.sh pilot     (edit SPLIT_KEY first if needed)"
  ;;

pilot)
  # SPLIT_KEY="" means cell-level splitting (the scIB pancreas benchmark has no
  # donor/sample column, only 'tech' -- which is the SHIFT axis and therefore
  # must not be used to split calibration). Report as cell-level empirical
  # calibration, not a finite-sample biological guarantee.
  SPLIT_KEY="${SPLIT_KEY:-}"
  QUERY="${QUERY:-smartseq2}"
  DATA="${DATA:-data/pancreas.h5ad}"
  require_env
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "NOTE: not inside a Slurm job. On Hyak, heavy jobs on the login node"
    echo "      may be killed. Prefer a compute node, e.g.:"
    echo "          srun -A <account> -p <partition> --time=2:00:00 \\"
    echo "               --mem=32G --cpus-per-task=4 --pty bash"
    echo "          conda activate $ENV_NAME"
    echo "          bash run_on_hyak.sh pilot"
    echo
  fi
  echo ">>> STAGE 4: pilot -- query=$QUERY  data=$DATA  split-key=${SPLIT_KEY:-<cell-level>}"
  if [ -n "$SPLIT_KEY" ]; then
    python pancreas_gonogo.py --query "$QUERY" --data "$DATA" \
        --split-key "$SPLIT_KEY" --out "results_${QUERY}.json"
  else
    python pancreas_gonogo.py --query "$QUERY" --data "$DATA" \
        --out "results_${QUERY}.json"
  fi
  echo
  echo ">>> WHAT TO LOOK AT, in priority order:"
  echo "  1. 'base annotator accuracy' -- want 0.85-0.95."
  echo "     Above 0.98 => the task is too easy here; the comparison is"
  echo "     uninformative. Lower --n-hvg/--n-pcs or pick a harder technology."
  echo "  2. worst-class coverage for each method. naive should be BELOW 0.90."
  echo "     If naive is already fine per-class, there is nothing to fix in"
  echo "     this transfer -- try another --query technology."
  echo "  3. combined_marg vs combined_est vs combined_lab:"
  echo "       combined_marg high  -> the simple unlabeled fix may suffice here"
  echo "       combined_marg low, combined_lab high -> labels are buying you"
  echo "         something; that motivates the label-budget experiment"
  echo "     (This is DIAGNOSTIC evidence, not proof of a shift mechanism.)"
  echo "  4. 'fallback fraction' -- any class at 1.0 got NO correction."
  echo "  5. 'within-cell relative ESS' + calibration counts -- a class with"
  echo "     few cells and low ESS has an unreliable weighted threshold."
  echo
  echo ">>> Then sweep other technologies, e.g.:"
  echo "      QUERY=celseq2 bash run_on_hyak.sh pilot"
  echo "      QUERY=inDrop3 bash run_on_hyak.sh pilot"
  echo "      QUERY=fluidigmc1 bash run_on_hyak.sh pilot"
  ;;

*)
  sed -n '2,16p' "$0"
  ;;
esac
