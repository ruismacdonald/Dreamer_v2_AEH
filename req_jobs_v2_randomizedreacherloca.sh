#!/bin/bash
#SBATCH --job-name=d_v2_sdh_rr
#SBATCH --account=def-rsdjjana
#SBATCH --time=6-23:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ng[11105,30708]
#SBATCH --mem=64G
#SBATCH --array=0
#SBATCH --acctg-freq=task=1
#SBATCH --output=/home/ruism/projects/def-rsdjjana/ruism/Dreamer_v2_SDH_fresh/results/randomizedreacherloca_v2_state_dist/%A-%a.out
#SBATCH --error=/home/ruism/projects/def-rsdjjana/ruism/Dreamer_v2_SDH_fresh/results/randomizedreacherloca_v2_state_dist/%A-%a.err

set -e -o pipefail

# Top-level results dir on Lustre
BASE_SAVE_DIR="$HOME/projects/def-rsdjjana/ruism/Dreamer_v2_SDH_fresh/results/randomizedreacherloca_v2_state_dist"
mkdir -p "$BASE_SAVE_DIR"

# Gentle stagger so all tasks don’t hammer Lustre at once
sleep $(( (SLURM_ARRAY_TASK_ID % 10) * 3 ))

# Modules/enviroment
module --force purge
set +u
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
set -u
module load StdEnv/2020
module load cuda/11.4
module load glfw/3.3.2
module load ffmpeg/4.3.2
export IMAGEIO_FFMPEG_EXE="$(command -v ffmpeg)"

# Activate venv
source "$HOME/projects/def-rsdjjana/ruism/loca_env/bin/activate"

# MuJoCo 2.1.0 (legacy)
export MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export MUJOCO_PLUGIN_PATH="$MUJOCO_PATH/bin/mujoco_plugin"
export MUJOCO_GL=glfw
export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$EBROOTGLFW/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

# Threading + wandb
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WANDB_MODE=offline

# Keep BLAS backends from spawning their own pools (most common crash source)
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make OpenMP behavior stable under Slurm binding
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Consistent CPU binding
export SLURM_CPU_BIND=cores

# Source code
DREAMER_SRC="$HOME/projects/def-rsdjjana/ruism/Dreamer_v2_SDH_fresh"
export PYTHONPATH="$DREAMER_SRC:${PYTHONPATH:-}"

export LOCA_DATALOADER_WORKERS=0

: "${SLURM_TMPDIR:=/tmp}"
SEED="${SLURM_ARRAY_TASK_ID}"

RUN_DIR="${SLURM_TMPDIR}/dreamer-v2-sdh-rr-${SLURM_JOB_ID:-0}-${SEED}"
FINAL_DIR="${BASE_SAVE_DIR}/${SEED}"
mkdir -p "$RUN_DIR" "$FINAL_DIR"

cd "$RUN_DIR"

python -u "$DREAMER_SRC/dreamer.py" \
  --env randomizedreacherloca-easy \
  --algo Dreamerv2 \
  --exp-name randomizedreacherloca_v2_state_dist \
  --train \
  --loca-all-phases \
  --buffer-size 2500000 \
  --loca-phase1-steps 1000000 \
  --loca-phase2-steps 1500000 \
  --loca-phase3-steps 0 \
  --loca-state-distance \
  --loca-hash-size 32 \
  --loca-hash-count 2000 \
  --kl-loss-coeff 0.1 \
  --seed "${SEED}"


rsync -a --partial --inplace --no-whole-file "$RUN_DIR/" "$FINAL_DIR/"
