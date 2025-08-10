#!/bin/bash

#SBATCH --job-name=generate_vDIT_vids_1.3_64x1_6       # Name of your job
#SBATCH --output=logs/out_cuda_test_%j.out  # Standard output file
#SBATCH --error=logs/err_generatevdit_%j.err  # Standard error file
#SBATCH --nodes=2                          # Request one node. 2 nodes doesn't work because of socket timeout
#SBATCH --ntasks-per-node=1                 # torchrun manages processes (spawns subprocesses)
#SBATCH --gpus-per-node=8
#SBATCH --mem=64gb
#SBATCH --time=08:10:00                     # Wall-clock time limit (HH:MM:SS)
#SBATCH --partition=defq                    # Specify the GPU partition (this may vary on your system)
#SBATCH --exclude=dgx-15,dgx-04             # these don't seem to work


export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export TENSORPIPE_TRANSPORTS=shm,uv

echo $MASTER_ADDR
echo $MASTER_PORT


mkdir -p logs

# CONT_WORKSPACE=/workspace
CONT_WORKSPACE=/home/sdurvasula/minimal
CONT_IMG=/home/sdurvasula/slurmruns/sqsh_images/torch_container.sqsh
CONT_MOUNTS=/home/sdurvasula/mounts:/workspace
CONT_NAME=generate_vid_job

CMD="torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    distributedrun.py"


# CMD="pip install imageio imageio-ffmpeg"
echo "--- SLURM Job Environment ---"
echo "Job ID: ${SLURM_JOBID}"
echo "Nodes allocated: ${SLURM_NNODES}"
echo "GPUs per node requested: ${SLURM_GPUS_PER_NODE}"
echo "CPUs per task requested: ${SLURM_CPUS_PER_TASK}"
echo "Master Address: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "------------------------------"

srun \
    --container-image=${CONT_IMG} \
    --container-workdir=${CONT_WORKSPACE} \
    --container-name=${CONT_NAME} \
    --distribution=block:block \
    ${CMD}

# srun \
#     --container-image=${CONT_IMG} \
#     --container-mounts=${CONT_MOUNTS} \
#     --container-workdir=${CONT_WORKSPACE} \
#     --container-name=${CONT_NAME} \
#     --distribution=block:block \
#     ${CMD}

