#!/bin/bash
#SBATCH --job-name="fmriprep_job"
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --account=g100-2194
#SBATCH --partition=okeanos
#SBATCH --output="fmriprep.out"
#SBATCH --time=48:00:00
#SBATCH --array=1-N  # Adjust N for the number of subjects!
#SBATCH --mail-user=%u@domain.tld
#SBATCH --mail-type=ALL

# Define Paths
STUDY="/home/perpetua"
export PATH=/opt/singularity/3.5.3/bin:$PATH

BIDS_DIR="${STUDY}/bids"
DERIVS_DIR="${BIDS_DIR}/derivatives/fmriprep-23.2.3"
LOCAL_FREESURFER_DIR="${STUDY}/freesurfer"
SINGULARITY_IMG="${STUDY}/images/fmriprep_23.2.3.sif"

# Use a Guaranteed Writable Work Directory
WORK_DIR="/home/perpetua/fmriprep_work"
mkdir -p "${WORK_DIR}"
chmod -R 777 "${WORK_DIR}"  # Ensure full write permissions

export SINGULARITYENV_TMPDIR="/work"
export SINGULARITYENV_WORK_DIR="/work"

# Move to a Safe Working Directory
cd "${WORK_DIR}" || exit 1

#  Optimized Singularity Command with Explicit Mounts
SINGULARITY_CMD="singularity exec --cleanenv \
    --bind ${BIDS_DIR}:/data \
    --bind ${WORK_DIR}:/work \
    --bind ${LOCAL_FREESURFER_DIR}:/fsdir \
    ${SINGULARITY_IMG}"

# Verify Work Directory Inside Singularity
echo "Checking if WORK_DIR is writable inside Singularity..."
${SINGULARITY_CMD} /bin/bash -c "touch /work/testfile" && echo "WORK_DIR is writable inside Singularity" || { echo "ERROR: WORK_DIR is not writable inside Singularity"; exit 1; }

# Extract Subject List Once
subjects=($(awk -F'\t' 'NR>1 {print substr($1,5)}' "${BIDS_DIR}/participants.tsv"))

# Compute Subject Index
task_index=$((SLURM_ARRAY_TASK_ID - 1))
subject=${subjects[$task_index]}

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Extracted subject: sub-${subject}"

if [[ -z "$subject" ]]; then
    echo "ERROR: Subject ID extraction failed."
    exit 1
fi

# Run fMRIPrep 
cmd="${SINGULARITY_CMD} fmriprep /data /data/derivatives/fmriprep-23.2.3 participant \
    --participant-label ${subject} \
    -w /work \
    -vv \
    --omp-nthreads 8 \
    --nthreads 24 \
    --mem_mb 96000 \
    --output-spaces MNI152NLin2009cAsym T1w fsnative fsaverage \ 
    --use-aroma \
    --fs-subjects-dir /fsdir/subjects \
    --clean-workdir"

echo "Running task ${SLURM_ARRAY_TASK_ID}"
echo "Commandline: ${cmd}"
srun --cpus-per-task=24 --mem=96G ${cmd}
exitcode=$?

# Log Results
echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${exitcode}" >> "${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"

exit ${exitcode}
