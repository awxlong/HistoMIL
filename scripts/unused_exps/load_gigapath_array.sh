#$ -l tmem=17G
#$ -l h_rt=1:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true
#$ -N load_gigapath_arrayjob_testing
#$ -t 1-3

cd secrier_lab/persistence/HistoMIL
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-10.2.source

# Your Python files
PYTHON_FILES=(
    "load_gigapath1.py"
    "load_gigapath2.py"
    "load_gigapath3.py"
)

# Get the Python file for this task
PYTHON_FILE=${PYTHON_FILES[$SGE_TASK_ID-1]}

# Print some information
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
echo "Current working directory: $(pwd)"
echo "SGE_TASK_ID: $SGE_TASK_ID"
echo "Python file: $PYTHON_FILE"

# Run the Python script
python3 $PYTHON_FILE

echo "Finished at: $(date)"
