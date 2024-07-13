#$ -l tmem=36G
#$ -l h_vmem=36G
#$ -l h_rt=8:00:00 
#$ -S /bin/bash
#$ -j y
#$ -N adj_matrix_uni_array
#$ -t 1-7
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/preprocess_adj_matrix.py --exp-name 'adj_matrix_uni' \
                                             --project-name 'g0-arrest' \
                                             --wandb-entity-name 'cell-x' \
                                             --localcohort-name 'COAD' \
                                             --task-name 'g0_arrest' \
                                             --pid-name 'PatientID' \
                                             --targets-name 'g0_arrest' \
                                             --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                             --split-ratio 0.99 0.01 \
                                             --step-size 224 \
                                             --backbone-name 'uni' \
                                             --label-dict "{0:0,1:1}" \
                                             --api-dir "/home/xuelonan/secrier_lab/persistence/HistoMIL/" \
                                             --array-job-idx ${SGE_TASK_ID} 
echo "Finished at: $(date)"