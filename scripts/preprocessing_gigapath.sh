#$ -l tmem=18G
#$ -l h_rt=72:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true
#$ -N preprocess_with_gigapath
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-10.2.source
python3 HistoMIL/Notebooks/pre_processing.py --exp-name 'preprocessing_gigapath' \
                                             --project-name 'g0-arrest' \
                                             --wandb-entity-name 'cell-x' \
                                             --localcohort-name 'COAD' \
                                             --task-name 'g0_arrest' \
                                             --pid-name 'PatientID' \
                                             --targets-name 'g0_arrest' \
                                             --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                             --split-ratio 0.99 0.01 \
                                             --step-size 224 \
                                             --backbone-name 'prov-gigapath' \
                                             --label-dict "{'negative':0,'positive':1}" \
                                             --api-dir "/home/xuelonan/secrier_lab/persistence/HistoMIL/" 
echo "Finished at: $(date)"