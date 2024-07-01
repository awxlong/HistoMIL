#$ -l tmem=16G
#$ -l h_rt=24:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true
#$ -N mil_pretrain_transformer_uni
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-10.2.source
python3 HistoMIL/Notebooks/mil_run.py --exp-name 'mil-pretrained-MSI' \
                                      --project-name 'g0-arrest' --wandb-entity-name 'cell-x' --localcohort-name 'COAD' --task-name 'g0_arrest' --pid-name 'PatientID' \
                                      --targets-name 'g0_arrest' \
                                      --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                      --split-ratio 0.8 0.2 --step-size 224 --precomputed 'uni' \
                                      --label-dict "{'negative':0,'positive':1}" \
                                      --mil-algorithm "Transformer" \
                                      --pretrained-weights-dir '/home/xuelonan/secrier_lab/persistence/HistoMIL/MODEL/Image/MIL/Transformer/pretrained_weights/' \
                                      --pretrained-weights-name 'MSI_high_CRC_model.pth'
echo "Finished at: $(date)"