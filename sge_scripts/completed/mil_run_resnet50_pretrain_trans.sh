#$ -l tmem=32G
#$ -l h_rt=6:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true,gpu_type=A40
#$ -N mil_pre_trans_resnet50_8epochs
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_run.py --exp-name 'mil_pretrained_MSI_resnet50_8epochs' \
                                      --project-name 'g0-arrest-resnet50' --wandb-entity-name 'cell-x' --localcohort-name 'COAD' --task-name 'g0_arrest' --pid-name 'PatientID' \
                                      --targets-name 'g0_arrest' \
                                      --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                      --split-ratio 0.8 0.2 --step-size 224 --precomputed 'resnet50' \
                                      --label-dict "{0:0,1:1}" \
                                      --mil-algorithm "Transformer" \
                                      --pretrained-weights-dir '/home/xuelonan/secrier_lab/persistence/HistoMIL/MODEL/Image/MIL/Transformer/pretrained_weights/' \
                                      --pretrained-weights-name 'MSI_high_CRC_model.pth'\
                                      --n-epochs 8 \
                                      --monitor-metric 'auroc_val'
echo "Finished at: $(date)"