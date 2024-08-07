#$ -l tmem=9G
#$ -l h_rt=2:30:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true
#$ -N mil_inference_ensemble_clam_remaining
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source

python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-clam-resnet50' \
                                            --project-name 'g0-arrest-inference-resnet50-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'resnet50' \
                                            --mil-algorithm "CLAM" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_clam_resnet50_42epochs_cv5cv=0_epoch=20-auroc_val=0.75' 'mil_clam_resnet50_42epochs_cv5cv=1_epoch=39-auroc_val=0.76' 'mil_clam_resnet50_42epochs_cv5cv=2_epoch=12-auroc_val=0.80' 'mil_clam_resnet50_42epochs_cv5cv=3_epoch=10-auroc_val=0.74' 'mil_clam_resnet50_42epochs_cv5cv=4_epoch=05-auroc_val=0.73'
                                            
python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-clam-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "CLAM" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_clam_uni_42epochs_cv5cv=0_epoch=12-auroc_val=0.77' 'mil_clam_uni_42epochs_cv5cv=1_epoch=14-auroc_val=0.83' 'mil_clam_uni_42epochs_cv5cv=2_epoch=05-auroc_val=0.80' 'mil_clam_uni_42epochs_cv5cv=3_epoch=21-auroc_val=0.76' 'mil_clam_uni_42epochs_cv5cv=4_epoch=24-auroc_val=0.73'

python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-clam-prov-gigapath' \
                                            --project-name 'g0-arrest-inference-prov-gigapath-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'prov-gigapath' \
                                            --mil-algorithm "CLAM" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_clam_gigapath_42epochs_cv5cv=0_epoch=00-auroc_val=0.77' 'mil_clam_gigapath_42epochs_cv5cv=1_epoch=04-auroc_val=0.80' 'mil_clam_gigapath_42epochs_cv5cv=2_epoch=02-auroc_val=0.79' 'mil_clam_gigapath_42epochs_cv5cv=3_epoch=27-auroc_val=0.77' 'mil_clam_gigapath_42epochs_cv5cv=4_epoch=08-auroc_val=0.71'




echo "Finished at: $(date)"