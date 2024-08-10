#$ -l tmem=8G
#$ -l h_rt=1:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true,gpu_type=!(titanxp|titanx)
#$ -N mil_inference_ensemble_uni_multimodal_continuous
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-transmil-multimodal-continuous-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "TransMILMultimodal" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_transmil_uni_multimodal_32epochs_cv5_continuous_onlycv=0_epoch=09-auroc_val=0.75' 'mil_transmil_uni_multimodal_32epochs_cv5_continuous_onlycv=1_epoch=13-auroc_val=0.79' 'mil_transmil_uni_multimodal_32epochs_cv5_continuous_onlycv=2_epoch=09-auroc_val=0.79' 'mil_transmil_uni_multimodal_32epochs_cv5_continuous_onlycv=3_epoch=10-auroc_val=0.74' 'mil_transmil_uni_multimodal_32epochs_cv5_continuous_onlycv=4_epoch=19-auroc_val=0.73'


echo "Finished at: $(date)"