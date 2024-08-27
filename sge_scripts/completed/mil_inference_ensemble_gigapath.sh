#$ -l tmem=16G
#$ -l h_rt=2:30:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true
#$ -N mil_inference_ensemble_gigapath
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-transmil-prov-gigapath' \
                                            --project-name 'g0-arrest-inference-prov-gigapath-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'prov-gigapath' \
                                            --mil-algorithm "TransMIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_transmil_gigapath_32epochs_cv5cv=0_epoch=07-auroc_val=0.78' 'mil_transmil_gigapath_32epochs_cv5cv=1_epoch=09-auroc_val=0.77' 'mil_transmil_gigapath_32epochs_cv5cv=2_epoch=10-auroc_val=0.76' 'mil_transmil_gigapath_32epochs_cv5cv=3_epoch=08-auroc_val=0.73' 'mil_transmil_gigapath_32epochs_cv5cv=4_epoch=12-auroc_val=0.71'

python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-attentionmil-prov-gigapath' \
                                            --project-name 'g0-arrest-inference-prov-gigapath-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'prov-gigapath' \
                                            --mil-algorithm "AttentionMIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_attentionMIL_gigapath_32epochs_cv5_reruncv=0_epoch=25-auroc_val=0.72' 'mil_attentionMIL_gigapath_32epochs_cv5_reruncv=1_epoch=30-auroc_val=0.69' 'mil_attentionMIL_gigapath_32epochs_cv5_reruncv=2_epoch=17-auroc_val=0.70' 'mil_attentionMIL_gigapath_32epochs_cv5_reruncv=3_epoch=31-auroc_val=0.64' 'mil_attentionMIL_gigapath_32epochs_cv5_reruncv=4_epoch=02-auroc_val=0.69'

python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-transformer-prov-gigapath' \
                                            --project-name 'g0-arrest-inference-prov-gigapath-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'prov-gigapath' \
                                            --mil-algorithm "Transformer" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_transformer_gigapath_8epochs_smaller_cv5cv=0_epoch=06-auroc_val=0.76' 'mil_transformer_gigapath_8epochs_smaller_cv5cv=1_epoch=07-auroc_val=0.78' 'mil_transformer_gigapath_8epochs_smaller_cv5cv=2_epoch=02-auroc_val=0.75' 'mil_transformer_gigapath_8epochs_smaller_cv5cv=3_epoch=05-auroc_val=0.72' 'mil_transformer_gigapath_8epochs_smaller_cv5cv=4_epoch=05-auroc_val=0.70'


python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-camil-prov-gigapath' \
                                            --project-name 'g0-arrest-inference-prov-gigapath-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'prov-gigapath' \
                                            --mil-algorithm "CAMIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --num-workers 0 \
                                            --ckpt-filenames 'mil_camil_gigapath_30epochs_cv5_reducelronplateucv=0_epoch=14-auroc_val=0.76' 'mil_camil_gigapath_30epochs_cv5_reducelronplateucv=1_epoch=16-auroc_val=0.77' 'mil_camil_gigapath_30epochs_cv5_reducelronplateucv=2_epoch=03-auroc_val=0.74' 'mil_camil_gigapath_30epochs_cv5_reducelronplateucv=3_epoch=09-auroc_val=0.79' 'mil_camil_gigapath_30epochs_cv5_reducelronplateucv=4_epoch=04-auroc_val=0.69'

python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-dtfdmil-prov-gigapath' \
                                            --project-name 'g0-arrest-inference-prov-gigapath-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'prov-gigapath' \
                                            --mil-algorithm "DTFD_MIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_dtfd_gigapath_42epochs_cv5_multisteplrcv=0_epoch=26-auroc_val=0.80' 'mil_dtfd_gigapath_42epochs_cv5_multisteplrcv=1_epoch=37-auroc_val=0.80' 'mil_dtfd_gigapath_42epochs_cv5_multisteplrcv=2_epoch=33-auroc_val=0.78' 'mil_dtfd_gigapath_42epochs_cv5_multisteplrcv=3_epoch=39-auroc_val=0.73' 'mil_dtfd_gigapath_42epochs_cv5_multisteplrcv=4_epoch=37-auroc_val=0.71'

python3 HistoMIL/Notebooks/mil_inference.py --exp-name 'mil-inference-graphtransformer-prov-gigapath' \
                                            --project-name 'g0-arrest-inference-prov-gigapath-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'prov-gigapath' \
                                            --mil-algorithm "GraphTransformer" \
                                            --label-dict "{0:0,1:1}" \
                                            --num-workers 0 \
                                            --ckpt-filenames 'mil_graphtransformer_gigapath_42epochs_cv5_multisteplrcv=0_epoch=05-auroc_val=0.60' 'mil_graphtransformer_gigapath_42epochs_cv5_multisteplrcv=1_epoch=00-auroc_val=0.58' 'mil_graphtransformer_gigapath_42epochs_cv5_multisteplrcv=2_epoch=08-auroc_val=0.59' 'mil_graphtransformer_gigapath_42epochs_cv5_multisteplrcv=3_epoch=13-auroc_val=0.72' 'mil_graphtransformer_gigapath_42epochs_cv5_multisteplrcv=4_epoch=09-auroc_val=0.61'


echo "Finished at: $(date)"