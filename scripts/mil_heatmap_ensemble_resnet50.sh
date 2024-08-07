#$ -l tmem=16G
#$ -l h_rt=3:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true,gpu_type=!(titanxp|titanx)
#$ -N mil_heatmap_ensemble_resnet50
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-transmil-resnet50' \
                                            --project-name 'g0-arrest-inference-resnet50-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'resnet50' \
                                            --mil-algorithm "TransMIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_transmil_resnet50_32epochs_reruncv=0_epoch=10-auroc_val=0.75' 'mil_transmil_resnet50_32epochs_reruncv=1_epoch=02-auroc_val=0.77' 'mil_transmil_resnet50_32epochs_reruncv=2_epoch=03-auroc_val=0.80' 'mil_transmil_resnet50_32epochs_reruncv=3_epoch=06-auroc_val=0.75' 'mil_transmil_resnet50_32epochs_reruncv=4_epoch=09-auroc_val=0.73'\
                                            --ensemble

python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-attentionmil-resnet50' \
                                            --project-name 'g0-arrest-inference-resnet50-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'resnet50' \
                                            --mil-algorithm "AttentionMIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'attentionMIL_resnet50_32epoch_reruncv=0_epoch=30-auroc_val=0.70' 'attentionMIL_resnet50_32epoch_reruncv=1_epoch=28-auroc_val=0.68' 'attentionMIL_resnet50_32epoch_reruncv=2_epoch=27-auroc_val=0.72' 'attentionMIL_resnet50_32epoch_reruncv=3_epoch=26-auroc_val=0.64' 'attentionMIL_resnet50_32epoch_reruncv=4_epoch=18-auroc_val=0.64'\
                                            --ensemble

python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-transformer-resnet50' \
                                            --project-name 'g0-arrest-inference-resnet50-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'resnet50' \
                                            --mil-algorithm "Transformer" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_transformer_resnet50_8epochs_full_small_0711_0939cv=0_epoch=05-auroc=0.00' 'mil_transformer_resnet50_8epochs_full_small_0711_0939cv=1_epoch=04-auroc=0.00' 'mil_transformer_resnet50_8epochs_full_small_0711_0939cv=2_epoch=04-auroc=0.00' 'mil_transformer_resnet50_8epochs_full_small_0711_0939cv=3_epoch=07-auroc=0.00' 'mil_transformer_resnet50_8epochs_full_small_0711_0939cv=4_epoch=04-auroc=0.00'\
                                            --ensemble


python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-camil-resnet50' \
                                            --project-name 'g0-arrest-inference-resnet50-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'resnet50' \
                                            --mil-algorithm "CAMIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --num-workers 0 \
                                            --ckpt-filenames 'mil_camil_resnet50_30epochs_cv5_reducelronplateucv=0_epoch=03-auroc_val=0.74' 'mil_camil_resnet50_30epochs_cv5_reducelronplateucv=1_epoch=11-auroc_val=0.76' 'mil_camil_resnet50_30epochs_cv5_reducelronplateucv=2_epoch=04-auroc_val=0.81' 'mil_camil_resnet50_30epochs_cv5_reducelronplateucv=3_epoch=07-auroc_val=0.73' 'mil_camil_resnet50_30epochs_cv5_reducelronplateucv=4_epoch=09-auroc_val=0.73'\
                                            --ensemble



python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-dtfdmil-resnet50' \
                                            --project-name 'g0-arrest-inference-resnet50-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'resnet50' \
                                            --mil-algorithm "DTFD_MIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_dtfd_resnet50_42epochs_cv5_multisteplrcv=0_epoch=41-auroc_val=0.72' 'mil_dtfd_resnet50_42epochs_cv5_multisteplrcv=1_epoch=41-auroc_val=0.75' 'mil_dtfd_resnet50_42epochs_cv5_multisteplrcv=2_epoch=40-auroc_val=0.79' 'mil_dtfd_resnet50_42epochs_cv5_multisteplrcv=3_epoch=41-auroc_val=0.68' 'mil_dtfd_resnet50_42epochs_cv5_multisteplrcv=4_epoch=41-auroc_val=0.69'\
                                            --ensemble

python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-graphtransformer-resnet50' \
                                            --project-name 'g0-arrest-inference-resnet50-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'resnet50' \
                                            --mil-algorithm "GraphTransformer" \
                                            --label-dict "{0:0,1:1}" \
                                            --num-workers 0 \
                                            --ckpt-filenames 'mil_graphtransformer_resnet50_42epochs_cv5_multisteplrcv=0_epoch=01-auroc_val=0.41' 'mil_graphtransformer_resnet50_42epochs_cv5_multisteplrcv=1_epoch=00-auroc_val=0.68' 'mil_graphtransformer_resnet50_42epochs_cv5_multisteplrcv=2_epoch=04-auroc_val=0.56' 'mil_graphtransformer_resnet50_42epochs_cv5_multisteplrcv=3_epoch=00-auroc_val=0.51' 'mil_graphtransformer_resnet50_42epochs_cv5_multisteplrcv=4_epoch=04-auroc_val=0.57'\
                                            --ensemble

python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-clam-resnet50' \
                                            --project-name 'g0-arrest-inference-resnet50-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'resnet50' \
                                            --mil-algorithm "CLAM" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_clam_resnet50_42epochs_cv5cv=0_epoch=20-auroc_val=0.75' 'mil_clam_resnet50_42epochs_cv5cv=1_epoch=39-auroc_val=0.76' 'mil_clam_resnet50_42epochs_cv5cv=2_epoch=12-auroc_val=0.80' 'mil_clam_resnet50_42epochs_cv5cv=3_epoch=10-auroc_val=0.74' 'mil_clam_resnet50_42epochs_cv5cv=4_epoch=05-auroc_val=0.73'\
                                            --ensemble
                                            
echo "Finished at: $(date)"