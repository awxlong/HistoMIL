#$ -l tmem=4G
#$ -l h_vmem=5G
#$ -l h_rt=18:00:00 
#$ -S /bin/bash
#$ -j y
#$ -N machine_config_download_COAD

source /home/xuelonan/secrier_lab/python3.9.5-biomedai.source

cd /SAN/ugi/WSI_Trans/DATA/COAD/TCGA-COAD/

/SAN/ugi/WSI_Trans/DATA/BRCA/TCGA-BRCA/gdc-client download -m gdc_manifest.2024-06-18.txt