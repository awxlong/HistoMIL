#$ -l tmem=4G
#$ -l h_vmem=5G
#$ -l h_rt=5:00:00 
#$ -S /bin/bash
#$ -j y
#$ -N machine_config_download_COAD

source /home/xuelonan/secrier_lab/python3.9.5-biomedai.source

cd ~/secrier_lab/persistence/

python3 HistoMIL/Notebooks/machine_config.py --cohort-name "COAD" --data-dir "/SAN/ugi/WSI_Trans/DATA/" --exp-dir "/home/xuelonan/secrier_lab/persistence/" --api-dir "/home/xuelonan/secrier_lab/persistence/HistoMIL/" 

cp ~/secrier_lab/persistence/HistoMIL/gdc_manifest.2024-06-18.txt /SAN/ugi/WSI_Trans/DATA/COAD/TCGA-COAD/

cd /SAN/ugi/WSI_Trans/DATA/COAD/TCGA-COAD/

/SAN/ugi/WSI_Trans/DATA/BRCA/TCGA-BRCA/gdc-client download -m gdc_manifest.2024-06-18.txt

