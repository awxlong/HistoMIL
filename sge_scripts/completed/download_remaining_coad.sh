#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=8:00:00 
#$ -S /bin/bash
#$ -j y
#$ -N download_remaining_COAD

source /home/xuelonan/secrier_lab/python3.9.5-biomedai.source

echo "Running on host: $(hostname)"
echo "Starting at: $(date)"


cd /SAN/ugi/WSI_Trans/DATA/COAD/TCGA-COAD/

/SAN/ugi/WSI_Trans/DATA/BRCA/TCGA-BRCA/gdc-client download -m gdc_manifest_remaining.txt

echo "Finished at: $(date)"