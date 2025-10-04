#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main		### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name my_job			### name of the job
#SBATCH --output job-%J.out			### output log for running job - %J for job number
#SBATCH --gpus=rtx_4090:1				### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1

# Note: the following 4 lines are commented out
##SBATCH --mail-user=liranitz@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --mem=60G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####

DATASET_PATH="/home/liranitz/LiranDeepFakeProject/FakeDetectionWithFlorance/GenImageDataset"
FORGED_DATASET_PATH="/sise/mickyfi-group/liranazr/data/forged_images/AutoSplice/split/test" # If needed, use forged_data flag in the script
FEATURE_OUT_ROOT="/home/liranitz/big_data_proj_dir/FinalProject/out/feature_map_analysis/xception"

EMBEDDINGS_DIR="/home/liranitz/big_data_proj_dir/FinalProject/out/embeddings_from_xception"
OUT_DIR="/home/liranitz/big_data_proj_dir/FinalProject/out/xception_outputs"
FEATURE_OUT_ROOT="/home/liranitz/big_data_proj_dir/FinalProject/out/feature_map_analysis/xception"
BATCH_SIZE=64

module load anaconda
source activate wt_conv_env

python /home/liranitz/big_data_proj_dir/FinalProject/src/compare_embeddings_vs_features.py \
  --embeddings_dir $EMBEDDINGS_DIR \
  --feature_root $FEATURE_OUT_ROOT \
  --feature_model xception \
  --feature_layer stage1 \
  --only_groups biggan sdv5 \
  --sample_per_group 300 \
  --tsne_iter 3000 \
  --tsne_perp 30 \
  --out_dir "/home/liranitz/big_data_proj_dir/FinalProject/out/xception_outputs/embeddings_vs_features_biggan_sdv5"


  # --only_groups REAL FAKE adm biggan glide sdv5 vqdm wukong \
