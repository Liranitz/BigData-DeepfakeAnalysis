#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main		### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name my_job			### name of the job
#SBATCH --output job-%J.out			### output log for running job - %J for job number
#SBATCH --gpus=1				### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1

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
MODEL="vit"
EMBEDDINGS_DIR="/home/liranitz/big_data_proj_dir/FinalProject/out/embeddings_from_${MODEL}"
OUT_DIR="/home/liranitz/big_data_proj_dir/FinalProject/out/${MODEL}_outputs"
BATCH_SIZE=64

module load anaconda
source activate wt_conv_env

## Running the t-SNE analysis script ###

# Separate plots per model:
python /home/liranitz/big_data_proj_dir/FinalProject/src/analyze_embedding_with_tsne.py \
  --embeddings_dir $EMBEDDINGS_DIR \
  --out_dir "$OUT_DIR/tsne_outputs" \
  --mode separate \
  --sample_per_model 500 \
  --perplexity 30 \
  --n_iter 5000

# One combined plot with colored classes and a legend sidebar:
python /home/liranitz/big_data_proj_dir/FinalProject/src/analyze_embedding_with_tsne.py \
  --embeddings_dir $EMBEDDINGS_DIR \
  --out_dir "$OUT_DIR/tsne_combined_outputs" \
  --mode combined \
  --sample_per_model 500 \
  --perplexity 30 \
  --n_iter 5000

# Pairwise plots between models - all pairs:
python /home/liranitz/big_data_proj_dir/FinalProject/src/analyze_embedding_with_tsne.py \
  --embeddings_dir $EMBEDDINGS_DIR \
  --out_dir "$OUT_DIR/tsne_pairs_outputs" \
  --mode pairs \
  --pairing_strategy random \
  --num_pairs 10 \
  --sample_per_model 500 \
  --perplexity 30 \
  --n_iter 5000

### Running the PCA Analysis script ###
Separate plots per model:
python /home/liranitz/big_data_proj_dir/FinalProject/src/analyze_embedding_with_pca.py \
  --embeddings_dir $EMBEDDINGS_DIR \
  --out_dir "$OUT_DIR/pca_outputs" \
  --mode separate \
  --sample_per_model 500 \
  --standardize

# # One combined plot with colored classes and a legend sidebar:
python /home/liranitz/big_data_proj_dir/FinalProject/src/analyze_embedding_with_pca.py \
  --embeddings_dir $EMBEDDINGS_DIR \
  --out_dir "$OUT_DIR/pca_combined_outputs" \
  --mode combined \
  --sample_per_model 500 \
  --standardize

