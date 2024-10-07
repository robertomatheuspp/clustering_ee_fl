cd ./src


NUM_EPOCHS=1000
LOCAL_EPOCHS=5
DATABASE_NAME=cifar10


NUM_USER_TO_PICK=10
DIRICHLET__PARTITION_ALPHA=1




##############
##############
############## Partition 1 and Alpha 1
##############
##############

NUM_USER_PARTITION=100
NUM_DATA_PARTITION=1
OUTPUT_DIR=../rerun_final/cnn/10users_1split_alpha1/


SEED=1


NUM_GROUPS=2
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=5
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=10
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=20
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=25
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=50
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


### random
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type random  --local_ep $LOCAL_EPOCHS  --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### FedCor
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type FedCor  --local_ep $LOCAL_EPOCHS  --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR








##############
##############
############## Alpha 1 and Partition 2
##############
##############

NUM_USER_PARTITION=50
NUM_DATA_PARTITION=2
OUTPUT_DIR=../rerun_final/cnn/10users_2split_alpha1/




NUM_GROUPS=2
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=5
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=10
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=20
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=25
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=50
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


### random
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type random  --local_ep $LOCAL_EPOCHS  --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### FedCor
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type FedCor  --local_ep $LOCAL_EPOCHS  --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR







##############
##############
############## Alpha 1 and Partition 5
##############
##############

NUM_USER_PARTITION=20
NUM_DATA_PARTITION=5
OUTPUT_DIR=../rerun_final/cnn/10users_5split_alpha1/




NUM_GROUPS=2
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=5
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=10
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=20
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=25
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
#
NUM_GROUPS=50
### Repulsive Clust
python experiment.py --seed $SEED --model cnn --dataset_name $DATABASE_NAME --method_type RepulsiveClust --local_ep $LOCAL_EPOCHS --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### Similarity Clust
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type SimilarClust --local_ep $LOCAL_EPOCHS  --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


### random
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type random  --local_ep $LOCAL_EPOCHS  --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
### FedCor
python experiment.py --seed $SEED --model cnn  --dataset_name $DATABASE_NAME --method_type FedCor  --local_ep $LOCAL_EPOCHS  --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR





