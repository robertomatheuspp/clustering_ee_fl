cd ./src



NUM_EPOCHS=1000
NUM_USER_TO_PICK=10
DATABASE_NAME=fmnist
DIRICHLET__PARTITION_ALPHA=1





SEED=0



##############
##############
############## Alpha 1, partition 1, seed 1
##############
##############


NUM_USER_PARTITION=100
NUM_DATA_PARTITION=1
OUTPUT_DIR=../fmnist/10users_1split_alpha1/




NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR



##############
##############
############## Alpha 1, partition 2, seed 1
##############
##############


NUM_USER_PARTITION=50
NUM_DATA_PARTITION=2
OUTPUT_DIR=../fmnist/10users_2split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


##############
##############
############## Alpha 1, partition 5, seed 1
##############
##############


NUM_USER_PARTITION=20
NUM_DATA_PARTITION=5
OUTPUT_DIR=../fmnist/10users_5split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


























SEED=1


##############
##############
############## Alpha 1, partition 1, seed 1
##############
##############


NUM_USER_PARTITION=100
NUM_DATA_PARTITION=1
OUTPUT_DIR=../fmnist/10users_1split_alpha1/




NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR



##############
##############
############## Alpha 1, partition 2, seed 1
##############
##############


NUM_USER_PARTITION=50
NUM_DATA_PARTITION=2
OUTPUT_DIR=../fmnist/10users_2split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


##############
##############
############## Alpha 1, partition 5, seed 1
##############
##############


NUM_USER_PARTITION=20
NUM_DATA_PARTITION=5
OUTPUT_DIR=../fmnist/10users_5split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
































SEED=2


##############
##############
############## Alpha 1, partition 1, seed 2
##############
##############


NUM_USER_PARTITION=100
NUM_DATA_PARTITION=1
OUTPUT_DIR=../fmnist/10users_1split_alpha1/




NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR



##############
##############
############## Alpha 1, partition 2, seed 2
##############
##############


NUM_USER_PARTITION=50
NUM_DATA_PARTITION=2
OUTPUT_DIR=../fmnist/10users_2split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


##############
##############
############## Alpha 1, partition 5, seed 2
##############
##############


NUM_USER_PARTITION=20
NUM_DATA_PARTITION=5
OUTPUT_DIR=../fmnist/10users_5split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
































SEED=3


##############
##############
############## Alpha 1, partition 1, seed 3
##############
##############


NUM_USER_PARTITION=100
NUM_DATA_PARTITION=1
OUTPUT_DIR=../fmnist/10users_1split_alpha1/




NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR



##############
##############
############## Alpha 1, partition 2, seed 3
##############
##############


NUM_USER_PARTITION=50
NUM_DATA_PARTITION=2
OUTPUT_DIR=../fmnist/10users_2split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


##############
##############
############## Alpha 1, partition 5, seed 3
##############
##############


NUM_USER_PARTITION=20
NUM_DATA_PARTITION=5
OUTPUT_DIR=../fmnist/10users_5split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR



































SEED=4


##############
##############
############## Alpha 1, partition 1, seed 4
##############
##############


NUM_USER_PARTITION=100
NUM_DATA_PARTITION=1
OUTPUT_DIR=../fmnist/10users_1split_alpha1/




NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR



##############
##############
############## Alpha 1, partition 2, seed 4
##############
##############


NUM_USER_PARTITION=50
NUM_DATA_PARTITION=2
OUTPUT_DIR=../fmnist/10users_2split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


##############
##############
############## Alpha 1, partition 5, seed 4
##############
##############


NUM_USER_PARTITION=20
NUM_DATA_PARTITION=5
OUTPUT_DIR=../fmnist/10users_5split_alpha1/



NUM_GROUPS=2
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=5
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=10
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=20
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=25
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
NUM_GROUPS=50
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type SimilarClust   --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp  --dataset_name $DATABASE_NAME --method_type RepulsiveClust --seed $SEED --num_groups_users $NUM_GROUPS --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR


python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type random --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
python experiment.py --model mlp --local_ep 5 --dataset_name $DATABASE_NAME --method_type FedCor --seed $SEED --dirichlet_alpha $DIRICHLET__PARTITION_ALPHA --num_dataset_partition $NUM_DATA_PARTITION --num_user_per_partition $NUM_USER_PARTITION --num_user_to_pick $NUM_USER_TO_PICK  --epochs $NUM_EPOCHS --base_dir_outputs $OUTPUT_DIR
