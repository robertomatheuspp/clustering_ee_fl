import copy
import os.path
import random

from datetime import datetime
import torch
from codecarbon import EmissionsTracker
from tensorboardX import SummaryWriter

from args_parser import args_parser
from clustering import *
from fedcor import fedcor_update, GPR
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, MLPFashion_Mnist
from sampling import non_iid_dirichlet_sampling
from sampling import select_idx_users_f
from update import LocalUpdate, test_inference
from utils import average_weights, count_user2class, get_dataset_split


################ init parameters

args = args_parser()

if args["num_user_to_pick"] == -1:
    args["num_user_to_pick"] = args["num_dataset_partition"]


# torch.manual_seed(args["seed"])
# np.random.seed(args["seed"])
# random.seed(args["seed"])
# # torch.use_deterministic_algorithms(True)



np.random.seed(args["seed"])
torch.manual_seed(args["seed"])
random.seed(args["seed"])
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


num_clients_per_part = args["num_user_per_partition"]
# num_dataset_partition = args["num_dataset_partition"]



epochs = args["epochs"]
lr = args["lr"]

local_bs = args["local_bs"]
local_ep = args["local_ep"]


use_cuda = args["gpu"]
use_cuda = use_cuda and torch.cuda.is_available()

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

num_users = num_clients_per_part * args["num_dataset_partition"]



tracker = EmissionsTracker()

emission = {}
emission["unit"] = "kWh"
emission["pre_processing"] = 0
emission["clustering"] = np.zeros((epochs, ))
emission["learning"] = np.zeros((epochs, num_users))
emission["communication"] = np.zeros((epochs, num_users))





########################## LOADING AND SPLITTING DATA


# here 'split2dataIdx' contains the data split according to their label.
# Each label is in one entry of 'split2dataIdx'
train_dataset, test_dataset, split2dataIdx = get_dataset_split(dataset=args["dataset_name"],
                                                               base_dir=args["base_dir_dataset"])

# Decide how many groups and which group has which label
# We will consider two clusters of data, one with classes [0,4] and another with [5,9]
if args["num_dataset_partition"] == 1:
    map_groups2class_split = [
        np.squeeze(np.hstack([split2dataIdx[idx] for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])),
    ]
elif args["num_dataset_partition"] == 2:
    map_groups2class_split = [
        np.squeeze(np.hstack([split2dataIdx[idx] for idx in [0, 1, 2, 3, 4]])),
        np.squeeze(np.hstack([split2dataIdx[idx] for idx in [5, 6, 7, 8, 9]])),
    ]
elif args["num_dataset_partition"] == 5:
    map_groups2class_split = [
        np.squeeze(np.hstack([split2dataIdx[idx] for idx in [0, 1]])),
        np.squeeze(np.hstack([split2dataIdx[idx] for idx in [2, 3]])),
        np.squeeze(np.hstack([split2dataIdx[idx] for idx in [4, 5]])),
        np.squeeze(np.hstack([split2dataIdx[idx] for idx in [6, 7]])),
        np.squeeze(np.hstack([split2dataIdx[idx] for idx in [8, 9]])),
    ]
else:
    raise NotImplementedError("Error in partitioning dataset. Number of partitions {} not implemented. ".format(args["num_dataset_partition"]))

# If this raises an error, then there is something wrong
assert(args["num_dataset_partition"] == len(map_groups2class_split))


########### non-iid data distribution

labels = np.array(train_dataset.targets)
user2dataIdxs = []
for cur_partition_idx, cur in enumerate(map_groups2class_split):
    original_idx = np.squeeze(map_groups2class_split[cur_partition_idx])
    cur_out = non_iid_dirichlet_sampling(labels[original_idx],
                                         n_parties=num_clients_per_part,
                                         alpha=args["dirichlet_alpha"])
    for elem_client in cur_out.values():
        user2dataIdxs.append(original_idx[elem_client])


################# TRAINING PARAMETERS
user2Class_count = count_user2class(user2dataIdxs, labels, num_classes=10)
user_count = user2Class_count.sum(1)

if args["method_type"] != "random":
    tracker.start_task("clustering")

clusteringSol = np.array([])
gpr_param = {"active": False}
if args["method_type"] == "SimilarClust":
    if args["num_groups_users"] < 0:
        clusteringSol = clusteringGroundTruthSol_similar_together(num_clients_per_part=num_clients_per_part,
                                                                  num_partitions=len(map_groups2class_split))
    else:
        clusteringSol = clusteringEmpirical_similar_together(num_groups=args["num_groups_users"],
                                                             num_users=num_users,
                                                             map_user_class_qt=user2Class_count)
        # clusterigSol = clusteringGroundTruthSol_similar_together(num_clients_per_part=num_clients_per_part,
        #                                           num_partitions=len(map_groups2class_split))

elif args["method_type"] == "RepulsiveClust":
    if args["num_groups_users"] < 0:
        clusteringSol = clustering_repulsiveGroundTruthSol(num_clients_per_part=num_clients_per_part,
                                                                  num_partitions=len(map_groups2class_split))
    else:
        clusteringSol = clustering_repulsiveEmpiricalSol(num_groups=args["num_groups_users"],
                                                         num_users=num_users,
                                                         map_user_class_qt=user2Class_count)
elif args["method_type"] == "CovClust":
    raise NotImplementedError("Method {} has not been implemented yet.".format(args["method_type"]))
elif args["method_type"] == "FedCor":
    type_selection = "GPR"
    gpr_param = {
        "active": True,
        "epsilon_greedy": 0,
        "dimension": 15,
        "update_mean": True,
        "poly_norm": 0,
        "gpr_begin": 0,
        "GPR_interval": 10,
        "group_size": 100,
        "GPR_gamma": 0.95,
        "warmup": 15,
        "discount": 0.95,
        "num_rand_users": args["num_user_to_pick"],  # len(map_groups2class_split),
        "num_user_to_pick": args["num_user_to_pick"],
        "verbose": False,
        "train_method": "MML",
        "GPR_Epoch": 100
    }
elif args["method_type"] == "Power-d":
    raise NotImplementedError("Method {} has not been implemented yet.".format(args["method_type"]))
elif args["method_type"] == "centralized":
    print("WORKING IN CENTRALIZED SHOULD HAVE A SINGLE USER!")
    assert(num_users == 1)
elif args["method_type"] != "random":
    raise NotImplementedError("Method {} is not in the list of accepted methods.".format(args["method_type"]))

if args["method_type"] != "random":
    emission["clustering"] += tracker.stop_task("clustering").energy_consumed


# Training settings
# type_selection = 'classical'
# type_selection = 'rand_clients'
# type_selection = 'diff_together'
# type_selection = "similar_together"

assert(num_users == len(user2dataIdxs))


################################## TRAINING


logger = SummaryWriter(args["base_dir_outputs"] + '/logs/{}'.format(args["method_type"]))


global_model = None

if args["dataset_name"] == "mnist":
    global_model = CNNMnist(1)
elif args["dataset_name"] == "fmnist":
    if args["model"] == "cnn":
        global_model = CNNFashion_Mnist()
    elif args["model"] == "mlp":
        global_model = MLPFashion_Mnist(dim_in=np.prod(train_dataset.data[0].shape), nb_class=10)
elif args["dataset_name"] == "cifar10":
    global_model = CNNCifar()



# Set the model to train and send it to device.
global_model.to(device)
global_model.train()

if args["verbose"]:
    print(global_model)

# copy weights
global_weights = global_model.state_dict()

# Training
train_loss = np.zeros((epochs, num_users))
train_accuracy = np.zeros((epochs, num_users))
test_accuracy = np.zeros((epochs, 1))
test_loss = np.zeros((epochs, 1))


val_acc_list, net_list = [], []
print_every = 2 if args["verbose"] else np.inf



gpr = None
weights = []
prev_list_loss = []

if gpr_param["active"]:
    tracker.start_task("pre_processing")

    gpr = GPR.Kernel_GPR(num_users, loss_type=gpr_param["train_method"], reusable_history_length=gpr_param["group_size"],
                         gamma=gpr_param["GPR_gamma"], device=device,
                         dimension=gpr_param["dimension"],
                         kernel=GPR.Poly_Kernel,
                         order=1,
                         Normalize=gpr_param["poly_norm"])

    weights = np.zeros((num_users, 1))
    for i in range(num_users):
        weights[i] = (len(user2dataIdxs[i]) / len(train_dataset))

    gpr.to(device)

    ##### THIS IS ONLY FOR  FedCor (also counts as pre processing step)
    for cur_user_idx in range(num_users):
        local_model = LocalUpdate(device, local_bs=min(local_bs, int(user_count[cur_user_idx])),
                                  dataset=train_dataset,
                                  idxs=user2dataIdxs[cur_user_idx],
                                  optimizer=args["optimizer"],
                                  lr=args["lr"],
                                  logger=None)

        # if gpr_param["active"]:
        _, loss = local_model.inference(model=global_model)

        prev_list_loss.append(np.copy(loss))

    emission["pre_processing"] += tracker.stop_task("pre_processing").energy_consumed


print(clusteringSol)

for epoch in range(epochs):
    local_weights, local_losses = [], []
    if args["verbose"]:
        print(f'\n SEED {args["seed"]} of {args["method_type"]}-{args["num_groups_users"]} | Global Training Round : {epoch + 1} |\n')

    global_model.train()

    #### TRACKING CLUSTERING
    tracker.start_task("clustering")

    # (CLUSTERING) SELECTION OF USERS
    if gpr_param["active"]:
        if epoch > gpr_param["warmup"]:
            # FedCor
            idxs_users = gpr.Select_Clients(args["num_user_to_pick"],
                                            gpr_param["epsilon_greedy"],
                                            weights=weights,
                                            Dynamic=False,
                                            Dynamic_TH=0.0)
            if args["verbose"]:
                print("GPR Chosen Clients:", idxs_users)
        else:
            idxs_users = select_idx_users_f(num_users, clusteringSol=[], type="random",
                                            qty_to_pick=args["num_user_to_pick"])
    else:
        idxs_users = select_idx_users_f(num_users, clusteringSol,
                                        type=args["method_type"],
                                        qty_to_pick=args["num_user_to_pick"])

    emission["clustering"][epoch] = tracker.stop_task("clustering").energy_consumed


    for cur_user_idx in idxs_users:
        tracker.start_task("train_{}_{}".format(epoch, cur_user_idx))
        if args["verbose"]:
            print("USER ", cur_user_idx)

        # print("THIS")
        # print ((local_bs, user_count[cur_user_idx]))
        local_model = LocalUpdate(device, local_bs=min(local_bs, int(user_count[cur_user_idx])),
                                  local_ep=local_ep,
                                  dataset=train_dataset,
                                  idx_client=cur_user_idx,
                                  idxs=user2dataIdxs[cur_user_idx], logger=logger,
                                  optimizer=args["optimizer"],
                                  lr=args["lr"],
                                  verbose=args["verbose"])
        w, loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                             global_round=epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))

        cur_emission = tracker.stop_task("train_{}_{}".format(epoch, cur_user_idx))

        emission["learning"][epoch, cur_user_idx] = cur_emission.energy_consumed
        if args["verbose"]:
            print("USER {} consumed {}".format(cur_user_idx, cur_emission))

    # update global weights
    global_weights = average_weights(local_weights)

    # update global weights
    global_model.load_state_dict(global_weights)

    train_loss[epoch][idxs_users] = np.copy(local_losses)

    # Calculate AVG TRAIN accuracy over all users at every epoch
    cur_list_acc, cur_list_loss = [], []
    global_model.eval()
    if gpr_param["active"]:
        tracker.start_task("clustering")

    for cur_user_idx in range(num_users):
        local_model = LocalUpdate(device, local_bs=min(4*local_bs, int(user_count[cur_user_idx])), # this is for test only, so we can put large
                                  dataset=train_dataset,
                                  idxs=user2dataIdxs[cur_user_idx], logger=logger,
                                  optimizer=args["optimizer"],
                                  lr=args["lr"],
                                  verbose=args["verbose"])

        acc, loss = local_model.inference(model=global_model)

        cur_list_acc.append(np.copy(acc))
        cur_list_loss.append(np.copy(loss))

    if gpr_param["active"]:
        gpr = fedcor_update.fedcor_update(gpr, gpr_param, epoch, global_model,
                                          idxs_users, user2dataIdxs, train_dataset,
                                          local_bs, local_ep, cur_list_loss, prev_list_loss,
                                          optimizer=args["optimizer"], lr=args["lr"],
                                          device=device)
        emission["clustering"][epoch] += tracker.stop_task("clustering").energy_consumed

    prev_list_loss = np.copy(cur_list_loss)

    train_accuracy[epoch] = np.copy(copy.deepcopy(cur_list_acc))
    cur_test_acc, cur_test_loss = test_inference(device, global_model, test_dataset)
    test_accuracy[epoch] = np.copy(cur_test_acc)
    test_loss[epoch] = np.copy(cur_test_loss)

    # print global training loss after every 'i' rounds
    if (epoch + 1) % print_every == 0 and args["verbose"]:
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss[epoch].mean()))}')
        print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[epoch].mean()))
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_accuracy[epoch].mean()))




##################### SAVING RESULTS

output_dir = args["base_dir_outputs"] + "/carbon_" + args["method_type"]  + '/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

torch.save(global_model.state_dict(), output_dir + "model_seed_{}".format(args["seed"]))


cur_time = datetime.now()
filename_out = "emission_data_{dataset}_dbPart_{num_dataset_partition}_model_{model}_method_{method_type}_numGroupU_{num_groups_users}_totalU_{total_users}_pick_{num_user_to_pick}_alpha_{dirichlet_alpha}_seed_{seed}".format(
        dataset=args["dataset_name"], model=args["model"],
        num_dataset_partition=args["num_dataset_partition"],
        dirichlet_alpha=args["dirichlet_alpha"], seed=args["seed"],
        method_type = args["method_type"],
        num_groups_users= args["num_groups_users"] if args["num_groups_users"] != -1 else "gtGroups",
        total_users=num_users,
        num_user_to_pick=args["num_user_to_pick"]
)


data_out = {"args": args,
            "emission": emission,
            "train_loss": train_loss,
            "train_acc": train_accuracy,
            "test_loss": test_loss,
            "test_acc": test_accuracy,
            "user2dataIdxs": user2dataIdxs,
            "user2Class_count": user2Class_count,
            "clusteringSol": clusteringSol,
            "date":cur_time
            }

np.save(output_dir + "/" + filename_out, data_out)
