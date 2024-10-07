
import numpy as np
import copy

import sys
import os
sys.path.append(os.path.abspath("../"))

if "src" not in os.path.abspath("./"):
    from ..sampling import select_idx_users_f
    from ..utils import average_weights
    from ..update import LocalUpdate
else:
    # it is probably running from a notebook
    from sampling import select_idx_users_f
    from utils import average_weights
    from update import LocalUpdate

def fedcor_update(gpr, gpr_param, epoch, global_model,
                    idxs_users, user2dataIdxs, train_dataset,
                    local_bs, local_ep,
                    cur_gt_global_losses, prev_gt_global_losses,
                    optimizer="sgd", lr=0.01,
                    device=None):
    num_users = len(user2dataIdxs)
    # train GPR

    if epoch >= gpr_param["gpr_begin"]:
        if epoch <= gpr_param["warmup"]:  # warm-up
            gpr.Update_Training_Data([np.arange(num_users), ],
                                     [np.array(cur_gt_global_losses) - np.array(prev_gt_global_losses), ],
                                     epoch=epoch)
            if not gpr_param["update_mean"]:
                # ("Training GPR")
                gpr.Train(lr=1e-2, llr=0.01, max_epoches=150, schedule_lr=False,
                          update_mean=gpr_param["update_mean"], verbose=gpr_param["verbose"])
            elif epoch == gpr_param["warmup"]:
                # ("Training GPR")
                gpr.Train(lr=1e-2, llr=0.01, max_epoches=1000, schedule_lr=False,
                          update_mean=gpr_param["update_mean"], verbose=gpr_param["verbose"])

        elif epoch > gpr_param["warmup"] and epoch % gpr_param[
            "GPR_interval"] == 0:  # normal and optimization round
            gpr.Reset_Discount()
            # print("Training with Random Selection For GPR Training:")
            random_idxs_users = select_idx_users_f(num_users, clusteringSol=[],
                                                   type="random", qty_to_pick=gpr_param["num_user_to_pick"])

            gpr_loss = fedcor_trainGP_fl(num_users, global_model=copy.deepcopy(global_model),
                                        idxs_users=random_idxs_users,
                                        train_dataset=train_dataset,
                                        local_bs=local_bs,local_ep=local_ep,
                                        user2dataIdxs=user2dataIdxs,
                                        device=device, optimizer=optimizer, lr=lr, verbose=False)

            gpr.Update_Training_Data([np.arange(num_users), ],
                                     [np.array(gpr_loss) - np.array(cur_gt_global_losses), ], epoch=epoch)

            # print("Training GPR")
            gpr.Train(lr=1e-2, llr=0.01, max_epoches=gpr_param["GPR_Epoch"], schedule_lr=False,
                      update_mean=gpr_param["update_mean"],
                      verbose=gpr_param["verbose"])

        else:  # normal and not optimization round
            gpr.Update_Discount(idxs_users, gpr_param["discount"])

    return gpr


def fedcor_trainGP_fl(num_users, global_model,
                         idxs_users, train_dataset,
                         local_bs,local_ep,
                         user2dataIdxs,
                         device, optimizer="sgd", lr=0.01, verbose=False):

    ## global_model needs to be copy. It should NOT BE RETURNED.

    local_weights = []
    for _ in range(num_users):
        local_weights.append(copy.deepcopy(global_model.state_dict()))

    local_weights = np.array(local_weights)
    global_model.train()

    for idx in idxs_users:
        local_model = copy.deepcopy(global_model)

        local_update = LocalUpdate(device, local_bs=min(local_bs, int(len(user2dataIdxs[idx])//2)),
                                   local_ep=local_ep,
                                   dataset=train_dataset,
                                   idx_client=idx,
                                   idxs=user2dataIdxs[idx],
                                    optimizer = optimizer,
                                    lr = lr,
                                   logger=None, verbose=verbose)

        w, _ = local_update.update_weights(model=local_model,
                                              global_round=-1)
        local_weights[idx] = copy.deepcopy(w)

    # update  weights
    global_weights = average_weights(local_weights[idxs_users])

    global_model.load_state_dict(global_weights)

    # Calculate test accuracy over all users at every epoch
    global_model.eval()
    list_loss = []

    for idx in range(num_users):
        local_model = copy.deepcopy(global_model)

        local_update = LocalUpdate(device, local_bs=min(local_bs, int(len(user2dataIdxs[idx])//2)),
                                  local_ep=2,
                                  dataset=train_dataset,
                                  idx_client=idx,
                                  idxs=user2dataIdxs[idx],
                                  logger=None, verbose=verbose)

        acc, loss = local_update.inference(model=local_model)
        list_loss.append(loss)

    return list_loss