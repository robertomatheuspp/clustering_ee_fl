
import numpy as np
import random


def build_distMat(Pmat, num_users, dist_f=None):
    """

    :param Pmat: a matrix (num clients, num classes) normalized by Pmat / Pmat.sum(axis=1)
    :param num_users: number of user/clients that are considered in the FL process
    :param dist_f: distance function that receives two vectors and outputs the element-wise distance
    :return:
    """
    if dist_f == None:
        dist_f = lambda p1, p2: 0.5 * p1 * np.log(p1 / p2) + 0.5 * p2 * np.log(p2 / p1)
    # This is important otherwise we keep getting NaN and it will have very bad impact in the maximum diversity clustering.
    Pmat[Pmat == 0] = 1e-6

    ### symmetrized KL distance
    D = np.zeros((num_users, num_users))

    for i in range(num_users):
        for j in range(i + 1, num_users):
            p1 = Pmat[i]
            p2 = Pmat[j]

            dist = dist_f(p1, p2)

            D[i, j] = dist.sum()
            D[j, i] = D[i, j]
    return D

# This can be done in a much more efficient way, when flipping the user, I don´t change all the other cluster, just the 2 that I flipped with.
# I can also smartly choose what element to change (the one that is most similar to some other element in the cluster)
# Similar, flip to the most dissimilar group
def performance(solution, Pmat, distMat, dist_f=None):
    if dist_f == None:
        dist_f = lambda p1, p2: 0.5 * p1 * np.log(p1 / p2) + 0.5 * p2 * np.log(p2 / p1)
    num_groups = len(solution)
    num_classes = Pmat.shape[1]

    pg_list = np.zeros((num_groups, num_classes))

    # the goal of maximization
    obj1 = 0
    for g in range(num_groups):
        curD = distMat[solution[g], :]
        curD = curD[:, solution[g]]
        obj1 += curD.mean()

        pg_list[g] = Pmat[solution[g], :].sum(0)

    pg_list[pg_list == 0] = 1e-6
    obj1 /= num_groups

    # the goal of minimization
    obj2 = 0
    for g in range(num_groups):
        for gprime in range(g + 1, num_groups):
            cur_dist = dist_f(pg_list[g], pg_list[gprime])
            obj2 += np.sum(cur_dist[np.logical_not(np.isnan(cur_dist))])

    obj2 /= num_groups * (num_groups - 1)

    return obj1, obj2, pg_list


def repulsive_clustering_heuristic(num_groups, Pmat, distMat, maxIter=np.inf):
    """

    Here we literally re-implement Salman code


    :param num_groups: how many clusters/partitions we want to have in the end
    :param Pmat: a matrix (num clients, num classes) normalized by Pmat / Pmat.sum(axis=1)
    :param distMat: a matrix of size (num clients, num clients)
    :param maxIter:
    :return:
    """
    num_users = Pmat.shape[0]

    numElPerClust = np.ceil(num_users / num_groups).astype(int)

    clusteringSol = np.arange(num_users)
    random.shuffle(clusteringSol)

    # initial random solution
    clusteringSol = [clusteringSol[g * numElPerClust: (g + 1) * numElPerClust] for g in range(num_groups)]

    prev_performance = -np.inf
    flag_improved = True

    while flag_improved and maxIter > 0:
        maxIter -= 1
        flag_improved = False
        for g in range(num_groups):
            for gprime in range(g + 1, num_groups):
                for idx1, elem1 in enumerate(clusteringSol[g]):
                    for idx2, elem2 in enumerate(clusteringSol[gprime]):
                        newClustSol = np.copy(clusteringSol)
                        newClustSol[g][idx1] = clusteringSol[gprime][idx2]
                        newClustSol[gprime][idx2] = clusteringSol[g][idx1]


                        obj1, obj2, _ = performance(newClustSol, Pmat, distMat)
                        cur_performance = obj1 - obj2

                        if cur_performance > prev_performance:
                            prev_performance = cur_performance
                            flag_improved = True
                            clusteringSol = newClustSol

    return clusteringSol


def repulsive_clustering_suboptmial(num_groups, Pmat, distMat, max_visting_groups=-1):
    """
        Here instead of only replicating salmans code, we optmize it a bit by sorting the classes based
        on their smallest intra cluster distance. Then we only perform the swapping on the top-K (defined via 'max_visting_groups')
        clusters.
        Also, the swapping is done only over the elements that are involved in this smallest distance.

        :param num_groups: how many clusters/partitions we want to have in the end
        :param Pmat: a matrix (num clients, num classes) normalized by Pmat / Pmat.sum(axis=1)
        :param distMat: a matrix of size (num clients, num clients)
        :param maxIter:
        :return:
    """
    num_users = Pmat.shape[0]
    numElPerClust = np.ceil(num_users / num_groups).astype(int)

    if max_visting_groups <= 0:
        max_visting_groups = numElPerClust

    clusteringSol = np.arange(num_users)
    random.shuffle(clusteringSol)

    # initial random solution
    clusteringSol = [clusteringSol[g * numElPerClust: (g + 1) * numElPerClust] for g in range(num_groups)]

    obj1, obj2, pg_list = performance(clusteringSol, Pmat, distMat)

    prev_performance = obj1 - obj2
    flag_improved = True

    while flag_improved:
        flag_improved = False
        aux_list = np.zeros((num_groups, 3))
        # 1: element 1
        # 2: element 2
        # 3: distance between 1 and 2
        for g in range(num_groups):
            curD = distMat[clusteringSol[g], :]
            curD = curD[:, clusteringSol[g]]

            ## build an eye with inf
            aux = np.eye(curD.shape[0]) * np.inf
            aux[np.isnan(aux)] = 0

            curD = curD + aux
            aux_list[g, 0], aux_list[g, 1] = np.unravel_index(curD.argmin(), curD.shape)
            aux_list[g, 2] = curD[aux_list[g, 0].astype(int)][aux_list[g, 1].astype(int)]

        # argsort is in ascending order, i.e., the first one has the smallest distance. We want exactly that
        most_similar_g_sorted = aux_list[:, 2].argsort()

        for gidx, g in enumerate(most_similar_g_sorted[:max_visting_groups]):
            for gprime in most_similar_g_sorted[gidx:max_visting_groups]:
                for idx1, elem1 in enumerate(clusteringSol[g]):
                    for idx2, elem2 in enumerate(clusteringSol[gprime]):
                        newClustSol = np.copy(clusteringSol)
                        newClustSol[g][idx1] = clusteringSol[gprime][idx2]
                        newClustSol[gprime][idx2] = clusteringSol[g][idx1]

                        # checking the performance of one pair is sufficient rather than doing (1,2) x (3,4)
                        obj1, obj2, pg_list = performance(newClustSol, Pmat, distMat)
                        cur_performance = obj1 - obj2
                        if cur_performance > prev_performance:
                            prev_performance = cur_performance
                            flag_improved = True
                            clusteringSol = newClustSol
    return clusteringSol
