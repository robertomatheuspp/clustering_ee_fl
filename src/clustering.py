
import numpy as np
import repulsive_clustering
from scipy.cluster import hierarchy
# from scipy.cluster.hierarchy import  ward, average, fcluster
from nltk.cluster.kmeans import KMeansClusterer


def clusteringGroundTruthSol_similar_together(num_clients_per_part, num_partitions):
    """
        Return the clustering solution of the most similar together.
        The kth entry of 'clusteringSol' contains the users associated to kth cluster.

        For example, if we had 3 partitions, the solution would be:
            clusteringSol = [[], [], []]

            clusteringSol[0] = list(range(10))
            clusteringSol[1] = list(range(10, 20))
            clusteringSol[2] = list(range(20, 30))

    :param num_clients_per_part: number of clients per group
    :param num_partitions: total number of groups

    :return: The kth entry of 'clusteringSol' contains the users associated to kth cluster.
    """
    # Clustering solution following "clust. together most similar ones"
    clusteringSol = [[]] * (num_partitions)
    for p in range(num_partitions):
        clusteringSol[p] = (np.arange(num_clients_per_part) + p*num_clients_per_part).tolist()

    return clusteringSol


def clusteringEmpirical_similar_together(num_groups, num_users, map_user_class_qt, linkage_method=hierarchy.ward, dist_f=None):
    """
    Replicating what Fernanda had, the difference here is that p_j for the jth user is normalized wrt to the samples in the entire dataset.
    As we have a centralized controller, that is possible. Note that by normalizing wrt. the entire dataset, we then have a probability that
    is comparable among the different devices.

    Here we use hierarchy, but we could have used any other method, e.g. K-medoid.

    :param num_groups:
    :param num_users:
    :param map_user_class_qt:
    :param linkage_method: Ward, but could also use mean
    :param dist_f: Default is the symetrized KL. This is because we are considering the empirical probability.
    :return:
    """
    assert (num_users >= num_groups)
    dist_f = lambda p1, p2: (
                0.5 * p1 * np.log(p1 / p2) + 0.5 * p2 * np.log(p2 / p1)).sum()

    Pmat = map_user_class_qt / map_user_class_qt.sum(0)
    Pmat[Pmat == 0] = 1e-6

    kclusterer = KMeansClusterer(num_groups, distance=dist_f, repeats=5)
    pred = kclusterer.cluster(Pmat, assign_clusters=True)
    pred = np.array(pred)

    clusteringSol = [np.where(pred == group)[0].tolist() for group in range(num_groups)]
    # print(pred)
    # print("eher", clusteringSol)
    # distMat = repulsive_clustering.build_distMat(Pmat, num_users, dist_f=dist_f)

    # pred = hierarchy.fcluster(linkage_method(distMat), num_groups, criterion="maxclust").astype(int)
    # clusteringSol = [np.where(pred - 1 == group)[0].tolist() for group in range(num_groups)]


    return clusteringSol

def clustering_repulsiveGroundTruthSol(num_clients_per_part, num_partitions):
    """
        Return the clustering solution of the most similar together.
        The kth entry of 'clusteringSol' contains the users associated to kth cluster.

        For example, if we had 10 partitions, the solution would be:
            clusteringSol = [[], [], []]

            clusteringSol[0] = [0, 10, 20]
            clusteringSol[1] = [1, 11, 21]
            clusteringSol[2] = [2, 12, 22]
            ...
            clusteringSol[G] = [G, 1G, 2G]

    :param num_clients_per_part: number of clients per group
    :param num_partitions: total number of groups

    :return: The kth entry of 'clusteringSol' contains the users associated to kth cluster.
    """

    total = num_clients_per_part*num_partitions

    aux = np.arange(total) % num_partitions

    # 1 2 3 4 1 2 3 4 1 2 3 4, 1 2 3 4
    clusteringSol = [[]] * (num_partitions)
    for p in range(num_partitions):
        clusteringSol[p] = np.where(aux == p)[0].tolist()

    return clusteringSol



def clustering_repulsiveEmpiricalSol(num_groups, num_users, map_user_class_qt):
    """

    :param num_groups:
    :param num_users:
    :param map_user_class_qt: output of
    :return:
    """

    assert(num_users >= num_groups)
    Pmat = map_user_class_qt / map_user_class_qt.sum(0)

    distMat = repulsive_clustering.build_distMat(Pmat, num_users)
    clusteringSol = repulsive_clustering.repulsive_clustering_suboptmial(num_groups, Pmat, distMat, max_visting_groups=5)

    return clusteringSol