from collections import Counter
from typing import Dict

import numpy as np


def dirichlet(
    targets: np.ndarray,
    label_set: set,
    client_num: int,
    alpha: float,
    least_samples: int,
    partition: dict,
    stats: dict,
):
    """Partition dataset according to Dirichlet with concentration parameter `alpha`.

    Args:
        targets (np.ndarray): Data label array.
        label_set (set): Label set.
        client_num (int): Number of clients.
        alpha (float): Concentration parameter. Smaller alpha indicates strong data heterogeneity.
        least_samples (int): Lease number of data samples each client should have.
        partition (Dict): Output data indices dict.
        stats (Dict): Output dict that recording clients data distribution.

    """
    min_size = 0
    indices_4_labels = {i: np.where(targets == i)[0] for i in label_set}
    while min_size < least_samples:
        # Initialize empty lists to hold data indices for each client
        partition["data_indices"] = [[] for _ in range(client_num)]

        # Iterate through each label in the label_set
        for label in label_set:
            # Shuffle the indices corresponding to the current label
            np.random.shuffle(indices_4_labels[label])

            # Generate a Dirichlet distribution for partitioning data among clients
            distribution = np.random.dirichlet(np.repeat(alpha, client_num))

            # Calculate the cumulative distribution to get split indices
            cumulative_distribution = np.cumsum(distribution) * len(
                indices_4_labels[label]
            )
            split_indices_position = cumulative_distribution.astype(int)[:-1]

            # Split the indices based on the calculated positions
            split_indices = np.split(indices_4_labels[label], split_indices_position)

            # Assign the split indices to each client
            for client_id in range(client_num):
                partition["data_indices"][client_id].extend(split_indices[client_id])

        # Update the minimum number of samples across all clients
        min_size = min(len(indices) for indices in partition["data_indices"])

    for i in range(client_num):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets[partition["data_indices"][i]])
        stats[i]["y"] = dict(Counter(targets[partition["data_indices"][i]].tolist()))

    sample_num = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": sample_num.mean().item(),
        "stddev": sample_num.std().item(),
    }
