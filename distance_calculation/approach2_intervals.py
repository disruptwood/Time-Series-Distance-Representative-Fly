import numpy as np

import numpy as np


def convert_intervals_to_dict(interval_list):
    """
    Converts a list of interval dictionaries into a behavior-structured dictionary.

    Args:
        interval_list (list of dict): A list of {"tStart": ..., "tEnd": ...} dicts.

    Returns:
        dict: { behavior_id: [(start, end), ...] } structured for `compute_bout_features`
    """
    behavior_dict = {}
    for i, interval in enumerate(interval_list):  # Assuming each entry represents a different behavior
        behavior_dict[i] = [(interval["tStart"], interval["tEnd"])]

    return behavior_dict


def compute_bout_features(interval_list):
    """
    Computes a feature vector for each fly based on its intervals.

    Args:
        interval_list (list of dict): Intervals for a fly, structured as:
                                      [ {"tStart": X, "tEnd": Y}, {"tStart": A, "tEnd": B}, ... ]

    Returns:
        dict: A feature vector for each behavior, keyed by behavior_id.
              Each feature vector contains [numBouts, avgLen, fractionActive, avgGap].
    """
    interval_dict = convert_intervals_to_dict(interval_list)  # Convert before processing
    feature_vectors = {}

    for behavior_id, intervals in interval_dict.items():
        if not intervals:
            feature_vectors[behavior_id] = [0, 0.0, 0.0, 0.0]
            continue

        # Sort intervals by start time
        intervals = sorted(intervals, key=lambda x: x[0])

        # Compute total number of bouts
        num_bouts = len(intervals)

        # Compute average bout length
        bout_lengths = [end - start for start, end in intervals]
        avg_len = np.mean(bout_lengths) if bout_lengths else 0.0

        # Compute total fraction of time active
        total_active_time = sum(bout_lengths)

        # Compute inter-bout gap average
        inter_bout_gaps = [
            intervals[i][0] - intervals[i - 1][1] for i in range(1, num_bouts)
        ]
        avg_gap = np.mean(inter_bout_gaps) if inter_bout_gaps else 0.0

        # Store the feature vector
        feature_vectors[behavior_id] = [num_bouts, avg_len, total_active_time, avg_gap]

    return feature_vectors


def approach2_interval_distance(interval_dict_A, interval_dict_B):
    """
    Computes the distance between two flies based on feature vectors derived from intervals.
    Features include number of bouts, average bout length, fraction active, and average gap length.
    Uses Euclidean distance between the feature vectors.

    Args:
        interval_dict_A (dict): Intervals for fly A, structured as:
                                { behavior_id: [(start1, end1), (start2, end2), ...], ... }
        interval_dict_B (dict): Intervals for fly B, structured similarly.

    Returns:
        float: The Euclidean distance between the feature vectors of fly A and fly B.
    """
    # Compute feature vectors for both flies
    features_A = compute_bout_features(interval_dict_A)
    features_B = compute_bout_features(interval_dict_B)

    # Combine all behavior IDs
    all_behaviors = set(features_A.keys()) | set(features_B.keys())

    # Initialize the distance
    total_distance = 0.0

    for behavior in all_behaviors:
        # Get the feature vectors for this behavior
        vec_A = features_A.get(behavior, [0, 0.0, 0.0, 0.0])
        vec_B = features_B.get(behavior, [0, 0.0, 0.0, 0.0])

        # Compute the Euclidean distance for this behavior
        behavior_distance = np.sqrt(np.sum((np.array(vec_A) - np.array(vec_B)) ** 2))
        total_distance += behavior_distance

    return total_distance
