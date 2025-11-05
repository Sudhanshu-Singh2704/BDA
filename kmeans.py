import random
import math
from collections import defaultdict


def squared_distance(x, y):
    """Squared distance for 1D points."""
    return (x - y) ** 2


def kmeans_map(data, means):
    mapped = []
    for x in data:
        # Find nearest mean
        min_dist = float("inf")
        closest_mean = None
        for i, mu in enumerate(means):
            dist = squared_distance(x, mu)
            if dist < min_dist:
                min_dist = dist
                closest_mean = i
        # Emit key-value (cluster_id, (x, 1))
        mapped.append((closest_mean, (x, 1)))
    return mapped


def kmeans_combine(mapped_data):
    combined = defaultdict(lambda: (0, 0))  # (sum, count)
    for cluster_id, (x, count) in mapped_data:
        total_sum, total_count = combined[cluster_id]
        combined[cluster_id] = (total_sum + x, total_count + count)
    return combined


def kmeans_reduce(combined_data):
    new_means = {}
    for cluster_id, (total_sum, total_count) in combined_data.items():
        if total_count > 0:
            new_means[cluster_id] = total_sum / total_count
    return new_means


def kmeans_mapreduce_1d(data, k=2, max_iters=10, tolerance=1e-4):
    # Step 1: Initialization
    means = random.sample(data, k)

    for iteration in range(max_iters):
        # Step 2: Map
        mapped = kmeans_map(data, means)

        # Step 3: Combine
        combined = kmeans_combine(mapped)

        # Step 4: Reduce
        new_means_dict = kmeans_reduce(combined)
        new_means = [new_means_dict[i] for i in sorted(new_means_dict.keys())]

        # Step 5: Convergence Check
        diff = sum(abs(a - b) for a, b in zip(means, new_means))
        print(f"Iteration {iteration+1}: Mean shift = {diff:.4f}")
        if diff < tolerance:
            break
        means = new_means

    return means



if __name__ == "__main__":
    # Input 1D dataset
    data = list(map(float, input("Enter 1D dataset values (e.g. 1 2 3 10 11 12): ").split()))
    k = int(input("Enter number of clusters (k): "))

    final_means = kmeans_mapreduce_1d(data, k=k, max_iters=20)

    print("\nFinal Cluster Means:")
    for i, mu in enumerate(final_means):
        print(f"Cluster {i+1}: {mu:.4f}")
