import random
from typing import List

class Item:
    def __init__(self, data: List[int], number: int):
        self.data = data
        self.number = number

    def __repr__(self):
        return f"{self.data} -> {self.number}"
    
class KNNMetricsAnalyzer:
    def __init__(self, k: int, dataset: List[Item]):
        self.k = k
        self.dataset = dataset

    def search(self, input_data: List[int], metric):
        distances = []
        for item in self.dataset:
            distance = metric(item.data, input_data)
            distances.append((item, distance))
        distances.sort(key=lambda x: x[1])
        nearest = [x[0] for x in distances[:self.k]]
        counts = {}
        for item in nearest:
            if item.number not in counts:
                counts[item.number] = 0
            counts[item.number] += 1
        return max(counts, key=counts.get)


def manhattan(v1, v2):
    return sum(abs(x - y) for x, y in zip(v1, v2))


def euclidean(v1, v2):
    return pow(sum(pow(x - y, 2) for x, y in zip(v1, v2)), 0.5)


def euclidean_pow(v1, v2):
    return sum(pow(x - y, 2) for x, y in zip(v1, v2))


def chebyshyov(v1, v2):
    return max(abs(x - y) for x, y in zip(v1, v2))


metrics_name_map = {
    manhattan: "manhattan",
    euclidean: "euclidean",
    euclidean_pow: "euclidean pow",
    chebyshyov: "chebyshyov"
}


def generate_noise_evenly(v, data, shadows):
    return [
        (random.uniform(0, 1) * 256 if shadows else (1 if x == 0 else 0)
         ) if random.uniform(0, 1) < v else x
        for x in data
    ]


def generate_uneven_noise(data, shadows, center, middle, edges):
    noise = []
    for i, x in enumerate(data):
        probability = center if i in [27, 28, 35, 36] else \
            (middle if i in range(18, 22) or i in range(
                42, 46) or i in [26, 34, 29, 37] else edges)

        if random.uniform(0, 1) < probability:
            if shadows:
                noise.append(random.uniform(0, 1) * 256)
            else:
                noise.append(0 if x == 1 else 1)
        else:
            noise.append(x)
    return noise


def check_metrics(iteration, noise_evenly, shadows, k, all_noise=0.4, center=0.4, middle=0.3, edges=0.2,
                  etalon_dataset=None, dataset=None):
    knn = KNNMetricsAnalyzer(k, dataset)
    metrics = {metric_func: {standardNumbers.number: {} for standardNumbers in etalon_dataset} for metric_func in
               [manhattan, euclidean, euclidean_pow, chebyshyov]}

    for standardNumbers in etalon_dataset:
        for _ in range(iteration):
            data_noisy = generate_noise_evenly(all_noise, standardNumbers.data, shadows) if noise_evenly else \
                generate_uneven_noise(
                    standardNumbers.data, shadows, center, middle, edges)

            for metric_func in metrics:
                select_number = knn.search(data_noisy, metric_func)
                counts = metrics[metric_func][standardNumbers.number]
                counts[select_number] = counts.get(select_number, 0) + 1

    for metric_func in metrics:
        print(metrics_name_map[metric_func])
        for standardNumbers in etalon_dataset:
            counts = metrics[metric_func][standardNumbers.number]
            sorted_counts = sorted(
                counts.items(), key=lambda x: x[1], reverse=True)
            percentages = [
                f"{number}={round((count / iteration) * 100, 2)}%" for number, count in sorted_counts]
            print(f"{standardNumbers.number}: {', '.join(percentages)}")
        print()
