import random
from typing import List, Set
from recognition.dataset.Dataset import *

def expand(data: Set[Item]) -> Set[Item]:
    expanded_data = set()
    for item in data:
        expanded_data.add(Item(item.data, item.clazz))
        expanded_data.add(Item(generate_noise_evenly(0.12, item.data, False), item.clazz))
        expanded_data.add(Item(generate_noise_evenly(0.1, item.data, False), item.clazz))
        expanded_data.add(Item(generate_noise_evenly(0.08, item.data, False), item.clazz))
        expanded_data.add(Item(generate_noise_evenly(0.04, item.data, False), item.clazz))
    return expanded_data


def get_outliers(data: Set[Item]) -> Set[Item]:
    outliers = set(random.sample(data, 10))
    for outlier in outliers:
        outlier.data = generate_noise_evenly(0.75, outlier.data, False)
    return outliers


def search_outliers() -> None:
    expanded = expand(blackWhiteNumbers)
    outliers = get_outliers(blackWhiteNumbers)
    dataset = expanded.union(outliers)

    print("Индексы выбросов:")
    for i, item in enumerate(dataset):
        if item in outliers:
            print(i, end=" ")
    print()

    print("Распознанные KNN:")
    for i, item in enumerate(dataset):
        knn = KNNMetricsAnalyzer(3, list(dataset - {item}))
        predicted_class = knn.search(item.data, euclidean)
        if predicted_class != item.clazz:
            print(i, end=" ")
    print()

    print("Распознанные ODIN:")
    inputs = [0] * len(dataset)
    for item in dataset:
        knn = KNNMetricsAnalyzer(5, list(dataset - {item}))
        dataset_list = list(dataset) # преобразуем множество в список
        for neighbor in knn.neighborhood(item.data, euclidean):
            inputs[dataset_list.index(neighbor)] += 1 # используем метод index для списка
    t = 3
    for i, count in enumerate(inputs):
        if count < t:
            print(i, end=" ")


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
            if item.clazz not in counts:
                counts[item.clazz] = 0
            counts[item.clazz] += 1
        return max(counts, key=counts.get)

    def neighborhood(self, input_data: List[int], metric):
        distances = []
        for item in self.dataset:
            distance = metric(item.data, input_data)
            distances.append((item, distance))
        distances.sort(key=lambda x: x[1])
        return [x[0] for x in distances[:self.k]]

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