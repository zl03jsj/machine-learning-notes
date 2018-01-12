import csv
import random
import math


def load_dataset(filename, split, training_set=[], test_set=[]):
    with open(filename, 'rt') as csv_file:
        lines = csv.reader(csv_file)
        print(type(lines))
        data_set = list(lines)
        for x in range(len(data_set) - 1):
            for y in range(4):
                data_set[x][y] = float(data_set[x][y])
            if random.random() < split:
                training_set.append(data_set[x])
            else:
                test_set.append(data_set[x])


def euclidean_diistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = euclidean_diistance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=lambda item: item[len(item) - 1])
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    # sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    sorted_votes = sorted(class_votes.items(), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    # prepare data
    training_set = []
    test_set = []
    split = 0.67
    load_dataset(r'iris.data', split, training_set, test_set)
    print('Train set: ' + repr(len(training_set)))
    print('Test set: ' + repr(len(test_set)))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(test_set[x][-1]))
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()

# KNN sklearn
# from sklearn import neighbors
# from sklearn import datasets
#
# knn = neighbors.KNeighborsClassifier()
# iris = datasets.load_iris()
# print(iris)
# knn.fit(iris.data, iris.target)
# predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
# print(predictedLabel)
