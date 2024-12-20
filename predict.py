import gzip
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import process

K_MAX = 1000

def calculate_dist(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

def gen_k_nearest_labels(joint_test, joint_train, path, k=K_MAX):
    if os.path.exists(path):
        return

    nearest_labels_lst = []
    # nearest_neighbors_lst[i] = an array of nearest neighbors of i-th image in test

    for test_features, test_label in joint_test:
        distances = []
        for train_features, train_label in joint_train:
            dist = calculate_dist(test_features, train_features)
            distances.append((dist, train_label))
        distances.sort(key=lambda x: x[0]) # Sort by distance
        k_nearest_labels = [train_label for ignore, train_label in distances[:k]] # Save k nearest labels
        nearest_labels_lst.append(k_nearest_labels)

    with open(path, 'wb') as f:
        pickle.dump(nearest_labels_lst, f) # Save to file, don't have to calculate again

def load_binary(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def predict_on_test_data(nearest_labels, k):
    # Use for preprocessed data (distance was calculated)
    k_nearest = nearest_labels[:k]
    return np.bincount(k_nearest).argmax()

def graph_test_accuracy(joint_test, nearest_labels_lst, k_range):
    true_labels = np.array([label for ignore, label in joint_test])
    accuracies_all = []
    for method in range(0, 3):
        accuracies = []
        for k in k_range:
            predictions = []
            for i in range(len(nearest_labels_lst[method])):
                predict = predict_on_test_data(nearest_labels_lst[method][i], k)
                predictions.append(predict)
            predictions = np.array(predictions)
            correct_predictions = np.sum(predictions == true_labels)
            accuracy = correct_predictions / len(joint_test)
            accuracies.append(accuracy)
        accuracies_all.append([method, accuracies])


    for method, accuracies in accuracies_all:
        if method == 0:
            plt.plot(k_range, accuracies, label="Vectorize", color="green")
        elif method == 1:
            plt.plot(k_range, accuracies, label="Sampling", color="blue")
        elif method == 2:
            plt.plot(k_range, accuracies, label="Histogram", color="red")

    plt.xlim(min(k_range)-1, max(k_range)+1)
    plt.ylim(0, 1)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("K (Number of neighbors)")
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

def table_accuracy_with_methods(joint_test, nearest_labels_lst, k_range):
    true_labels = np.array([label for ignore, label in joint_test])
    res = []
    for k in k_range:
        for method in range(0, 3):
            predictions = []
            for i in range(len(nearest_labels_lst[method])):
                predict = predict_on_test_data(nearest_labels_lst[method][i], k)
                predictions.append(predict)
            predictions = np.array(predictions)
            correct_predictions = np.sum(predictions == true_labels)
            accuracy = correct_predictions / len(joint_test)
            res.append([k, method, accuracy])

    with open('accuracy_table.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["k", "flat", "chunk", "histogram"])  # Tiêu đề cột
        for k in k_range:
            flat_accuracy = next((x[2] for x in res if x[0] == k and x[1] == 0), None)
            chunk_accuracy = next((x[2] for x in res if x[0] == k and x[1] == 1), None)
            histogram_accuracy = next((x[2] for x in res if x[0] == k and x[1] == 2), None)
            writer.writerow([k, flat_accuracy, chunk_accuracy, histogram_accuracy])

def predict_label(feature, k, joint_compare):
    # Use for input data (distance was not calculated)

    distances = []
    for comparing_feature, comparing_label in joint_compare:
        dist = calculate_dist(feature, comparing_feature)
        distances.append([dist, comparing_label])
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for ignore, label in distances[:k]]
    predict = np.bincount(np.array(k_nearest_labels)).argmax()
    return predict

def predict_with_methods(image, extract_methods, k_values = [50, 50, 50], *methods_data):
    #Return predictions with different extract methods

    results = []
    image = image.reshape((1, 28, 28))
    image_features = process.extract_features(image)
    for i in range(len(extract_methods)):
        k = k_values[i]
        # Append [Method name, predict]
        results.append([extract_methods[i], predict_label(image_features[i][0], k, methods_data[i])])

    return results

def find_optimize_k(joint_test, nearest_labels_lst, k_range):
    true_labels = np.array([label for ignore, label in joint_test])
    accuracies = []
    for k in k_range:
        predictions = []
        for i in range(len(nearest_labels_lst)):
            predict = predict_on_test_data(nearest_labels_lst[i], k)
            predictions.append(predict)
        predictions = np.array(predictions)
        correct_predictions = np.sum(predictions == true_labels)
        accuracy = correct_predictions / len(joint_test)
        accuracies.append(accuracy)
    return np.array(accuracies).argmax() + 1


def probability_percentage_of_each_digit(extract_methods, joint_test, nearest_labels_lst, index, k):
    frequents = []
    for i in range(len(extract_methods)):
        frequents.append(np.bincount(nearest_labels_lst[i][index][:k], minlength=10))

    result = []
    for i in range(len(extract_methods)):
        accuracy = []
        for x in range(0, 10):
            accuracy.append(frequents[i][x] / k)
        result.append([extract_methods[i], accuracy])
    return result
