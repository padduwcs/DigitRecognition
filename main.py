import process
import predict
from matplotlib import pyplot as plt

def main():
    x_train, y_train = process.load_mnist("data/", kind="train")
    x_test, y_test = process.load_mnist("data/", kind="t10k")

    train_flat, train_chunk, train_histogram = process.extract_features(x_train)
    test_flat, test_chunk, test_histogram = process.extract_features(x_test)

    combined_train_flat = process.combine(train_flat, y_train)
    combined_train_chunk = process.combine(train_chunk, y_train)
    combined_train_histogram = process.combine(train_histogram, y_train)
    combined_test_flat = process.combine(test_flat, y_test)
    combined_test_chunk = process.combine(test_chunk, y_test)
    combined_test_histogram = process.combine(test_histogram, y_test)

    flat_nearest_vectors = 'data/flat_nearest_vectors'
    chunk_nearest_vectors = 'data/chunk_nearest_vectors'
    histogram_nearest_vectors = 'data/histogram_nearest_vectors'

    predict.gen_k_nearest_labels(combined_test_flat, combined_train_flat, flat_nearest_vectors, k = predict.K_MAX)
    predict.gen_k_nearest_labels(combined_test_chunk, combined_train_chunk, chunk_nearest_vectors)
    predict.gen_k_nearest_labels(combined_test_histogram, combined_train_histogram, histogram_nearest_vectors)

    flat_data = predict.load_binary(flat_nearest_vectors)
    chunk_data = predict.load_binary(chunk_nearest_vectors)
    histogram_data = predict.load_binary(histogram_nearest_vectors)

    k_range = range(1, 1001)
    predict.graph_test_accuracy(combined_test_flat, flat_data, k_range)
    #predict.graph_test_accuracy(combined_test_chunk, chunk_data, k_range)
    #predict.graph_test_accuracy(combined_test_histogram, histogram_data, k_range)
    print("Sampling optimized k for test =", find_optimize_k(combined_test_chunk, chunk_data, k_range)

if __name__ == "__main__":
    main()
