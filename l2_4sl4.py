import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bisect import bisect_left
import itertools

TRAINING_SET_SIZE = 201
TEST_SET_SIZE = 101
K_RANGE = range(1, 61)
NUM_FOLDS = 5

def generate_samples(X, sigma = 0.2):
    return np.sin(2*np.pi*X) + sigma * np.random.randn(len(X))

def k_nearest_indices(X_train, x, k) -> tuple[int, int]:
    n = len(X_train)

    if k < 0 or k > n:
        raise ValueError("k must be between 0 and len(arr)")

    if k == 0:
        return (0, 0)

    left = right = bisect_left(X_train, x)

    while right - left < k:
        if left == 0:
            right += 1
        elif right == n:
            left -= 1
        elif abs(X_train[left - 1] - x) <= abs(X_train[right] - x):
            left -= 1
        else:
            right += 1

    return range(left, right)

def predict(X_train, t_train, x, k):
    nearest_t_sum = 0
    for i in k_nearest_indices(X_train, x, k):
        nearest_t_sum += t_train[i]
    return nearest_t_sum / k

def rmse(X_train, t_train, X_valid, t_valid, k):
    total_squared_error = 0
    for i in range(len(X_valid)):
        total_squared_error += (predict(X_train, t_train, X_valid[i], k) - t_valid[i]) ** 2
    return np.sqrt(total_squared_error / len(X_valid))

def divide_folds(data: np.ndarray, num_folds):
    n = len(data)
    folds = []
    for i in range(num_folds):
        folds.append([])
    i = 0
    while True:
        for j in range(num_folds):
            if i < n:
                folds[j].append(data[i])
                i += 1
            else:
                return folds

if __name__ == "__main__":

    Path("./out").mkdir(parents=True, exist_ok=True)
    
    np.random.seed(616)

    X_train = np.linspace(0.,1.,TRAINING_SET_SIZE)
    t_train = generate_samples(X_train)

    X_test = np.linspace(0., 1., TEST_SET_SIZE)
    t_test = generate_samples(X_test)

    cross_validation_errors = []
    training_errors = []
    for k in K_RANGE:
        total_cross_validation_error = 0
        for i in range(5):
            X_train_folds = divide_folds(X_train, NUM_FOLDS)
            t_train_folds = divide_folds(t_train, NUM_FOLDS)
            X_cross_valid = X_train_folds.pop(i)
            t_cross_valid = t_train_folds.pop(i)
            X_cross_train = list(itertools.chain.from_iterable(X_train_folds))
            t_cross_train = list(itertools.chain.from_iterable(t_train_folds))
            total_cross_validation_error += rmse(X_cross_train, t_cross_train, X_cross_valid, t_cross_valid, k)
        cross_validation_errors.append(total_cross_validation_error / 5)
        training_errors.append(rmse(X_train, t_train, X_train, t_train, k))

    plt.plot(K_RANGE, cross_validation_errors, label='Cross Validation Error', color='blue', marker='.')
    plt.plot(K_RANGE, training_errors, label='Training Error', color='red', marker='.')
    plt.title("Error vs k")
    plt.xlabel("k")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("out/errors_vs_k")
    plt.close()

    k = 4
    predictor = []
    for x in X_train:
        predictor.append(predict(X_train, t_train, x, k))
    f_opt = generate_samples(X_train, 0)
    test_error = rmse(X_train, t_train, X_test, t_test, k)
    print(f"Test error with k = {k}: {test_error}")

    plt.scatter(X_train, f_opt, label='f_opt', color='red', marker='.')
    plt.scatter(X_train, predictor, label=f'Predictons', color='blue', marker='.')
    plt.scatter(X_test, t_test, label='Test Samples', color='green', marker='.')
    plt.title(f"{k}NN Model")
    plt.xlabel("Feature")
    plt.ylabel("Label")
    plt.legend()
    plt.savefig("out/knn_model")
    plt.close()


