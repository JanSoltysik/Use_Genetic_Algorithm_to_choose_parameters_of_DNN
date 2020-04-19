def run_optimizer_fashion_mnist(alpha=1):
    import nn_optimizer
    from tensorflow import keras

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train /255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    op = nn_optimizer.NNOptimize(nn_size_scaler=alpha)
    model = op.fit(X_train[:10], y_train[:10])

    model.summary()
    model.evaluate(X_test, y_test)


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    run_optimizer_fashion_mnist()
