def run_optimizer_classification(alpha=0.2, dataset_name="fashion_mnist"):
    import importlib
    import nn_optimizer
    from tensorflow import keras

    (X_train, y_train), (X_test, y_test) = \
        importlib.import_module(f"tensorflow.keras.datasets.{dataset_name}").load_data()
    X_train = X_train /255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    op = nn_optimizer.NNOptimize(nn_size_scaler=alpha)
    model = op.fit(X_train, y_train)

    model.summary()
    model.evaluate(X_test, y_test)


if __name__ == '__main__':
    import sys
    import logging
    import tensorflow as tf
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    sys.path.append('..')


    run_optimizer_classification()
