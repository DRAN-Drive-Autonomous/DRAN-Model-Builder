import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import datetime
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CONSTANTS
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100

def createCallbacks():
    timenow = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    log_dir = os.path.join(os.path.join("./train/", timenow), "logs/fit/")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_filepath = os.path.join(os.path.join("./train/", timenow), "checkpoint/fit/")
    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)
    checkpoint_filepath = os.path.join(checkpoint_filepath, "/weights.{epoch:02d}-{val_loss:.2f}.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    return [[tensorboard_callback, checkpoint_callback], [log_dir, checkpoint_filepath]]


def main():
    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Allow GPU memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU available. Training on GPU.")

        # Load data from NPZ file
        print("\nLoading and Preprocessing the data")
        data = np.load("../dataset/data_10000.npz")
        X = data['X'] / 255.0
        y = data['y']

        # Load your model
        print("\nLoading the  model")
        model_path = "../files/dranNN.h5"
        model = load_model(model_path)

        # Compile your model
        print("\nCompiling model")
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )   

        print("\nCreating callbacks")
        callbacks = createCallbacks()
        print(f"Checkpoint Filepath: {callbacks[1][1]}\nTensorboard Filepath: {callbacks[1][0]}")

        # Split data into training and validation sets
        print("\nSpliting dataset")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model using GPU if available
        print("\nTraining model")
        return
        with tf.device('/GPU:0'):
            history = model.fit(
                x=X_train,
                y=y_train,
                batch_size=batch_size,
                epochs=num_epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks[0])
            
            model.load_weights(callbacks[1][1])

        # Save the trained model
        print("Saving model")
        model.save("../files/dranNNT.h5")

    else:
        print("GPU not available. Training on CPU.")
    
    return

if __name__ == "__main__":
    main()