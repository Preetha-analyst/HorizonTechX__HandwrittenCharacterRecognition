import os
import numpy as np
from scipy.io import loadmat

import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

# ==========================================================
# EMNIST Character Mapping
# ==========================================================
EMNIST_MAPPING = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# ==========================================================
# Load EMNIST ByClass (.mat)
# ==========================================================
def load_emnist_dataset():

    print("=" * 60)
    print("Loading EMNIST ByClass Dataset...")
    print("=" * 60)

    mat_path = os.path.join(
        "emnist_extracted",
        "matlab",
        "emnist-byclass.mat"
    )

    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"\nDataset not found:\n{mat_path}\n"
            "Extract matlab.zip first."
        )

    data = loadmat(mat_path)

    dataset = data["dataset"]

    obj = dataset[0, 0]

    train = obj["train"]
    print("Train type:", type(train))
    print("Train shape:", train.shape)

    print("\nTrain object:")
    print(train)

    print("\nTrain[0,0] type:", type(train[0,0]))
    print("Train[0,0]:", train[0,0])
    test = obj["test"]

    # Extract train data
    train_data = train[0, 0]

    X_train = train_data[0]
    y_train = train_data[1]

    # Extract test data
    test_data = test[0, 0]

    X_test = test_data[0]
    y_test = test_data[1]

    print("Training Images :", X_train.shape)
    print("Training Labels :", y_train.shape)
    print("Testing Images  :", X_test.shape)
    print("Testing Labels  :", y_test.shape)

    return X_train, y_train, X_test, y_test


# ==========================================================
# Preprocess Images
# ==========================================================
def preprocess_images(images):

    images = images.reshape((-1, 28, 28))

    corrected = []

    for img in images:

        img = np.transpose(img)

        img = np.fliplr(img)

        corrected.append(img)

    corrected = np.array(corrected)

    corrected = corrected.reshape((-1, 28, 28, 1))

    corrected = corrected.astype("float32") / 255.0

    return corrected


# ==========================================================
# Prepare Dataset
# ==========================================================
def prepare_dataset():

    X_train, y_train, X_test, y_test = load_emnist_dataset()

    print("\nPreprocessing images...")

    X_train = preprocess_images(X_train)

    X_test = preprocess_images(X_test)

    y_train = y_train.flatten()

    y_test = y_test.flatten()

    print("\nDataset Ready")

    print("Train :", X_train.shape)

    print("Test  :", X_test.shape)

    print("Classes :", len(np.unique(y_train)))

    return X_train, y_train, X_test, y_test                                                    # ==========================================================
# Build Advanced CNN Model
# ==========================================================
def build_advanced_cnn():

    print("\n" + "=" * 60)
    print("Building Advanced CNN Model...")
    print("=" * 60)

    model = models.Sequential([

        # ---------------- Block 1 ----------------
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Dropout(0.25),

        # ---------------- Block 2 ----------------
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Dropout(0.30),

        # ---------------- Block 3 ----------------
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Dropout(0.35),

        # ---------------- Classifier ----------------
        layers.Flatten(),

        layers.Dense(
            512,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(0.50),

        layers.Dense(
            256,
            activation="relu"
        ),

        layers.Dropout(0.40),

        layers.Dense(
            62,
            activation="softmax"
        )

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )

    model.summary()

    return model


# ==========================================================
# Callbacks
# ==========================================================
def get_callbacks():

    early_stop = callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=3,

        restore_best_weights=True

    )

    checkpoint = callbacks.ModelCheckpoint(

        "best_emnist_model.keras",

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1

    )

    reduce_lr = callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        verbose=1

    )

    return [

        early_stop,

        checkpoint,

        reduce_lr

    ]                                                                                                                             # ==========================================================
# Train Model
# ==========================================================
def train_model():

    X_train, y_train, X_test, y_test = prepare_dataset()

    model = build_advanced_cnn()

    print("\n" + "=" * 60)
    print("Training Started...")
    print("=" * 60)

    history = model.fit(

        X_train,

        y_train,

        epochs=10,

        batch_size=128,

        validation_data=(X_test, y_test),

        callbacks=get_callbacks(),

        verbose=1

    )

    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60)

    loss, accuracy = model.evaluate(

        X_test,

        y_test,

        verbose=1

    )

    print("\nFinal Test Accuracy : {:.2f}%".format(accuracy * 100))
    print("Final Test Loss     : {:.4f}".format(loss))

    print("\nSaving Final Model...")

    model.save("advanced_emnist_model.h5")

    print("Model saved successfully.")
    print("File : advanced_emnist_model.h5")

    # ------------------------------------------------------
    # Sample Prediction
    # ------------------------------------------------------

    print("\nTesting Sample Predictions...\n")

    predictions = model.predict(X_test[:10], verbose=0)

    for i in range(10):

        predicted = np.argmax(predictions[i])

        actual = y_test[i]

        print(
            f"Sample {i+1} : "
            f"Actual = {EMNIST_MAPPING[actual]} | "
            f"Predicted = {EMNIST_MAPPING[predicted]}"
        )

    return model, history


# ==========================================================
# Main Function
# ==========================================================
if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("ADVANCED EMNIST HANDWRITTEN CHARACTER RECOGNITION")
    print("=" * 60)

    train_model()

    print("\n")
    print("=" * 60)
    print("Training Completed Successfully")
    print("=" * 60)