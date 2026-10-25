import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4

GOOGLENET_URL = (
    "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5"
)


def build_model(num_classes: int = NUM_CLASSES):
   
    backbone = hub.KerasLayer(
        GOOGLENET_URL,
        trainable=False,
        name="googlenet_feature_extractor",
    )

    inputs = keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="mri_image",
    )


    features = backbone(inputs)

    x = keras.layers.Dropout(0.3)(features)

    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_output",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="BrainScan_GoogLeNet",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()

    print("\nOutput shape:", model.output_shape)
    print(
        "Trainable parameters:",
        sum(
            keras.backend.count_params(w)
            for w in model.trainable_weights
        ),
    )