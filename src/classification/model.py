import tensorflow_hub as hub
import tf_keras as keras

IMAGE_SIZE = (299, 299)
NUM_CLASSES = 4

INCEPTION_V3_URL = (
    "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5"
)

BACKBONE_NAME = "inception_v3_feature_extractor"


def build_model(
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    trainable: bool = False,
):
    """Build InceptionV3 + classification head.

    ``trainable=False`` builds a frozen backbone (feature extraction),
    which is the starting point of the two-phase training pipeline.
    """
    backbone = hub.KerasLayer(
        INCEPTION_V3_URL,
        trainable=trainable,
        name=BACKBONE_NAME,
    )

    inputs = keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="mri_image",
    )

    features = backbone(inputs)

    x = keras.layers.Dropout(dropout_rate)(features)

    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_output",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="BrainScan_InceptionV3",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def set_backbone_trainable(model, trainable: bool = True) -> None:
    """Enable or disable fine-tuning of the InceptionV3 backbone.

    TensorFlow Hub's ``KerasLayer`` is a single opaque layer, so the whole
    module is toggled at once (sub-layer freezing is not exposed by the API).
    Call ``model.compile(...)`` afterwards, normally with a much smaller
    learning rate than the feature-extraction phase.
    """
    model.get_layer(BACKBONE_NAME).trainable = trainable