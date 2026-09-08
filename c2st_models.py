from __future__ import annotations

import tensorflow as tf


def make_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    key = str(name).strip().lower()
    if key == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if key == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    raise ValueError(
        f"Unknown optimizer {name!r}. Supported values are 'adam' and 'sgd'."
    )


def build_binary_classifier(
    n_features: int,
    *,
    hidden,
    optimizer: str,
    learning_rate: float,
    batch_normalization: bool = False,
    seed: int = 0,
) -> tf.keras.Model:
    """Build the dense binary classifier used by the C2ST/DCTR framework."""
    hidden = tuple(int(n) for n in hidden)
    if any(n <= 0 for n in hidden):
        raise ValueError(f"All hidden-layer sizes must be positive, got {hidden!r}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    layers = [tf.keras.layers.Input(shape=(n_features,))]
    if batch_normalization:
        layers.append(tf.keras.layers.BatchNormalization())

    for n_nodes in hidden:
        layers.append(tf.keras.layers.Dense(n_nodes, activation="relu"))

    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=make_optimizer(optimizer, learning_rate),
        loss="binary_crossentropy",
    )
    return model
