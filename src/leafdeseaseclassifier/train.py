import click
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from .preprocessing import get_dataset
from tensorflow.keras.applications.resnet50  import ResNet50
import warnings

warnings.filterwarnings("ignore")

SEED = 2007

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

@click.command()
@click.option(
    "-i",
    "--image-path",
    default="data/images",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
)
@click.option(
    "-d",
    "--metadata-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-m",
    "--model-path",
    default="models/model.h5",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-p",
    "--param-path",
    default="config/train_params.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def train(
    image_path: Path,
    metadata_path: Path,
    model_path: Path,
    param_path: Path
) -> None:
    """
    Train a ResNet50 model on plant pathology dataset with сertain parameters.
    Save trained model to .h5 file.
    
    Args:
        image_path: Path of folder with data of plant.
        metadata-path: Path of file with information about images.
        model_path: Path of model file.
        param-path: Path of file with train parameters.

    Return:
        None
    """
    params = yaml.safe_load(param_path.read_text())
    size = (params['inner_size'], params['inner_size'], 3)
    augmentation = params['augmentation']
    train_ds = get_dataset(image_path, metadata_path, augmentation)
    # Create base model
    base_model = ResNet50(weights='imagenet', input_shape=size, include_top=False)
    base_model.trainable = False
    # Create a new model on top.
    inputs = keras.Input(shape=size)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(params['drop_rate'])(x)
    outputs = keras.layers.Dense(4, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    print('====== Strart transfer-learning ======')
    # Transfer-learning train
    model.compile(
                  optimizer=keras.optimizers.Adam(params['lr_tl']),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.CategoricalAccuracy()],
                  )
    model.fit(train_ds, epochs=params['n_epoch'])
    if params['fine_tune']:
        print('====== Strart fine-tuning ======')
        # Fine-tuning train
        base_model.trainable=True

        model.compile(
                      optimizer=keras.optimizers.Adam(params['lr_ft']),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=[keras.metrics.CategoricalAccuracy()],
                      )
        model.fit(train_ds, epochs=params['n_epoch'])
    model.save(model_path)
    click.echo(click.style("Model was successful saved.", fg="green"))

    if __name__ == '__main__':
        train()
