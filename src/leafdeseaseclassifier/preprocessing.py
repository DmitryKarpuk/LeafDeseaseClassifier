import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DataFrameIterator


SEED = 2007
BATCH_SIZE = 64
TARGET = ['healthy', 'multiple_diseases',	'rust',	'scab']

np.random.seed(SEED)
tensorflow.keras.utils.set_random_seed(SEED)


def get_dataset(image_path: Path,
                metadata_path: Path,
                augmentation: bool) -> DataFrameIterator:

    meta_df = pd.read_csv(metadata_path)
    meta_df['image_path'] = str(image_path) + meta_df['image_id'] + '.jpg'
    
    if augmentation:
        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.1,
                                        horizontal_flip=True,)
    else:
        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_ds = train_gen.flow_from_dataframe(meta_df,
                                            x_col='image_path',
                                            y_col=TARGET,
                                            class_mode='raw',
                                            batch_size=BATCH_SIZE,
                                            seed=SEED)
    return train_ds










