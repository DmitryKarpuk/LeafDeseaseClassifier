import click
import pandas as pd
from pathlib import Path
from tensorflow import keras
from preprocessing import get_dataset


TARGET = ["healthy", "multiple_diseases", "rust", "scab"]


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
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-m",
    "--model-path",
    default="models/simple_model.h5",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--submission-path",
    default="data/simple_submission.csv",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
def predict(
    image_path: Path,
    metadata_path: Path,
    model_path: Path,
    submission_path: Path,
) -> None:
    """
    Predict health of apple leafs by using pretrained keras model.
    Save prediction to csv file.
    File format corresponds to Kaggle submission file format.

    Args:
        image_path: Path of folder with data of plant.
        metadata-path: Path of file with information about images.
        model_path: Path of pretrainde model file.
        submission_path: Path of submission file.

    Return:
        None.
    """
    img_indx = pd.read_csv(metadata_path)["image_id"]
    test_ds = get_dataset(image_path, metadata_path, False, mode="test")
    model = keras.models.load_model(model_path)
    pred = model.predict(test_ds)
    submission = pd.DataFrame(pred, columns=TARGET, index=img_indx)
    submission.index.name = "image_id"
    submission.to_csv(submission_path)
    click.echo(
        click.style(f"Submission is saved to {submission_path}", fg="green")
    )


if __name__ == "__main__":
    predict()
