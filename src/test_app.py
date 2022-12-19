import requests
import click


URL = "http://localhost:9696/predict"


@click.command(help="Script with request for testing model app.")
@click.option(
    "-u",
    "--url",
    default="https://cid-inc.com/app/uploads/2020/10/leaf_area.jpg",
    type=str,
)
def predict_req(url: str) -> None:
    """
    Test tensorflow learning service on url using url of leaf image.
    Args:
        url: Url of leaf image.
    Return:
        Probabilities of leaf state.
    """
    data = {'url': url}

    result = requests.post(URL, json=data).json()
    click.echo(click.style(result, fg="green"))

if __name__ == '__main__':
    predict_req()
