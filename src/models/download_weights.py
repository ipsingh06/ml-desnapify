from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import requests
import re
import logging
from tqdm import tqdm


FILES_TO_DOWNLOAD = [
    (
        '1stzgfygorYHXspYa8J6gG5dA4VnE5jOg',  # DCGAN_weights_epoch030.h5
        os.path.join('models', 'doggy-512x512-v1')
    ),
    (
        '1l8v8PtvM3IchZP6AucDLKcmxkQovLjcg',  # disc_weights_epoch030.h5
        os.path.join('models', 'doggy-512x512-v1')
    ),
    (
        '1XRT2JKuJMY9Sb57lci-0cY25EbrKvERV',  # gen_weights_epoch030.h5
        os.path.join('models', 'doggy-512x512-v1')
    ),
]


def download_file_from_google_drive(id, dir_path):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(_response, _destination, file_size):
        CHUNK_SIZE = 32768

        with tqdm(total=file_size, unit='B', unit_scale=True, desc=_destination) as progress_bar:
            path = os.path.join(str(project_dir), _destination)
            with open(path, "wb") as f:
                for chunk in _response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        progress_bar.update(CHUNK_SIZE)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    headers = {'Range': 'bytes=0-'}
    response = session.get(URL, params={'id': id }, headers=headers, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, headers=headers, stream=True)

    content_range = response.headers['Content-Range']
    content_length = int(content_range.partition('/')[-1])
    filename = re.findall("filename=\"(.+)\"", response.headers['Content-Disposition'])[0]

    # File checks
    file_path = os.path.join(dir_path, filename)
    if os.path.exists(file_path):
        print("Skipping file '{}'... already exists".format(filename))
        return
    os.makedirs(dir_path, exist_ok=True)

    save_response_content(response, file_path, content_length)


def main():
    for id, dir_path in FILES_TO_DOWNLOAD:
        download_file_from_google_drive(
            id,
            dir_path
        )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
