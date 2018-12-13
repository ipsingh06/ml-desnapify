# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import shutil
import cv2
import matplotlib.pyplot as plt
from enum import Enum


MIN_FACE_SIZE = 200

OUTPUT_IMAGE_WIDTH = 500
OUTPUT_IMAGE_HEIGHT = 500

INTERIM_ORIG_DIR = "orig"
INTERIM_TRANSFORMED_DIR = "transformed"


class FaceDetector:
    def __init__(self, cascades_path):
        self.cascade_path = os.path.join(cascades_path, 'haarcascade_frontalface_alt.xml')

    def has_face(self, img_path: str) -> bool:
        # extract pre-trained face detector
        face_cascade = cv2.CascadeClassifier(self.cascade_path)

        # load color (BGR) image
        img = cv2.imread(img_path)
        # convert BGR image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(cv_rgb)
        # plt.show()

        faces = face_cascade.detectMultiScale(gray)
        if len(faces) != 1:
            return False
        x, y, h, w = tuple(faces[0])
        return h > MIN_FACE_SIZE and w > MIN_FACE_SIZE


class ImageProcessor:
    class Filter(Enum):
        ORIGINAL = 1
        DOG = 2

    def __init__(self, output_width: int, output_height: int):
        self.__output_width = output_width
        self.__output_height = output_height

    def load(self, path: str):
        img = cv2.imread(path)
        # resize the image
        height, width, _ = img.shape
        scale_height = self.__output_height / height
        scale_width = self.__output_width / width
        if scale_height == scale_width:
            resize_height = self.__output_height
            resize_width = self.__output_width
        elif scale_height > scale_width:
            resize_height = self.__output_height
            resize_width = int(width * scale_height)
        else:
            resize_width = self.__output_width
            resize_height = int(height * scale_width)
        img = cv2.resize(img, (resize_width, resize_height))
        # crop image
        if resize_width > self.__output_width:
            x = (resize_width-self.__output_width)//2
            img = img[:, x:x+self.__output_width]
        elif resize_height > self.__output_height:
            y = (resize_height-self.__output_height)//2
            img = img[y:y+self.__output_height, :]
        return img

    def process(self, img, output_filter: Filter):
        pass


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('interim_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, interim_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    database_name = os.path.basename(input_filepath)

    # create interim directory
    database_interim_path = os.path.join(interim_filepath, database_name)
    database_orig_path = os.path.join(database_interim_path, INTERIM_ORIG_DIR)
    database_transformed_path = os.path.join(database_interim_path, INTERIM_TRANSFORMED_DIR)
    if os.path.exists(database_interim_path):
        shutil.rmtree(database_interim_path)
    os.makedirs(database_interim_path)
    os.makedirs(database_orig_path)
    os.makedirs(database_transformed_path)

    # grab raw images
    raw_images = [img for img in Path(input_filepath).glob('**/*.jpg')]
    raw_images = raw_images[15:25]

    # extract faces
    num_accepted = 0
    face_detector = FaceDetector(os.path.join(str(project_dir), 'src/data'))
    image_processor = ImageProcessor(output_width=OUTPUT_IMAGE_WIDTH, output_height=OUTPUT_IMAGE_HEIGHT)
    for raw_image_path in raw_images:
        raw_image_path = str(raw_image_path)
        has_face = face_detector.has_face(raw_image_path)
        if has_face:
            num_accepted += 1
            filename = "{:05}.jpg".format(num_accepted)
            print("{} -> {}".format(raw_image_path, filename))
            orig_image = image_processor.load(raw_image_path)
            cv2.imwrite(os.path.join(database_orig_path, filename), orig_image)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
