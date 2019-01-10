import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import logging
import click
import keras.backend as K
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from collections import namedtuple

import models
import data_utils


# Store information related to a batch
BatchRecord = namedtuple("BatchRecord", "input, truth, image_names")


def load_and_resize_image(path,
                          size,
                          fill_color=(0, 0, 0, 0)):
    im = Image.open(path)
    im.thumbnail(size, Image.ANTIALIAS)
    x, y = im.size
    out_x, out_y = size
    new_im = Image.new('RGB', size, fill_color)
    new_im.paste(im, ((out_x - x) // 2, (out_y - y) // 2))
    # Image is in RGB HxWxC format
    # HxWxC -> CxHxW
    open_cv_image = np.array(new_im).transpose([2, 0, 1])
    return open_cv_image


def get_batch_from_images(input_paths, truth_paths, batch_size, image_size):
    def gen_batch_array(paths):
        batch_images = [load_and_resize_image(path, image_size) for path in paths]
        batch = np.stack(batch_images)
        # pad with zeros if needed
        if batch.shape[0] < batch_size:
            padded_batch = np.zeros((batch_size,)+batch.shape[1:])
            padded_batch[:batch.shape[0], :, :, :] = batch
            batch = padded_batch
        batch = data_utils.normalization(batch)
        return batch

    for i in range(0, len(input_paths), batch_size):
        batch_input_paths = input_paths[i:i + batch_size]
        batch_truth_paths = truth_paths[i:i + batch_size]
        batch_record = BatchRecord(
            input=gen_batch_array(batch_input_paths),
            truth=gen_batch_array(batch_truth_paths) if truth_paths else None,
            image_names=[os.path.basename(path) for path in batch_input_paths]
        )
        yield batch_record


def get_batch_from_hdf5(generator):
    i = 1
    for truth_batch, input_batch in generator:
        batch_record = BatchRecord(
            input=input_batch,
            truth=truth_batch,
            image_names=["{:05}.jpg".format(j) for j in range(i, i+input_batch.shape[0])]
        )
        i += input_batch.shape[0]
        yield batch_record


# noinspection PyShadowingBuiltins
@click.command()
@click.argument('model', type=click.Path(exists=True))
@click.argument('input', type=click.Path(exists=True))
@click.option('-t', '--truth', type=click.Path(exists=True),
              help='Path to ground truth image or directory of images')
@click.option('-o', '--output', type=click.Path(),
              help='Output images to file')
@click.option('--image_size', type=(int, int), default=(256, 256),
              help='Model input size (default=256 256)')
@click.option('--concat/--no_concat', default=True,
              help='Whether to concat input and ground truth to output images (default=True)')
@click.option('-b', '--batch_size', type=int, default=4,
              help='default=4')
@click.option('--dataset', type=str, default='test',
              help='Dataset label to use for hdf5 (default=test)')
def main(model,
         input,
         truth,
         output,
         image_size,
         concat,
         batch_size,
         dataset):

    K.set_image_data_format("channels_first")

    # Load the model
    img_dim = (3,) + image_size
    generator_model = models.generator_unet_upsampling(img_dim)
    generator_model.load_weights(str(model))

    # Setup the data generator
    if h5py.is_hdf5(input):
        generator = data_utils.DataGenerator(file_path=input,
                                             dataset_type=dataset,
                                             batch_size=batch_size)
        batch_gen = get_batch_from_hdf5(generator)
        count = len(generator) * batch_size
    elif os.path.isdir(input):
        # Directory of images
        input_paths = [str(img) for img in sorted(Path(input).glob('**/*.jpg'))]
        if truth is not None:
            truth_paths = [str(img) for img in sorted(Path(truth).glob('**/*.jpg'))]
        else:
            truth_paths = []
        batch_gen = get_batch_from_images(input_paths, truth_paths, batch_size, image_size)
        count = len(input_paths)
    else:
        # Single image file
        input_paths = [input]
        if truth is not None:
            truth_paths = [truth]
        else:
            truth_paths = []
        batch_size = 1
        batch_gen = get_batch_from_images(input_paths, truth_paths, batch_size, image_size)
        count = 1

    # Setup the output
    if output is not None:
        # If input is multiple images, create directory for output images
        if count > 1:
            if os.path.exists(output):
                raise IOError('Output directory already exists')
            os.makedirs(output)

    # Inference
    with tqdm(None, total=count) as progress_bar:
        for batch_record in batch_gen:
            input_batch = batch_record.input
            truth_batch = batch_record.truth
            image_names = batch_record.image_names
            out_batch = generator_model.predict(input_batch)
            progress_bar.update(input_batch.shape[0])

            if output is not None:
                # Individually save each image in the batch
                for i in range(len(image_names)):
                    input_b = input_batch[i:i+1]
                    output_b = out_batch[i:i+1]
                    truth_b = truth_batch[i:i+1] if truth_batch is not None else None
                    image_name = image_names[i]
                    out_image = generate_output_image(input_b, output_b, truth_b, concat)
                    if count > 1:
                        out_path = os.path.join(str(output), image_name)
                    else:
                        out_path = output
                    plt.imsave(out_path, out_image)
            else:
                # Show the image
                out_image = generate_output_image(input_batch, out_batch, truth_batch, concat)
                plt.figure()
                plt.imshow(out_image)
                plt.show()
                plt.clf()
                plt.close()


def generate_output_image(input_batch,
                          output_batch,
                          truth_batch,
                          concat,
                          images_per_row=4):
    n = min(images_per_row, input_batch.shape[0])
    # Incoming format: BxCxHxW
    list_rows = []
    for r in range(int(input_batch.shape[0] // n)):
        # in each row, concat n images along the width
        # input on the top, output in the middle, groud truth at the bottom
        # new format in each row: CxHxW
        if concat:
            list_rows.append(
                np.concatenate([input_batch[k] for k in range(n * r, n * (r + 1))], axis=2)
            )
        list_rows.append(
            np.concatenate([output_batch[k] for k in range(n * r, n * (r + 1))], axis=2)
        )
        if concat and truth_batch is not None:
            list_rows.append(
                np.concatenate([truth_batch[k] for k in range(n * r, n * (r + 1))], axis=2)
            )
    # concat rows along the height
    # new format: CxHxW
    plt_image = np.concatenate(list_rows, axis=1)
    # CxHxW -> HxWxC
    plt_image = plt_image.transpose([1, 2, 0])
    plt_image = data_utils.inverse_normalization(plt_image)
    return plt_image


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = str(Path(__file__).resolve().parents[2])

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
