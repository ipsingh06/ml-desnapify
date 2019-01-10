# Desnapify

Desnapify is a deep convolutional generative adversarial networks (DCGAN)
trained to remove Snapchat filters from selfie images.
It is based on the excellent [pix2pix](https://phillipi.github.io/pix2pix) project by Isola et al.,
and specifically the [Keras](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)
implementation by [Thibault de Boissiere](https://github.com/tdeboissiere).

The following figure shows a few examples of the output of the Desnapify on the doggy filter selfie.
Top row is the input, middle row is the Desnapified output, bottom row is the target.

![Desnapification of the doggy filter](/docs/figures/sample_results.jpg?raw=true "Desnapification of the doggy filter")


Desnapify currently includes a model trained to remove the doggy filter from images of size 512x512.
It can be easily trained to remove other filters as well.
This is left for future work.

## Requirements

* Python 3.5
* Tensorflow (tensorflow-gpu with cuDNN is recommended)
* Keras
(Note: as of this writing the latest release of Keras 2.2.4 has a
[bug](https://github.com/keras-team/keras/issues/10648#issuecomment-441090611)
which prevents the model from being loaded. You must install the latest Keras from source as described
[here](https://github.com/keras-team/keras#installation))
* Python modules listed in requirements.txt

## Setup

1. Set up a virtualenv
   ```bash
   mkvirtualenv -p python3.5 desnapify
   ```

2. Clone this repo
   ```bash
   git clone https://github.com/ipsingh06/ml-desnapify.git
   ```

3. Install the python modules
   ```bash
   cd ml-desnapify
   pip install -r requirements.txt
   ```
4. Install Keras from source as described [here](https://github.com/keras-team/keras#installation)


## <a name="predict"></a> Using the provided model to Desnapify images

This section describes how to use the provided trained model to remove the doggy filter from your images.
The provided script `src/models/predict_model.py` takes care of scaling the images.
We will be using the samples provided in `data/raw/samples/doggy`.

1. Download the pre-trained model weights.
   ```bash
   python src/models/download_weights.py
   ```

2. Run the prediction script. This will cycle through and display the resulting images.
   ```bash
   python src/models/predict_model.py \
       models/doggy-512x512-v1/gen_weights_epoch030.h5 \
       data/raw/samples/doggy \
       --image_size 512 512
   ```

   To store the results into images files, specify the `--output <path to directory>` option.
   You can also specify `--no_concat` to output the output image only (and not the input).
   ```bash
   python src/models/predict_model.py \
       models/doggy-512x512-v1/gen_weights_epoch030.h5 \
       data/raw/samples/doggy \
       --image_size 512 512 \
       --output out \
       --no_concat
   ```

## <a name="train"></a> Training the model

This section describes how to train the model yourself.
This requires generating pairs of images with and without the filter applied.
The provided data generation script includes functionality to apply the doggy filter to images.
It can be extended to apply other Snapchat filters as well.


The following figure shows the pipeline for generating training data from selfie images.
![Data pipeline](/docs/figures/data_pipeline.png?raw=true "Data pipeline")


The provided model was trained with the [People's Republic of Tinder](https://www.kaggle.com/chrisroths/peoples-republic-of-tinder-1)
dataset available on Kaggle.
You should be able to use any dataset of selfie images for training.

Note: the training script uses multiprocess queues for batch loading, so you don't have to worry
about fitting your entire dataset into memory. Use as large a dataset as you like.


1. Place the set of selfie images in `data/raw/dataset` directory.

2. Run the data generation script to generate image-pairs with and without the doggy filter.
   ```bash
   python src/data/make_dataset.py apply-filter \
       data/raw/dataset \
       data/interim/dataset \
       --output_size 512 512 \
       --no_preserve_dir
   ```
   This script will create two directories `data/interim/dataset/orig` and `data/interim/dataset/transformed`.
   The first contains images to be used as input to the model, and the second images to be used as the target.

3. Run the data generation script again to split the dataset into training, validation and testing sets.
   This also packages into a single HDF5 file.
   ```bash
   python src/data/make_dataset.py create-hdf5 \
       data/interim/dataset \
       data/processed/dataset.h5
   ```
   This script will create the file `data/processed/dataset.h5`

3. We should verify that the dataset was created properly.
   ```bash
   python src/data/make_dataset.py check-hdf5 data/processed/dataset.h5
   ```

4. Finally, we can run the training script.
   ```bash
   python src/models/train_model.py \
       data/processed/dataset.h5 \
       --batch_size 4 \
       --patch_size 128 128 \
       --epochs 30
   ```

   We can visualize the performance metrics using Tensorboard:
   ```bash
   tensorboard --logdir=reports/logs
   ```
   ![Tensorboard](/docs/figures/training_tensorboard.png?raw=true "Tensorboard")

5. After training completes the model weights are saved in `models/`.
   To test the model, see the [Using the provided model](#predict) section.


## Acknowledgments

1. [pix2pix](https://phillipi.github.io/pix2pix) project by Isola et al.

2. [Keras](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)
implementation of pix2pix by [Thibault de Boissiere](https://github.com/tdeboissiere)

3. The apply-filters script is based off [snapchat-filters-opencv](https://github.com/charlielito/snapchat-filters-opencv)
