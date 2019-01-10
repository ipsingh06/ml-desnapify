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


## Using the provided model to Desnapify images

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

## Training the model

This section describes how to train the model yourself.
This requires generating pairs of images with and without the filter applied.
The provided data generation script includes functionality to apply the doggy filter to images.
It can be extended to apply other Snapchat filters as well.

