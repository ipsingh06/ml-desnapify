import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import logging
import time
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import plot_model, generic_utils
from keras.callbacks import TensorBoard
import numpy as np
import click
import math
from datetime import datetime

import data_utils
import models


@click.command()
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--batch_size', type=int, default=4)
@click.option('--patch_size', type=(int, int), default=(64, 64))
@click.option('--epochs', type=int, default=10)
@click.option('--label_smoothing/--no_label_smoothing', default=False)
@click.option('--label_flipping', type=click.FloatRange(0.0, 1.0), default=0.0)
def main(dataset,
         batch_size,
         patch_size,
         epochs,
         label_smoothing,
         label_flipping):
    print(project_dir)

    image_data_format = "channels_first"
    K.set_image_data_format(image_data_format)

    # configuration parameters
    print("Config params:")
    print("  dataset = {}".format(dataset))
    print("  batch_size = {}".format(batch_size))
    print("  patch_size = {}".format(patch_size))
    print("  epochs = {}".format(epochs))
    print("  label_smoothing = {}".format(label_smoothing))
    print("  label_flipping = {}".format(label_flipping))

    model_dir = os.path.join(project_dir, "models")
    fig_dir = os.path.join(project_dir, "reports", "figures")
    logs_dir = os.path.join(
        project_dir, "reports", "logs",
        datetime.strftime(datetime.now(), '%y%m%d-%H%M')
    )

    # Load and rescale data
    X_transformed_train, X_orig_train, X_transformed_val, X_orig_val = \
        data_utils.load_data(dataset)
    img_dim = X_orig_train.shape[-3:]

    n_batch_per_epoch = int(math.ceil(X_orig_train.shape[0] / batch_size))
    epoch_size = n_batch_per_epoch * batch_size

    print("Derived params:")
    print("  # training samples = {}".format(X_orig_train.shape[0]))
    print("  # validation samples = {}".format(X_orig_val.shape[0]))
    print("  n_batch_per_epoch = {}".format(n_batch_per_epoch))
    print("  epoch_size = {}".format(epoch_size))

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size)

    tensorboard = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=True
    )

    try:
        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        generator_model = models.generator_unet_upsampling(img_dim)
        generator_model.summary()
        plot_model(generator_model,
                   to_file=os.path.join(fig_dir, "generator_model.png"),
                   show_shapes=True,
                   show_layer_names=True)

        # Load discriminator model
        discriminator_model = models.DCGAN_discriminator(img_dim_disc, nb_patch)
        discriminator_model.summary()
        plot_model(discriminator_model,
                   to_file=os.path.join(fig_dir, "discriminator_model.png"),
                   show_shapes=True,
                   show_layer_names=True)

        generator_model.compile(loss='mae', optimizer=opt_discriminator)
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   img_dim,
                                   patch_size,
                                   image_data_format)

        loss = [models.l1_loss, 'binary_crossentropy']
        loss_weights = [1E1, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        tensorboard.set_model(DCGAN_model)

        # Start training
        print("Start training")
        for e in range(epochs):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            for X_transformed_batch, X_orig_batch in data_utils.gen_batch(
                    X_transformed_train, X_orig_train, batch_size):

                # Create a batch to feed the discriminator model
                X_disc, y_disc = data_utils.get_disc_batch(
                    X_transformed_batch,
                    X_orig_batch,
                    generator_model,
                    batch_counter,
                    patch_size,
                    label_smoothing=label_smoothing,
                    label_flipping=label_flipping
                )

                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

                # Create a batch to feed the generator model
                X_gen_target, X_gen = next(data_utils.gen_batch(
                    X_transformed_train, X_orig_train, batch_size
                ))
                y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                batch_counter += 1
                metrics = [("D logloss", disc_loss),
                           ("G tot", gen_loss[0]),
                           ("G L1", gen_loss[1]),
                           ("G logloss", gen_loss[2])]
                progbar.add(batch_size, values=metrics)

                tensorboard.on_batch_end(
                    batch_counter,
                    logs={k: v for (k, v) in metrics}
                )

                # Save images for visualization
                if batch_counter % (n_batch_per_epoch / 2) == 0:
                    # Get new images from validation
                    data_utils.plot_generated_batch(
                        X_transformed_batch,
                        X_orig_batch,
                        generator_model,
                        os.path.join(fig_dir, "current_batch_training.png")
                    )
                    X_transformed_batch, X_orig_batch = next(data_utils.gen_batch(
                        X_transformed_val, X_orig_val, batch_size
                    ))
                    data_utils.plot_generated_batch(
                        X_transformed_batch,
                        X_orig_batch,
                        generator_model,
                        os.path.join(fig_dir, "current_batch_validation.png")
                    )

                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, epochs, time.time() - start))
            tensorboard.on_epoch_end(
                e + 1,
                logs={k: v for (k, v) in metrics}
            )

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = str(Path(__file__).resolve().parents[2])

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
