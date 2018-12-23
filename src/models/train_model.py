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

import data_utils
import models


def main():
    print(project_dir)

    image_data_format = "channels_first"
    K.set_image_data_format(image_data_format)

    # configuration parameters
    batch_size = 4
    n_batch_per_epoch = 100
    patch_size = (64, 64)
    nb_epoch = 10
    label_smoothing = False
    label_flipping = 0

    epoch_size = n_batch_per_epoch * batch_size

    model_dir = os.path.join(project_dir, "models")
    fig_dir = os.path.join(project_dir, "reports", "figures")
    logs_dir = os.path.join(project_dir, "reports", "logs")
    data_dir = os.path.join(project_dir, "data", "processed")

    dataset_file = os.path.join(data_dir, "tinder_small_256.h5")

    # Load and rescale data
    X_transformed_train, X_orig_train, X_transformed_val, X_orig_val = \
        data_utils.load_data(dataset_file)
    img_dim = X_orig_train.shape[-3:]

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
        for e in range(nb_epoch):
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
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
            tensorboard.on_epoch_end(
                e + 1,
                logs={k: v for (k, v) in metrics}
            )

    except KeyboardInterrupt:
        pass

    tensorboard.on_train_end()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = str(Path(__file__).resolve().parents[2])

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
