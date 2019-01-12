import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import logging
import time
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import plot_model, generic_utils
from keras.utils.data_utils import OrderedEnqueuer
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import click
from datetime import datetime
import sys

import data_utils
import models


@click.command()
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--batch_size', type=int, default=4,
              help='default=4')
@click.option('--patch_size', type=(int, int), default=(64, 64),
              help='default=64 64')
@click.option('--epochs', type=int, default=10,
              help='default=10')
@click.option('--label_smoothing/--no_label_smoothing', default=False,
              help='default=False')
@click.option('--label_flipping', type=click.FloatRange(0.0, 1.0), default=0.0,
              help='default=0.0')
def main(dataset,
         batch_size,
         patch_size,
         epochs,
         label_smoothing,
         label_flipping):
    print(project_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

    image_data_format = "channels_first"
    K.set_image_data_format(image_data_format)

    save_images_every_n_batches = 30
    save_model_every_n_epochs = 0

    # configuration parameters
    print("Config params:")
    print("  dataset = {}".format(dataset))
    print("  batch_size = {}".format(batch_size))
    print("  patch_size = {}".format(patch_size))
    print("  epochs = {}".format(epochs))
    print("  label_smoothing = {}".format(label_smoothing))
    print("  label_flipping = {}".format(label_flipping))
    print("  save_images_every_n_batches = {}".format(save_images_every_n_batches))
    print("  save_model_every_n_epochs = {}".format(save_model_every_n_epochs))

    model_name = datetime.strftime(datetime.now(), '%y%m%d-%H%M')
    model_dir = os.path.join(project_dir, "models", model_name)
    fig_dir = os.path.join(project_dir, "reports", "figures")
    logs_dir = os.path.join(project_dir, "reports", "logs", model_name)

    os.makedirs(model_dir)

    # Load and rescale data
    ds_train_gen = data_utils.DataGenerator(file_path=dataset,
                                            dataset_type="train",
                                            batch_size=batch_size)
    ds_train_disc = data_utils.DataGenerator(file_path=dataset,
                                             dataset_type="train",
                                             batch_size=batch_size)
    ds_val = data_utils.DataGenerator(file_path=dataset,
                                      dataset_type="val",
                                      batch_size=batch_size)
    enq_train_gen = OrderedEnqueuer(ds_train_gen,
                                    use_multiprocessing=True,
                                    shuffle=True)
    enq_train_disc = OrderedEnqueuer(ds_train_disc,
                                     use_multiprocessing=True,
                                     shuffle=True)
    enq_val = OrderedEnqueuer(ds_val,
                              use_multiprocessing=True,
                              shuffle=False)

    img_dim = ds_train_gen[0][0].shape[-3:]

    n_batch_per_epoch = len(ds_train_gen)
    epoch_size = n_batch_per_epoch * batch_size

    print("Derived params:")
    print("  n_batch_per_epoch = {}".format(n_batch_per_epoch))
    print("  epoch_size = {}".format(epoch_size))
    print("  n_batches_val = {}".format(len(ds_val)))

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size)

    tensorboard = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=True,
        update_freq='batch'
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
        # TODO: modify disc to accept real input as well
        discriminator_model = models.DCGAN_discriminator(img_dim_disc, nb_patch)
        discriminator_model.summary()
        plot_model(discriminator_model,
                   to_file=os.path.join(fig_dir, "discriminator_model.png"),
                   show_shapes=True,
                   show_layer_names=True)

        # TODO: pretty sure this is unnecessary
        generator_model.compile(loss='mae', optimizer=opt_discriminator)
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   img_dim,
                                   patch_size,
                                   image_data_format)

        # L1 loss applies to generated image, cross entropy applies to predicted label
        loss = [models.l1_loss, 'binary_crossentropy']
        loss_weights = [1E1, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        tensorboard.set_model(DCGAN_model)

        # Start training
        enq_train_gen.start(workers=1, max_queue_size=20)
        enq_train_disc.start(workers=1, max_queue_size=20)
        enq_val.start(workers=1, max_queue_size=20)
        out_train_gen = enq_train_gen.get()
        out_train_disc = enq_train_disc.get()
        out_val = enq_val.get()

        print("Start training")
        for e in range(1, epochs+1):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            start = time.time()

            for batch_counter in range(1, n_batch_per_epoch+1):
                X_transformed_batch, X_orig_batch = next(out_train_disc)

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
                X_gen_target, X_gen = next(out_train_gen)
                y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                # Set labels to 1 (real) to maximize the discriminator loss
                y_gen[:, 1] = 1

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                metrics = [("D logloss", disc_loss),
                           ("G tot", gen_loss[0]),
                           ("G L1", gen_loss[1]),
                           ("G logloss", gen_loss[2])]
                progbar.add(batch_size, values=metrics)

                logs = {k: v for (k, v) in metrics}
                logs["size"] = batch_size

                tensorboard.on_batch_end(
                    batch_counter,
                    logs=logs
                )

                # Save images for visualization
                if batch_counter % save_images_every_n_batches == 0:
                    # Get new images from validation
                    data_utils.plot_generated_batch(
                        X_transformed_batch,
                        X_orig_batch,
                        generator_model,
                        os.path.join(logs_dir, "current_batch_training.png")
                    )
                    X_transformed_batch, X_orig_batch = next(out_val)
                    data_utils.plot_generated_batch(
                        X_transformed_batch,
                        X_orig_batch,
                        generator_model,
                        os.path.join(logs_dir, "current_batch_validation.png")
                    )

            print("")
            print('Epoch %s/%s, Time: %s' % (e, epochs, time.time() - start))
            tensorboard.on_epoch_end(e, logs=logs)

            if (save_model_every_n_epochs >= 1 and e % save_model_every_n_epochs == 0) or \
                    (e == epochs):
                print("Saving model for epoch {}...".format(e), end="")
                sys.stdout.flush()
                gen_weights_path = os.path.join(model_dir, 'gen_weights_epoch{:03d}.h5'.format(e))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join(model_dir, 'disc_weights_epoch{:03d}.h5'.format(e))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join(model_dir, 'DCGAN_weights_epoch{:03d}.h5'.format(e))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)
                print("done")

    except KeyboardInterrupt:
        pass

    enq_train_gen.stop()
    enq_train_disc.stop()
    enq_val.stop()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = str(Path(__file__).resolve().parents[2])

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
