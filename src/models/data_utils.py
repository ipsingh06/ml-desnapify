import numpy as np
import h5py
import matplotlib.pylab as plt


def normalization(X):
    return X / 127.5 - 1


def inverse_normalization(X):
    return (X + 1.) / 2.


def load_data(file):
    with h5py.File(file, "r") as hf:

        X_transformed_train = hf["train_transformed"][:].astype(np.float32)
        X_transformed_train = normalization(X_transformed_train)

        X_orig_train = hf["train_orig"][:].astype(np.float32)
        X_orig_train = normalization(X_orig_train)

        X_transformed_val = hf["val_transformed"][:].astype(np.float32)
        X_transformed_val = normalization(X_transformed_val)

        X_orig_val = hf["val_orig"][:].astype(np.float32)
        X_orig_val = normalization(X_orig_val)

        return X_transformed_train, X_orig_train, X_transformed_val, X_orig_val


def get_nb_patch(img_dim, patch_size):
    assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
    assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
    nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
    img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])
    return nb_patch, img_dim_disc


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx]


def get_disc_batch(X_transformed_batch, X_orig_batch, generator_model, batch_counter, patch_size,
                   label_smoothing=False, label_flipping=0):
    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_orig_batch)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_transformed_batch
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, patch_size)

    return X_disc, y_disc


def extract_patches(X, patch_size):
    # Now extract patches form X_disc
    X = X.transpose(0, 2, 3, 1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    for i in range(len(list_X)):
        list_X[i] = list_X[i].transpose([0, 3, 1, 2])

    return list_X


def plot_generated_batch(X_transformed, X_orig, generator_model, file_path):
    # Generate images
    X_gen = generator_model.predict(X_orig)

    X_orig = inverse_normalization(X_orig)
    X_transformed = inverse_normalization(X_transformed)
    X_gen = inverse_normalization(X_gen)

    Xo = X_orig[:8]
    Xg = X_gen[:8]
    Xt = X_transformed[:8]

    X = np.concatenate((Xo, Xg, Xt), axis=0)
    list_rows = []
    for i in range(int(X.shape[0] // 4)):
        Xt = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
        list_rows.append(Xt)

    Xt = np.concatenate(list_rows, axis=1)
    Xt = Xt.transpose([1, 2, 0])

    if Xt.shape[-1] == 1:
        plt.imshow(Xt[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xt)
    plt.axis("off")
    plt.savefig(file_path)
    plt.clf()
    plt.close()
