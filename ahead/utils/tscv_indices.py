import numpy as np


def get_tscv_indices(
    n=25, p=0.8, initial_window=5, horizon=3, fixed_window=False
):
    """Generate indices to split data into training and test set.

    Parameters
    ----------
    n : int, input series length.

    p : float, proportion of data in the training set

    initial_window : int, initial number of consecutive values in each
                     training set sample

    horizon : int, number of consecutive values in test set sample

    fixed_window : boolean, fixed window or increasing window

    """

    # Initialization of indices -----

    indices = np.arange(n)
    train_test_indices = []
    hold_out_indices = []

    # train index
    min_index_train = 0
    max_index_train = initial_window

    # test index
    min_index_test = max_index_train
    max_index_test = initial_window + horizon

    # Main loop -----

    if p == 1:

        if fixed_window == True:

            while max_index_test <= n:

                train_test_indices.append(
                    {
                        "train": indices[min_index_train:max_index_train],
                        "test": indices[min_index_test:max_index_test],
                    }
                )

                min_index_train += 1
                min_index_test += 1
                max_index_train += 1
                max_index_test += 1

        else:  # fixed_window == False

            while max_index_test <= n:

                train_test_indices.append(
                    {
                        "train": indices[min_index_train:max_index_train],
                        "test": indices[min_index_test:max_index_test],
                    }
                )

                max_index_train += 1
                min_index_test += 1
                max_index_test += 1

        return train_test_indices

    # else if p < 1

    if fixed_window == True:

        while max_index_test <= n:

            if max_index_test <= int(p * n):

                train_test_indices.append(
                    {
                        "train": indices[min_index_train:max_index_train],
                        "test": indices[min_index_test:max_index_test],
                    }
                )

            else:

                hold_out_indices.append(
                    {
                        "train": indices[min_index_train:max_index_train],
                        "test": indices[min_index_test:max_index_test],
                    }
                )

            min_index_train += 1
            min_index_test += 1
            max_index_train += 1
            max_index_test += 1

    else:  # fixed_window == False

        while max_index_test <= n:

            if max_index_test <= int(p * n):

                train_test_indices.append(
                    {
                        "train": indices[min_index_train:max_index_train],
                        "test": indices[min_index_test:max_index_test],
                    }
                )

            else:

                hold_out_indices.append(
                    {
                        "train": indices[min_index_train:max_index_train],
                        "test": indices[min_index_test:max_index_test],
                    }
                )

            max_index_train += 1
            min_index_test += 1
            max_index_test += 1

    return train_test_indices, hold_out_indices
