import numpy as np


def create_sequences(data, window_size, feature_cols):
    """
    Create sequences from input data.

    Parameters:
    - data: 2D array, shape (n_samples, n_features)
      Scaled input data.
    - window_size: int
      Size of the window (number of past time steps to consider in each sequence).
    - feature_cols: list
      List of indices representing the feature columns.

    Returns:
    - sequences: 3D array, shape (n_samples - window_size + 1, window_size, n_features)
      Formatted sequences.
    """

    sequences = []
    n_samples = len(data)

    for i in range(window_size, n_samples + 1):
        sequence = data[i - window_size:i, feature_cols]
        sequences.append(sequence)

    return np.array(sequences)
