class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    context_size = 0
    num_mfcc_features = 13
    num_final_features = num_mfcc_features * (2 * context_size + 1)

    batch_size = 16
    num_classes = 12 # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12
    num_hidden = 128

    num_epochs = 50
    l2_lambda = 0.0000001
    lr = 1e-3