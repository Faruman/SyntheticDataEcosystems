# Cycle 2 - 02_paramSearch
This file briefly outlines the hyperparameter lists and ranges tested for each algorithm as part of the algorithm evaluation in cycle 2 where we evaluated which algorithm works best for generating synthetic financial transaction data.

## GMM
The hyperparameters tested for the TVAE model are as follows. A detailed explenation of the meaning of these hyperparameter can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
- n_components: [1 - 40]
- covariance_type: ["full", "tied", "diag", "spherical"]
- max_iter: [50 - 300]
- init_params: ["kmeans", "k-means++", "random", "random_from_data"]

## CTGAN
The hyperparameters tested for the TVAE model are as follows. A detailed explenation of the meaning of these hyperparameter can be found [here](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer).
- embedding_dim: [32, 64, 256]
- generator_dim: [(128, 128), (256, 256), (512, 512)]
- discriminator_dim: [(128, 128), (256, 256), (512, 512)]
- generator_lr: [0.00001 - 0.001]
- generator_decay: [0.0 - 0.05]
- discriminator_lr: [0.00001 - 0.001]
- discriminator_decay: [0.0 - 0.05]
- discriminator_steps: [1 - 15]
- epochs: [100 - 1000]
- batch_size: [5000]

## TVAE
The hyperparameters tested for the TVAE model are as follows. A detailed explenation of the meaning of these hyperparameter can be found [here](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/tvaesynthesizer).
- embedding_dim: [32, 64, 256]
- compress_dims: [(128, 128), (256, 256), (512, 512)]
- decompress_dims: [(128, 128), (256, 256), (512, 512)]
- l2scale: [0.00001 - 0.001]
- loss_factor: [1 - 5]
- learning_rate: [0.00001 - 0.001]
- epochs: [100 - 1000]
- batch_size: [5000]

## TIMEGAN
The hyperparameters tested for the TVAE model are as follows. A detailed explenation of the meaning of these hyperparameter can be found [here](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf).
- max_sequence_len: [5 - 40]
- recursive_module: ["gru", "lstm", "rnn"]
- hidden_dim: [16, 32, 64]
- num_layers: [2 - 4]
- metric_iteration: [2 - 8]
- beta1: [0.5 - 0.998]
- gamma: [0.5 - 2.0]
- encoder_loss_weight_s: [0.05 - 0.5]
- encoder_loss_weight_0: [5 - 20]
- generator_loss_weight: [50 - 200]
- generator_steps: [1 - 5]
- learning_rate: [0.00001 - 0.001]
- epochs: [100 - 1000]
- batch_size: [256]