# 02_paramSearch
This file briefly outlines the hyperparameter lists and ranges tested for each algorithm.

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