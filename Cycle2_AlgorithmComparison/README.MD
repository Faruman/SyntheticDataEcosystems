# Cycle 2: Synthetic financial transaction data generation and privacy
In the second cycle of the DSRM project, DP2 was refined by exploring various methods for synthetic data generation, addressing prior expert feedback. A systematic literature review identified algorithms like GANs, Gaussian mixture models, and variational autoencoders, which were tested on a financial fraud detection dataset. Privacy checks confirmed the safety of the generated data. The findings underscored the importance of combining synthetic and real data, leading to a new design principle (DP5) and an updated system architecture.

## Identification of Synthetic Data Generation Algorithms
- [literatureReview.xlsx](LiteratureReview.xlsx) - contains information about the papers which were used to select the algorithms for the later experiments.

## Evaluation of the proposed Synthetic Data Sharing Ecosystem 
### Instructions
To run the code to conduct the exeriments described in the paper please follow the steps below:

1. Use the information provided in [DATA.MD](data.md) to download the necessary data.
2. (Optional) Run [00_assessFeatureImportance.py](00_assessFeatureImportance.py) to assess the importance of the features in the dataset and generate some descriptive statistics.
3. Run [01_preprocessData.py](01_preprocessData.py) to preprocess the data and generate the necessary files for the experiments.
4. In the [02_paramSearch](02_paramSearch) folder run each of the listed scripts to conduct the hyperparameter search for the respective algorithm. Please note you need to be logged in to [wandb.ai](https://wandb.ai) for this to work.
5. Run [03_generateSyntheticData.py](03_generateSyntheticData.py) to generate the synthetic data using the best hyperparameters found in the previous step.
6. Run [04_performanceEvaluation.py](04_performanceEvaluation.py) to evaluate the performance of the synthetic data generated in the previous step.
7. Run [05_privacyEvaluation.py](05_privacyEvaluation.py) to evaluate the privacy of the synthetic data generated in the previous step.
8. Run [06_visualizeEvaluation.py](06_visualizeEvaluation.py) to visualize the results of the performance evaluation and generate the plots displayed in the paper.