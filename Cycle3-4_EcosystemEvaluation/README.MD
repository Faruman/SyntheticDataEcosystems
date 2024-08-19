# Cycle 3: Local Synthetic data recombination and usage / Cycle 4: Network Effects of Financial Data Sharing

## Evaluation of the proposed Synthetic Data Sharing Ecosystem on additional datasets and analysis of network effects
### Instructions
To run the code to conduct the exeriments described in the paper please follow the steps below:
- Use the information provided in [DATA.MD](DATA.MD) to download the necessary data. 
- From the folder [01_preprocessData](01_preprocessData) run the scripts for the different datasets to preprocess the data and generate the necessary files for the experiments.
- In the [02_paramSearch](02_paramSearch) folder run each of the listed scripts to conduct the hyperparameter search for the respective algorithm. Please note you need to be logged in to [wandb.ai](https://wandb.ai) for this to work.
- Run [03_generateSyntheticData.py](03_generateSyntheticData.py) to generate the synthetic data using the best hyperparameters found in the previous step.
- In the folder [04_performanceScoring](04_performanceScoring) contains the scripts to benchmark performance. 
  - [baselineScoring.py](04_performanceScoring/baselineScoring.py) is used to score the performance on the real data only.
  - [partlySynthTransactionScoring.py](04_performanceScoring/partlySynthTransactionScoring.py) is used to score the performance of the synthetic data models mixing synthetic and real data.
  - [synthTransactionScoring.py](04_performanceScoring/synthTransactionScoring.py) is used to score the performance of models using purely synthetic data.)
- Form the folder [05_analysis](05_analysis) run the scripts to analyze and visualize the results of the performance evaluation and generate the plots displayed in the paper.