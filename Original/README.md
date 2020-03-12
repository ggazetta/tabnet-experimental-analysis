# Codebase for "TabNet: Attentive Interpretable Tabular Learning"

Authors: Sercan O. Arik, Tomas Pfister

Paper: https://arxiv.org/abs/1908.07442

Modifications done by: Gabriel Gazetta de Araujo

### Running the code:

The first step is NOT necessary since the dataset has been uploaded to the repository:
First, run `python download_prepare_dataset.py` to download and prepare the Wine dataset.
This command creates train.csv, val.csv and test.csv files under the "data/" directory.

To run the pipeline for training and evaluation, simply use `python experiment.py`.


To modify the experiment to other tabular datasets:
- Substitute the train.csv, val.csv, and test.csv files under "data/" directory,
- Modify the data_helper function with the numerical and categorical features of the new dataset,
- Reoptimize the TabNet hyperparameters for the new dataset.
