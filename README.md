# Yan's Insight_Project_Framework
Framework for NLP Transfer Learning project as AI Fellow at Insight Data Science.

## Motivation for this project format:
- **src** : This folder includes all the functionalities of this project for data, model and training.
- **tests** : This directory includes inference functionalities.
#- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Put your dataset under the /raw directory as well as the trained models.
#- **build** : Include scripts that automate building of a standalone environment
#- **static** : Any images or content to include in the README or web framework if part of the pipeline

## Setup
git clone https://github.com/yzhanglearning/Insight/tree/master

Create new development branch and switch onto it

branch_name=your_branch_name
git checkout -b $your_branch_name
git push origin $your_branch_name


## Requisites

- fastai
- numpy
- html
- pandas
- pickle
- scikit-learn


## Build Environment
You only need to change your path to your dataset and modify hyperparameters in the two main scripts: main_train.py and main_infer.py for fine-tuning and testing.
- main_train.py
- main_infer.py


## Configs
- Download and ready to use.

## Test/Inference
- In your command line, run python main_infer.py


## Analysis
The main_infer.py script will output the statistical analysis of the model using the following metrics:
- Classification Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, f1 score)
