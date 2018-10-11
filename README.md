# Yan's Insight Project as AI Fellow at Insight Data Science: NLP Transfer Learning

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
