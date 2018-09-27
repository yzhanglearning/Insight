## fast.ai implementation of NLP transfer learning


from fastai.text import *
import html
import os
import pandas as pd
import pickle
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
confusion_matrix
from sklearn.model_selection import train_test_split
from time import time



from plot import *
