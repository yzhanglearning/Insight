################ data loading/ingestion ############
import os
import pandas as pd
from os.path import join as Path ### simple joining path


# function to load the original data files
def DataLoad(path, file_name):
    data = []
    with open(Path(path, file_name), 'r', encoding='utf8') as file:
        for line in file:
            data.append(file.readline())
    return data



# function to transform the text data into pandas data frame and save content and label as .csv files
def DataLabelFormat(path, data, data_name_str):
    BOS = 'xbos' # beginning-of-sentence tag
    FLD = 'xfld' # data field SyntaxWarning

    class_path = Path(path, 'amazon_class')
    os.makedirs(class_path, exist_ok=True)

    lm_path = Path(path, 'amazon_lm')
    os.makedirs(lm_path, exist_ok=True)

    # each item is '__label__1/2' and then the review so we split to get texts and labels_
    data_texts, data_labels = [text[10:] for text in data], [text[:10] for text in data]
    data_labels = [0 if label == '__label__1' else 1 for label in data_labels]

    # following fast.ai recommendations we put our data in pandas dataframes
    col_names = ['labels', 'texts']

    df_data = pd.DataFrame({'texts': data_texts, 'labels': data_labels}, columns=col_names)

    df_data.head(10)

    df_data.to_csv(Path(class_path, data_name_str + '.csv'), header=False, index=False)

    classes = ['neg', 'pos']
    open(Path(class_path, 'classes.txt'), 'w').writelines(f'{o}\n' for o in classes)


path = 'data/raw/Amazon'
train = DataLoad(path, 'train.ft.txt')
test = DataLoad(path, 'test.ft.txt')

print(f'The train data contains {len(train)} examples')
print(f'The test data contains {len(test)} examples')

train = train[:1000]
test = test[1:1000]


DataLabelFormat(path, train, 'train')
DataLabelFormat(path, test, 'test')
