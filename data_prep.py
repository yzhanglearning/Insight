from utils import *


###############################


path = 'data/Amazon'
train = []
with open(os.path.join(path, 'train.ft.txt'), 'r', encoding='utf8') as file:
    for line in file:
        train.append(file.readline())

test = []
with open(os.path.join(path, 'test.ft.txt'), 'r', encoding='utf8') as file:
    for line in file:
        test.append(file.readline())

print(f'The train data contains {len(train)} examples')
print(f'The test data contains {len(test)} examples')


train = train[:1000]
test = test[1:1000]



BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

PATH=Path(path)

CLAS_PATH=PATH/'amazon_class'
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH=PATH/'amazon_lm'
LM_PATH.mkdir(exist_ok=True)


# Each item is '__label__1/2' and then the review so we split to get texts and labels
trn_texts,trn_labels = [text[10:] for text in train], [text[:10] for text in train]
trn_labels = [0 if label == '__label__1' else 1 for label in trn_labels]
val_texts,val_labels = [text[10:] for text in test], [text[:10] for text in test]
val_labels = [0 if label == '__label__1' else 1 for label in val_labels]


# Following fast.ai recommendations we put our data in pandas dataframes
col_names = ['labels','text']

df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)


df_trn.head(10)

df_trn.to_csv(CLAS_PATH/'train.csv', header=False, index=False)
df_val.to_csv(CLAS_PATH/'test.csv', header=False, index=False)

CLASSES = ['neg', 'pos']
(CLAS_PATH/'classes.txt').open('w').writelines(f'{o}\n' for o in CLASSES)
