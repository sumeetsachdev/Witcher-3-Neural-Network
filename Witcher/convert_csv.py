import numpy as np
import pandas as pd

train_data = np.load('training_data.npy', allow_pickle=True)
directions = ['left', 'straight', 'right']

labels = pd.DataFrame({'id':[f'witcher_{i}' for i in range(len(train_data))], 
         'label':[directions[np.argmax(train_data[i][1])] for i in range(len(train_data))]})
label_csv = 'labels_3.csv'
n = len(list(open(label_csv))) - 1 # header is not counted (-1)
val_idxs = get_cv_idxs(n) # random 20% data for validation set
val_idxs.shape, get_cv_idxs(n).shape
label_df = pd.read_csv(label_csv)
