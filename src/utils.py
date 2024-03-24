import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import torch
from src.dataloader import LymphBags

def train_val_dataset(df, bags_dir, train_transforms=None, val_transforms=None, val_split=0.25):
    df_copy = df.copy()
    # if age > 65, set label to 1
    df_copy['AGE'] = df_copy['AGE'].apply(lambda x: int(x>65))
    train_idx, val_idx = train_test_split(list(range(len(df))), test_size=val_split, stratify = df_copy[['LABEL', 'GENDER', 'AGE', 'LYMPH_COUNT']])
    train_set = LymphBags(bags_dir, df, indices = train_idx, transforms=train_transforms)
    val_set = LymphBags(bags_dir, df, indices = val_idx, transforms=val_transforms)
    return train_set, val_set

def balanced_accuracy(outputs, labels):
    outputs = outputs.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    return balanced_accuracy_score(labels, (outputs >= 0.5).astype(int))

def load_dataframes(path_train, path_test):
    df_train = pd.read_csv(path_train)
    df_train['GENDER'] = df_train.GENDER.apply(lambda x: int(x=='F'))
    df_train['DOB'] = df_train['DOB'].apply(lambda x: x.replace("-", "/"))
    df_train['AGE'] = df_train['DOB'].apply(lambda x: 2020-int(x.split("/")[-1]))
    
    df_test = pd.read_csv(path_test)
    df_test['GENDER'] = df_test.GENDER.apply(lambda x: int(x=='F'))
    df_test['DOB'] = df_test['DOB'].apply(lambda x: x.replace("-", "/"))
    df_test['AGE'] = df_test['DOB'].apply(lambda x: 2020-int(x.split("/")[-1]))

    return df_train, df_test

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
