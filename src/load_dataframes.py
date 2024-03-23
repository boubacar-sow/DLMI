import pandas as pd

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