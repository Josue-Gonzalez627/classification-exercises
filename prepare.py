import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def prep_iris(df):
    """
    Simply cleans up the df by dropping some columns that aren't needed
    """
    df = df.drop(columns=['species_id','measurement_id'])
    df = df.rename(columns={"species_name":'species'})
    
    return df

def clean_titanic(df): #edit later to prep_titantic to match the other "preps"
    """
    Function designed to perform the actions above more or less, such as dropping
    the unnecesarry columns and asigning the pclass column as an object rather than
    an INT
    since we want it to be treated as a categorical value. 
    
    I also drop all rows with null values 
        - can also impute them with the mean, mode, back fill, front fill, etc
    """
    #drop unncessary columns
    df = df.drop(columns=['embarked','deck','class'])
    
    #drop the rows with null values
    df = df.dropna()
    
    #made this a string so its categorical
    df.pclass = df.pclass.astype(object)
    
    #filled nas with the mode
    df.embark_town = df.embark_town.fillna('Southampton')
    
    return df

def split_titanic(df):
    '''
    Does the process of splitting the titanic dataframe, setting the train
    size and stratifying it. Does this twice 'first split' and 'second split'.
    '''
    #first split
    train, validate_test = train_test_split(df, #send in intitial df
                train_size=0.60, #size of the train df, and the test will default to 1 - train_size
                random_state = 123, # set any number here for consistency
                stratify=df.survived #need to statify on target variable)
                )
    
    #second split
    validate, test = train_test_split(validate_test, #this is the df that we are splitting now
                test_size=0.50, #set test or train size to 50%
                random_state=123, #any num (doesn't have to be the same as the one above)
                stratify=validate_test.survived #still got to stratify
                )
    return train, validate, test

# Another function? YES PLZ!
def clean_and_split_titanic_data(df):
    '''
    Now this is more peak efficiency! calling the cleaning function together
    with the splitting function to create an ultimate "Prep" function that'll do it
    all for me!! 
    '''
    
    #calling my clean function
    df = clean_titanic(df)
    
    #calling my split function
    train, validate, test = split_titanic(df)
    
    return train, validate, test

def prep_telco(df):
    '''
    Drops some columns that have been brought over from the joins that are not needed. 
    Also replaces empty space values with 0.0 to give it a value.
    '''
    df = df.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id'])
    df.total_charges = df.total_charges.str.replace(' ', '0.0')
    ###Note: should also change the null values to 'None' since they aren't really null
    ### ... they just say none as in they don't have an internet service.
    
    return df

def splitting_data(df, col, seed=123):
    '''
    Just like the splitting Titanic function but it can be used for any df now!
    must provide the df and column. Does not clean it though
    '''

    #first split
    train, validate_test = train_test_split(df,
                     train_size=0.6,
                     random_state=seed,
                     stratify=df[col]
                    )
    
    #second split
    validate, test = train_test_split(validate_test,
                                     train_size=0.5,
                                      random_state=seed,
                                      stratify=validate_test[col]
                        
                                     )
    return train, validate, test


def preprocess_titanic(train_df, val_df, test_df):
    '''
    preprocess_titanic will take our 3 split df I did from the Titanic,
    which are clean also (see documentation on acquire.py and prepare.py)
    
    output:
    encoded, ML-ready versions of our clean data, with the sex and embark_town
    columns fully encoded in the one-hot fashion return: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    we get three df's basically.
    '''
    # with a looping structure:
    # for df in [train_df, val_df, test_df]:
    #    df.drop(blah blah blah)
    #    df['plcass'] = df['pclass'].astype(int)
    train_df = train_df.drop(columns = 'passenger_id')
    train_df['pclass'] = train_df['pclass'].astype(int)
    
    val_df = val_df.drop(columns = 'passenger_id')
    val_df['pclass'] = val_df['pclass'].astype(int)
    
    test_df = test_df.drop(columns = 'passenger_id')
    test_df['pclass'] = test_df['pclass'].astype(int)
    
    encoding_var = ['sex', 'embark_town']
    encoded_dfs = []
    for df in [train_df, val_df, test_df]:
        df_encoded_cats = pd.get_dummies(
            df[encoding_var], drop_first = True).astype(int)
        encoded_dfs.append(pd.concat(
            [df, df_encoded_cats], axis=1).drop(columns = encoding_var))
    return encoded_dfs

def preprocess_telco(train_df, val_df, test_df):
    '''
    preprocess_telco will take in three pandas dataframes
    of our telco data, expected as cleaned versions of this 
    telco data set (see documentation on acquire.py and prepare.py)
    
    output:
    encoded, ML-ready versions of our clean data, with 
    columns sex and embark_town encoded in the one-hot fashion
    return: (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    '''
    # with a looping structure:
    # go through the three dfs, set the index to customer id
    for df in [train_df, val_df, test_df]:
        df = df.set_index('customer_id')
        df['total_charges'] = df['total_charges'].astype(float)
    # initialize an empty list to see what needs to be encoded:
    encoding_vars = []
    # loop through the columns to fill encoded_vars with appropriate
    # datatype field names
    for col in train_df.columns:
        if train_df[col].dtype == 'O':
            encoding_vars.append(col)
    encoding_vars.remove('customer_id')
    # initialize an empty list to hold our encoded dataframes:
    encoded_dfs = []
    for df in [train_df, val_df, test_df]:
        df_encoded_cats = pd.get_dummies(
            df[encoding_vars],
              drop_first=True).astype(int)
        encoded_dfs.append(pd.concat(
            [df,
            df_encoded_cats],
            axis=1).drop(columns=encoding_vars))
    return encoded_dfs

def compute_class_metrics(y_train, y_pred):
    '''
    Will provide the confusion matrix with the pd.crosstab and label in 'counts'.
    Uses 'counts.iloc' to label each point in the matrix as TP, TN, FP, or FN.
    
    Using the variables and formulas for accuracy, TPR, FPR, TNR, FNR, precision, F1 score, 
    support_pos, and support_neg, will print out the metrics!
    '''
    counts = pd.crosstab(y_train, y_pred)
    TP = counts.iloc[1,1]
    TN = counts.iloc[0,0]
    FP = counts.iloc[0,1]
    FN = counts.iloc[1,0]
    
    
    all_ = (TP + TN + FP + FN)

    accuracy = (TP + TN) / all_

    TPR = recall = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TNR = TN / (FP + TN)
    FNR = FN / (FN + TP)

    precision =  TP / (TP + FP)
    f1 =  2 * ((precision * recall) / ( precision + recall))

    support_pos = TP + FN
    support_neg = FP + TN
    
    print(f"Accuracy: {accuracy}\n")
    print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
    print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
    print(f"False Negative Rate/Miss Rate: {FNR}\n")
    print(f"Precision/PPV: {precision}")
    print(f"F1 Score: {f1}\n")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")