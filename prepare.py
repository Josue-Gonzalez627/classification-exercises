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

def clean_titanic(df):
    """
    Function designed to perform the actions above more or less, such as dropping
    the unnecesarry columns and asigning the pclass column as an object rather than
    an INT
    since we want it to be treated as a categorical value.  
    """
    #drop unncessary columns
    df = df.drop(columns=['embarked', 'age','deck', 'class'])
    
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
def prep_titanic_data(df):
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
    must provide the df and column.
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