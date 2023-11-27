import pandas as pd
import numpy as np

import env
import os

def check_file_exists(filename, query, url):
    '''
    As the name implies, this here is to see if the file(csv) we are calling/using exists AND what to do 
    if it doesn't exist. 
    If it doesn't exist, it will read the query using the url (env info) and makes it into a csv file!
    '''
    
    if os.path.exists(filename):
        print('this file exists, reading csv')
        df = pd.read_csv(filename, index_col=0)
    else:
        print('this file doesnt exist, read from sql, and export to csv')
        df = pd.read_sql(query, url)
        df.to_csv(filename)
        
    return df


def get_iris_data():
    '''
    will retrieve the iris dataframe. The query will be the MySQL syntax to help organize (join) the 
    table. the check file function is attached to direct this function on what to do if the iris.csv 
    exists or not, and will run the function once it is found or created then read.
    '''
    url = env.get_db_url('iris_db')
    query = '''
    select * 
    from measurements
        join species
            using (species_id)
    '''
    
    filename = 'iris.csv'
    
    #call the check_file_exists fuction 
    df = check_file_exists(filename, query, url)
    return df

def get_titanic_data():
    '''
    Retrieves the titanic dataframe. The MySQL query will return all columns from the passengers table.
    The check file function will assure the titanic file exists and what to do if it doesn't.
    '''
    url = env.get_db_url('titanic_db')
    query = 'select * from passengers'
    
    filename = 'titanic.csv'
    
    #call the check_file_exists fuction 
    df = check_file_exists(filename, query, url)
    return df


def get_telco_data():
    '''
    Retrieves the telco dataframe. The MySQL query will return all columns from the customers table,
    with the three additional columns because of the joins using their ids.
    The check file function will assure the telco file exists and what to do if it doesn't.
    '''
    url = env.get_db_url('telco_churn')
    query = '''
    select *
    from customers
        join contract_types
            using (contract_type_id)
        join internet_service_types
            using (internet_service_type_id)
        join payment_types
            using (payment_type_id)
    '''
    
    filename = 'telco_churn.csv'

    #call the check_file_exists fuction 
    df = check_file_exists(filename, query, url)
    return df