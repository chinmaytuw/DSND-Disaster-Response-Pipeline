import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_path, categories_path):
    ''' 
    Script to load data for messages, categories 

    Args:
        messages_path = string with filepath for messages file
        categories_path = string with filepath for categories file

    Returns:
        df: merged dataframe consisting of messages and categories
    '''
    # read messages and categories data
    messages =  pd.read_csv(messages_path)
    categories =  pd.read_csv(categories_path)

    # merge the datasets
    df = pd.merge(messages,categories, on = 'id')
    
    return df


def clean_data(df):
    ''' 
    Script to load data for messages, categories 

    Args:
        df: dataframe to clean

    Returns:
        df:  cleaned dataframe
    '''

    # split categories into separate category columns
    categories = df.categories.str.split(";", expand=True)

    # Creating appropriate column names
    # get the first row
    row = categories.loc[0,:]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split("-").str[0]

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # removing string values to have clean columns labelled either 1 or 0
    for column in categories:
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column], errors='coerce')

    # Dropping the original categories column
    df = df.drop('categories', 1)
    # merge the new columns with original dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df=df.drop_duplicates(keep='first')
    
    return df
    
   


def save_data(df, database_filename):
    ''' 
    Function to save the data in a database file
    Args:
        df: Dataframe to save
        database_filename: name of database file

    Returns:
        None
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_path, categories_path, database_filepath = sys.argv[1:]

        # Steps for Loading data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_path, categories_path))
        df = load_data(messages_path, categories_path)


        # Steps for Cleaning data
        print('Cleaning data...')
        df = clean_data(df)

        
        # Steps for savings data to database
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\n Example: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()