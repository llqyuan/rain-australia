import numpy as np
import pandas
from sklearn.model_selection import train_test_split


def _get_months_from_date_strings(dates):
    '''
    Returns a list obtained from dates by extracting the month from 
    the date. Transforms "2008-12-01" -> 12. 
    '''
    months = []
    for date in dates:
        month = int(date.split('-')[1])
        months.append(month)
    return months

def _initial_preprocessing():
    '''
    Transformations on the original columns from the raw data
    '''
    raw_data = pandas.read_csv('../data/raw/weatherAUS.csv')

    data = pandas.DataFrame(data={
        'Month': _get_months_from_date_strings(raw_data['Date']),
        'MinTemp': raw_data['MinTemp'],
        'MaxTemp': raw_data['MaxTemp'],
        'Evaporation': raw_data['Evaporation'],
        'LogEvaporation': np.log1p(raw_data['Evaporation']),
        'Sunshine': raw_data['Sunshine'],
        'WindGustSpeed': raw_data['WindGustSpeed'],
        'WindSpeed9am': raw_data['WindSpeed9am'],
        'WindSpeed3pm': raw_data['WindSpeed3pm'],
        'Humidity9am': raw_data['Humidity9am'],
        'Humidity3pm': raw_data['Humidity3pm'],
        'Pressure9am': raw_data['Pressure9am'],
        'Pressure3pm': raw_data['Pressure3pm'],
        'Cloud9am': raw_data['Cloud9am'],
        'Cloud3pm': raw_data['Cloud3pm'],
        'Temp9am': raw_data['Temp9am'],
        'Temp3pm': raw_data['Temp3pm'],
        'Location': raw_data['Location'],
        'WindGustDir': raw_data['WindGustDir'],
        'WindDir9am': raw_data['WindDir9am'],
        'WindDir3pm': raw_data['WindDir3pm'],
        'RainToday': raw_data['RainToday'],
        'RainTomorrow': raw_data['RainTomorrow']
    })
    data['Month'] = data['Month'].astype('category')
    data['Location'] = data['Location'].astype('category')
    data['WindGustDir'] = data['WindGustDir'].astype('category')
    data['WindDir9am'] = data['WindDir9am'].astype('category')
    data['WindDir3pm'] = data['WindDir3pm'].astype('category')
    data['RainToday'] = data['RainToday'].astype('category')
    data['RainTomorrow'] = data['RainTomorrow'].astype('category')

    return data

def _model_specific_preprocessing(dataframe):
    '''
    Preprocessing specific to Model 1. Drop specified columns, impute, etc
    '''
    # ==============================
    # Drop the specified columns
    # ==============================

    dataframe.drop(
        labels=['Sunshine', 'Evaporation', 'LogEvaporation', 'Cloud9am', 'Cloud3pm'], 
        axis=1, inplace=True)

    # ==============================
    # Handle numerical features: impute with median
    # ==============================

    numerical_features = dataframe[[
        'MinTemp', 'MaxTemp', 'WindGustSpeed', 'WindSpeed9am',
        'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
        'Pressure3pm', 'Temp9am', 'Temp3pm']].copy()
    median_features_train = numerical_features.dropna().median()
    numerical_features.fillna(median_features_train, inplace=True)

    # ==============================
    # Handle categorical features: replace with indicator variables for each category
    # ==============================

    categorical_features = dataframe[[
        'Month', 'Location', 'WindGustDir', 
        'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']].copy()
    categorical_features = pandas.get_dummies(categorical_features)

    # ==============================
    # Merge numerical and categorical dataframes into one unified dataframe. (Can write this to train and test)
    # ==============================

    dataframe_final = pandas.concat([numerical_features, categorical_features], axis=1, join='inner')
    return dataframe_final

def produce_dataframe():
    '''
    Produces the dataframe that will be used for Model 1
    '''
    df = _initial_preprocessing()
    return _model_specific_preprocessing(df)

def produce_X_from_dataframe(dataframe):
    '''
    Produces the design matrix X (to be used in sklearn fitting) from the dataframe
    '''
    pass

if __name__ == '__main__':
    print('Calling as main -- will produce train and test dataframes and write them to train and test.')
    print('Handling preprocessing...')
    # intend to further subdivide train into training set and cross-validation set in k-fold cross validation
    data = produce_dataframe()
    train, test = train_test_split(data, test_size=0.2)
    train.to_csv('../data/train/weatherAUS.csv', mode='w')
    print('Wrote to data/train/weatherAUS.csv')
    test.to_csv('../data/test/weatherAUS.csv', mode='w')
    print('Wrote to data/test/weatherAUS.csv')
    