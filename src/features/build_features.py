import numpy as np
import pandas
from sklearn.model_selection import train_test_split


def get_months_from_date_strings(dates):
    '''
    Returns a list obtained from dates by extracting the month from 
    the date. Transforms "2008-12-01" -> 12. 
    '''
    months = []
    for date in dates:
        month = int(date.split('-')[1])
        months.append(month)
    return months

def produce_dataframe():
    raw_data = pandas.read_csv('../data/raw/weatherAUS.csv')

    data = pandas.DataFrame(data={
        'Month': get_months_from_date_strings(raw_data['Date']),
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

def produce_X_from_dataframe(dataframe):
    pass

if __name__=='__main__':
    print('Calling as main -- will produce train and test dataframes and write them to train and test.')
    # intend to further subdivide train into training set and cross-validation set in k-fold cross validation
    data = produce_dataframe()
    train, test = train_test_split(data)
    train.to_csv('../data/train/weatherAUS.csv', mode='w')
    print('Wrote to data/train/weatherAUS.csv')
    test.to_csv('../data/test/weatherAUS.csv', mode='w')
    print('Wrote to data/test/weatherAUS.csv')
    