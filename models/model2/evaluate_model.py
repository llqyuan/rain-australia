from build_features import produce_X_y_from_dataframe

import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas

pandas.options.display.max_columns = 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options for data paths')
    parser.add_argument('--train-data-file', dest='train', 
                        help='Which file to read training data from',
                        default='../data/train/weatherAUS.csv')
    parser.add_argument('--test-data-file', dest='test', 
                        help='Which file to read test data from',
                        default='../data/test/weatherAUS.csv')
    cli_args = parser.parse_args()
    cli_args_dict = vars(cli_args)

    print('Reading training data from {0} and test data from {1}'
          .format(cli_args_dict['train'], cli_args_dict['test']))
    train1 = pandas.read_csv(cli_args_dict['train'])
    test1 = pandas.read_csv(cli_args_dict['test'])

    print('Head (train):')
    print(train1.head())

    print('Head (test):')
    print(test1.head())

    print('Chosen model: random forest classifier with 500 trees. Creating the random forest classifier')
    X_train1, y_train1 = produce_X_y_from_dataframe(train1)
    X_test1, y_test1 = produce_X_y_from_dataframe(test1)

    model = RandomForestClassifier(n_estimators=500)

    print('Fitting model')
    model.fit(X_train1, y_train1)
    print('Using model to predict outcomes for test set')
    y_pred = model.predict(X_test1)
    print('Accuracy: {0}'.format(accuracy_score(y_test1, y_pred)))
    print('Precision: {0}'.format(precision_score(y_test1, y_pred)))
    print('Recall: {0}'.format(recall_score(y_test1, y_pred)))
