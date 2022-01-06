from build_features import produce_X_y_from_dataframe

import argparse
from numpy import mean, std
from sklearn.model_selection import cross_val_score
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

    print('Reading training data from {0}'.format(cli_args_dict['train']))
    train1 = pandas.read_csv(cli_args_dict['train'])

    print('Head:')
    print(train1.head())

    print('About to try random forest models with 10, 50, 100, 500, 1000 trees')
    X_train1, y_train1 = produce_X_y_from_dataframe(train1)
    for number_of_trees in (10, 50, 100, 500, 1000):
        model = RandomForestClassifier(n_estimators=number_of_trees)
        accuracy_scores = cross_val_score(model, X_train1, y_train1, scoring='accuracy', cv=5, n_jobs=-1)
        precision_scores = cross_val_score(model, X_train1, y_train1, scoring='precision', cv=5, n_jobs=-1)
        recall_scores = cross_val_score(model, X_train1, y_train1, scoring='recall', cv=5, n_jobs=-1)
        print('{0} mean (accuracy, precision, recall): {1}, {2}, {3}'.format(
            number_of_trees, 
            mean(accuracy_scores), 
            mean(precision_scores), 
            mean(recall_scores)))
        print('{0} standard deviation (accuracy, precision, recall): {1}, {2}, {3}'.format(
            number_of_trees, 
            std(accuracy_scores), 
            std(precision_scores),
            std(recall_scores)))