import argparse
import numpy as np
import rich
import argparse
from ga_feature_selection.genetic_algorithm import GA_FeatureSelector
from sklearn import datasets
from sklearn.datasets import make_classification, make_regression
from sklearn import linear_model


def main(args):
    """Loading X(features), y(targets) from datasets"""
    data = datasets.load_wine()
    X, y = data['data'], data['targets']
    LogisticRegression = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    Genetic_Algorithm = GA_FeatureSelector(model=LogisticRegression, args=args, seed=args.seed)
    
    """Making train and test set"""
    X_train, X_test, y_train, y_test = Genetic_Algorithm.data_prepare(X, y)
    Genetic_Algorithm.run(X_train, X_test, y_train, y_test)

    """Show the result"""
    table, summary_table = Genetic_Algorithm.summary_table()
    rich.print(table)
    rich.print(summary_table)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--normalization", default=False, type=bool)
    parser.add_argument("--n-generation", default=10, type=int, help="Determines the maximum number of generations to be carry out.")
    parser.add_argument("--n-population", default=100, type=int, help="Determines the size of the population (number of chromosomes).")
    parser.add_argument("--crossover-rate", default=0.7, type=float, help="Defines the crossing probability. It must be a value between 0.0 and 1.0.")
    parser.add_argument("--mutation-rate", default=0.1, type=float, help="Defines the mutation probability. It must be a value between 0.0 and 1.0.")
    parser.add_argument("--tournament-k", default=2, type=int, help="Defines the size of the tournament carried out in the selection process. Number of chromosomes facing each other in each tournament.")
    parser.add_argument("--n-jobs", default=1, choices=[1, -1], type=int, help="Number of cores to run in parallel. By default a single-core is used.")
    parser.add_argument("--initial-best-chromosome", default=None, type=np.ndarray, help="A one-dimensional binary matrix of size equal to the number of features (M). Defines the best chromosome (subset of features) in the initial population.")
    parser.add_argument("--verbose", default=0, type=int, help="Control the output verbosity level. It must be an integer value between 0 and 2.")
    parser.add_argument("--c-metric", default='accuracy', choices=['accuracy', 'f1_score', 'roc_auc_socre'], type=str)
    parser.add_argument("--r-metric", default='rmse', choices=['rmse', 'corr', 'mape', 'mae'], type=str)
    
    args = parser.parse_args()
    
    main(args)