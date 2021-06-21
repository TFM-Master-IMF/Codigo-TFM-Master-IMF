from skopt import dump, load
from skopt.callbacks import DeltaXStopper
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, forest_minimize
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from os.path import dirname, abspath
import os
import pickle

# define the space of hyperparameters to search
SPACE = {
    "LogisticRegression": [
        # Categorical(['none', 'l2'], name='penalty'),
        Real(1e-6, 100.0, 'log-uniform', name='tol'),
        Real(1e-6, 100.0, 'log-uniform', name='C'),
        Categorical([True, False], name='fit_intercept'),
        # Real(-5, 5, 'log-uniform', name='intercept_scaling'),
        Categorical(['newton-cg', 'lbfgs', 'sag', 'saga'], name='solver'),
        Categorical(['auto', 'ovr', 'multinomial'], name='multi_class'),
        Integer(100, 200, name='max_iter'),
        Integer(0, 50, name='random_state'),
        # Integer(0, 100, name='verbose')
    ],
    "RandomForestClassifier": [
        Integer(100, 700, name='n_estimators'),
        Categorical(['gini', 'entropy'], name='criterion'),
        Integer(1, 30, name='max_depth'),
        Categorical(['auto', 'sqrt', 'log2'], name='max_features'),
        # Categorical([True, False], name='bootstrap'),
        Categorical([True, False], name='oob_score'),
        Integer(0, 100, name='random_state'),
    ],
    "MLPClassifier": [
        Categorical(['identity', 'logistic', 'tanh', 'relu'], name='activation'),
        Categorical(['lbfgs', 'sgd', 'adam'], name='solver'),
        Categorical(['constant', 'invscaling', 'adaptive'], name='learning_rate'),
        Integer(100, 500, name='max_iter'),
        Real(1e-6, 100.0, 'log-uniform', name='tol'),
        Real(1e-8, 100.0, 'log-uniform', name='epsilon'),
    ],
    "XGBClassifier": [
        Integer(1, 18, name='max_depth'),
        Integer(1, 9, name='gamma'),
        Integer(40, 180, name='reg_alpha'),
        Real(0, 1.0, name='reg_lambda'),
        Real(0.5, 1.0, name='colsample_bytree'),
        Integer(0, 10, name='min_child_weight'),
        Integer(0, 180, name='n_estimators'),
    ],
    "QuadraticDiscriminantAnalysis": [
        Real(0, 1, name='reg_param')
    ],
    "SVC": [
        Real(0, 100, name='C'),
        Real(0.001, 1, name='gamma'),
        Categorical(['rbf', 'poly', 'sigmoid'], name='kernel')
    ],
    'LinearDiscriminantAnalysis': [
        Categorical(['lsqr', 'eigen'], name='solver'),
        Real(0, 1, name='shrinkage'),
    ]
}


def evaluate_hyperparameter(model, X_train, X_val, y_train, y_val):
    classifier = model["classifier"] if type(model) == Pipeline else model

    filter_space = SPACE[type(classifier).__name__]

    # define the function used to evaluate a given configuration
    @use_named_args(filter_space)
    def evaluate_model(**params):
        # configure the model with specific hyperparameters
        if type(model) == Pipeline:
            model["classifier"].set_params(**params)
        else:
            model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)

    # perform optimization
    results = forest_minimize(evaluate_model, filter_space, callback=DeltaXStopper(1e-2))

    # save the results
    path = dirname(abspath(__file__))
    directory = '/artifacts'
    result_file_name = '%s_results.pkl' % type(classifier).__name__
    best_params_file_name = '%s_best_params.txt' % type(classifier).__name__

    if not os.path.exists(path + directory):
        os.makedirs(path + directory)

    params = dict(zip([p.name for p in filter_space], results.x))
    #previous_result = os.path.join(os.getcwd(), directory, result_file_name)
    #if not os.path.isfile(previous_result) or load(previous_result).fun < results.fun:
    with open(os.path.join(path + directory, result_file_name), 'wb') as file:
        dump(results, file, store_objective=False)
        with open(os.path.join(path + directory, best_params_file_name), 'wb') as params_file:
            pickle.dump(params, params_file)

    # summarizing finding:
    print('Accuracy achieved with the validation set: %.3f' % results.fun)
    print('Best Parameters: %s' % params)
    return params