from skopt import dump
from skopt.callbacks import DeltaXStopper
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, forest_minimize
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import os



# define the space of hyperparameters to search
SPACE = [
    Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
    Integer(1, 30, name='max_depth'),
    Integer(100, 700, name='n_estimators'),
    Integer(0, 1, name='n_components'),
    Integer(100, 200, name='max_iter'),
    Integer(2, 100, name='num_leaves'),
    Integer(10, 1000, name='min_data_in_leaf'),
    Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
    Real(0.1, 1.0, name='subsample', prior='uniform'),
    Real(1e-6, 100.0, 'log-uniform', name='C'),
    Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'),
    # Categorical(['svd', 'lsqr', 'eigen'], name='solver'),
    Integer(1, 5, name='degree'),
    Real(1e-6, 100.0, 'log-uniform', name='gamma'),
    Real(0, 1, name='reg_param')
]


def evaluate_hyperparameter(model, X_train, X_val, y_train, y_val):

    classifier = model["classifier"] if type(model) == Pipeline else model

    filter_space = [p for p in SPACE if p.name in classifier.get_params().keys()]

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
    directory = 'artifacts'
    file_name = '%s_results.pkl' % type(classifier).__name__

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file_name), 'wb') as file:
        dump(results, file, store_objective=False)

    # summarizing finding:
    dictionary = dict(zip([p.name for p in filter_space], results.x))
    print('Accuracy achieved with the validation set: %.3f' % results.fun)
    print('Best Parameters: %s' % dictionary)
    return dictionary
