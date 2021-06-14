import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from juicer.scikit_learn.classification_operation import \
    LogisticRegressionOperation, LogisticRegressionModelOperation
from tests.scikit_learn import util
from tests.scikit_learn.util import get_label_data, get_X_train_data


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

@pytest.fixture
def get_columns():
    return ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']


@pytest.fixture
def get_df(get_columns):
    return pd.DataFrame(util.iris(get_columns))


@pytest.fixture
def get_arguments(get_columns):
    return {
        'parameters': {'features': get_columns,
                       'label': [get_columns[0]],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }


# LogisticRegression
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({"random_state": 1}, {"random_state": 1}),

    ({"random_state": 1, "penalty": "l1"}, {"random_state": 1, "penalty": "l1"}),

    ({"random_state": 1, 'solver': 'newton-cg'},
     {"random_state": 1, 'solver': 'newton-cg'}),

    ({"random_state": 1, "tol": 1e-5}, {"random_state": 1, "tol": 1e-5}),

    ({"random_state": 1, 'solver': 'lbfgs'},
     {"random_state": 1, 'solver': 'lbfgs'}),

    ({"random_state": 1, 'regularization': 0.9}, {"random_state": 1, 'C': 0.9}),

    ({"random_state": 1, 'solver': 'sag'}, {"random_state": 1, 'solver': 'sag'}),

    ({"random_state": 1, 'max_iter': 102}, {"random_state": 1, 'max_iter': 102}),

    ({"random_state": 1, 'solver': 'saga'},
     {"random_state": 1, 'solver': 'saga'}),

    ({"random_state": 1, 'dual': True}, {"random_state": 1, 'dual': True}),

    ({"random_state": 1, 'fit_intercept': False},
     {"random_state": 1, 'fit_intercept': False}),

    ({"random_state": 1, 'intercept_scaling': 1.1},
     {"random_state": 1, 'intercept_scaling': 1.1}),

    ({"random_state": 1, 'multi_class': 'auto'},
     {"random_state": 1, 'multi_class': 'auto'}),

    ({"random_state": 1, 'n_jobs': 10}, {"random_state": 1, 'n_jobs': 10}),

    ({"random_state": 1, 'l1_ratio': 0.5}, {"random_state": 1, 'l1_ratio': 0.5})

], ids=["default_params", "penalty_param", "solver_param_newton-cg",
        "tol_param", "solver_param_lbfgs", "regularization_param",
        "solver_param_sag", "max_iter_param", "solver_param_saga", "dual_param",
        "fit_intercept_param", 'intercept_scaling_param', "multi_class_param",
        "n_jobs_param", "l1_ratio_param"])
def test_logistic_regression_solver_params_success(get_df,
                                                   get_arguments,
                                                   get_columns,
                                                   algorithm_par,
                                                   operation_par):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments
    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0,
                                 max_iter=100, solver='liblinear',
                                 random_state=1,
                                 penalty='l2', dual=False,
                                 fit_intercept=True,
                                 intercept_scaling=1.0,
                                 multi_class='ovr', n_jobs=None,
                                 l1_ratio=None)
    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_logistic_regression_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_logistic_regression_missing_label_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop('label')
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameters 'label' must be informed for task" \
           f" {LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_missing_features_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop('features')
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameters 'features' must be informed for task" \
           f" {LogisticRegressionOperation}" in str(
        val_err.value)


@pytest.mark.parametrize('par',
                         ["tol", "intercept_scaling", "n_jobs", "regularization",
                          "max_iter"])
def test_logistic_regression_invalid_params_fail(get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_invalid_newton_cg_dual_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({"solver": "newton-cg",
                                    "dual": True})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"For 'newton-cg' solver supports only dual=False for task " \
           f"{LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_invalid_multi_class_multinomial_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'multi_class': "multinomial"})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert 'Parameter "solver" does not support multi_class="multinomial"' in \
           str(val_err.value)


def test_logistic_regression_elastic_net_penalty_invalid_l1_ratio_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update(
        {'solver': 'saga', 'l1_ratio': -1, 'penalty': 'elasticnet'})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameter 'l1_ratio' must be 0 <= x <= 1 for task" \
           f" {LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_liblinear_invalid_l1_ratio_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'l1_ratio': -1, 'penalty': 'elasticnet'})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"For 'liblinear' solver, the penalty type must be in ['l1', 'l2']" \
           f" for task {LogisticRegressionOperation}" in str(val_err.value)
