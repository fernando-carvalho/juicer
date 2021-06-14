from tests.scikit_learn import util
from tests.scikit_learn.util import get_label_data, get_X_train_data
from juicer.scikit_learn.classification_operation import \
    PerceptronClassifierOperation, PerceptronClassifierModelOperation
from sklearn.linear_model import Perceptron
import pytest
import pandas as pd
import numpy as np


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
                       'multiplicity': {'train input data': 0},
                       'label': [get_columns[0]]},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }


# PerceptronClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #

@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),
    ({'alpha': 0.0002}, {'alpha': 0.0002}),
    ({'tol': 0.002}, {'tol': 0.002}),
    ({"seed": 1, "shuffle": 1}, {"random_state": 1, "shuffle": True}),
    ({"penalty": "l1"}, {"penalty": "l1"}),
    ({"max_iter": 1002}, {"max_iter": 1002}),
    ({'fit_intercept': 2}, {'fit_intercept': 2}),
    ({'eta0': 2.0}, {'eta0': 2.0}),
    ({'n_jobs': -1}, {'n_jobs': -1}),
    ({"seed": 1, 'early_stopping': 1},
     {"random_state": 1, 'early_stopping': True}),
    ({'validation_fraction': 0.2}, {'validation_fraction': 0.2}),
    ({'n_iter_no_change': 6}, {'n_iter_no_change': 6}),
    ({'class_weight': "'balanced'"}, {'class_weight': "balanced"})
], ids=["default_params", "alpha_param", "tol_param", "seed_and_shuffle_params",
        "penalty_param", "max_iter_param", "fit_intercept_param", "eta0_param",
        "n_jobs_param", "early_stopping_param", "validation_fraction_param",
        "n_iter_no_change_param", "class_Weight_param"])
def test_perceptron_classifier_params_success(get_arguments, get_df, get_columns,
                                              algorithm_par, operation_par):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = PerceptronClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = Perceptron(tol=0.001, alpha=0.0001,
                         max_iter=1000, shuffle=False, random_state=None,
                         penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         validation_fraction=0.1,
                         n_iter_no_change=5,
                         class_weight=None, warm_start=False)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_perceptron_classifier_prediction_param_success(get_df, get_columns,
                                                        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = PerceptronClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df.astype(np.int64())})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_perceptron_classifier_no_code_success(selector, drop, get_arguments):
    arguments = get_arguments
    arguments[selector].pop(drop)
    util.add_minimum_ml_args(arguments)
    instance = PerceptronClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
@pytest.mark.parametrize('par', ['max_iter', 'alpha'])
def test_perceptron_classifier_invalid_params_fail(par,
                                                   get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        PerceptronClassifierModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {PerceptronClassifierOperation}" in str(val_err.value)


def test_perceptron_classifier_invalid_validation_fraction_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'validation_fraction': -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        PerceptronClassifierModelOperation(**arguments)
    assert f"Parameter 'validation_fraction' must be 0 <= x =< 1 for task " \
           f"{PerceptronClassifierOperation}" in \
           str(val_err.value)
