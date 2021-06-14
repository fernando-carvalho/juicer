import numpy as np
import pytest
from sklearn.linear_model import SGDRegressor

from juicer.scikit_learn.regression_operation import SGDRegressorModelOperation, \
    SGDRegressorOperation
from tests.scikit_learn import util
from tests.scikit_learn.util import get_X_train_data, get_label_data


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


@pytest.fixture
def get_columns():
    return ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']


@pytest.fixture
def get_df(get_columns):
    return util.iris(get_columns)


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


# SGDRegressor:
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),
    ({'alpha': 0.1}, {'alpha': 0.1}),
    ({'l1_ratio': 0.3}, {'l1_ratio': 0.3}),
    ({'max_iter': 500}, {'max_iter': 500}),
    ({'tol': 0.1}, {'tol': 0.1}),
    ({'random_state': 2002}, {'random_state': 2002}),
    ({'power_t': 0.1}, {'power_t': 0.1}),

    ({'loss': 'epsilon_insensitive', 'epsilon': 0.2},
     {'loss': 'epsilon_insensitive', 'epsilon': 0.2}),

    ({'n_iter_no_change': 2}, {'n_iter_no_change': 2}),

    ({'penalty': 'l1'}, {'penalty': 'l1'}),

    ({'fit_intercept': 3}, {'fit_intercept': 3}),

    ({'average': 0}, {'average': 0}),

    ({'shuffle': 0}, {'shuffle': False}),

    ({'eta0': 0.02, 'learning_rate': 'adaptive'},
     {'eta0': 0.02, 'learning_rate': 'adaptive'}),

    ({'early_stopping': 1, 'validation_fraction': 0.3},
     {'early_stopping': True, 'validation_fraction': 0.3})

], ids=["default_params", "alpha_param", 'l1_ratio_param', 'max_iter_param',
        'tol_param', 'random_state_param', 'power_t_param',
        "loss_and_epsilon_param", 'n_iter_no_change_param', 'penalty_param',
        'fit_intercept_param', 'average_param', 'shuffle_param',
        'eta_0_and_learning_rate_param',
        'early_stopping_and_validation_fraction_params'])
def test_sgd_regressor_params_success(get_arguments, get_df, get_columns,
                                      operation_par, algorithm_par):
    df = get_df.copy()
    test_df = get_df.copy()
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = SGDRegressorModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])

    model_1 = SGDRegressor(loss='squared_loss', power_t=0.5,
                           tol=0.001, early_stopping=False,
                           n_iter_no_change=5, penalty='l2',
                           fit_intercept=1, average=1,
                           learning_rate='invscaling',
                           shuffle=True, alpha=0.0001,
                           l1_ratio=0.15, max_iter=1000,
                           random_state=None, eta0=0.01)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert np.allclose(result['out'], test_df, atol=1)
    assert str(result['algorithm']) == str(model_1)


def test_sgd_regressor_prediction_param_success(get_df, get_arguments):
    df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = SGDRegressorModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),
    ('named_inputs', 'train input data')
], ids=["missing_output", "missing_input"])
def test_sgd_regressor_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = SGDRegressorModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
@pytest.mark.parametrize('par', ['alpha', 'max_iter',
                                 'n_iter_no_change', 'eta0'])
def test_sgd_regressor_multiple_invalid_params_fail(get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        SGDRegressorModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {SGDRegressorOperation}" in str(val_err.value)


@pytest.mark.parametrize('par', ['l1_ratio', 'validation_fraction'])
def test_sgd_regressor_multiple_invalid_params_2_fail(get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        SGDRegressorModelOperation(**arguments)
    assert f"Parameter '{par}' must be 0 <= x =< 1 for task" \
           f" {SGDRegressorOperation}" in str(val_err.value)


def test_sgd_regressor_invalid_random_state_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'random_state': -1})
    with pytest.raises(ValueError) as val_err:
        SGDRegressorModelOperation(**arguments)
    assert f"Parameter 'random_state' must be x >= 0 for task" \
           f" {SGDRegressorOperation}" in str(val_err.value)
