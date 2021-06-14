import numpy as np
import pytest
from sklearn.linear_model import ElasticNet

from juicer.scikit_learn.regression_operation import \
    LinearRegressionModelOperation, LinearRegressionOperation
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


# LinearRegression:
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({"normalize": False}, {"normalize": False}),

    ({'alpha': 0.5}, {'alpha': 0.5, 'normalize': True}),

    ({'l1_ratio': 1.0}, {'l1_ratio': 1.0, 'normalize': True}),

    ({'normalize': True}, {'normalize': True}),

    ({'max_iter': 500}, {'max_iter': 500, 'normalize': True}),

    ({'tol': 0.652}, {'tol': 0.652, 'normalize': True}),

    ({'random_state': 2002}, {'random_state': 2002, 'normalize': True}),

    ({'positive': True}, {'positive': True, 'normalize': True}),

    ({'random_state': 1, 'fit_intercept': True},
     {'random_state': 1, 'fit_intercept': True, 'normalize': True})
], ids=["default_params", 'alpha_param', 'l1_ratio_param', 'normalize_param',
        'max_iter_param', 'tol_param', "random_state_param", 'positive_param',
        'fit_intercept_param'])
def test_linear_regression_params_success(get_arguments, get_columns, get_df,
                                          operation_par, algorithm_par):
    df = get_df.copy()
    test_df = get_df.copy()
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = LinearRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])

    model_1 = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.0001,
                         max_iter=1000, random_state=None,
                         normalize=False, positive=False,
                         fit_intercept=False)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert np.allclose(result['out'], test_df, atol=1)
    assert str(result['regressor_model']) == str(model_1)


def test_linear_regression_prediction_param_success(get_arguments, get_df):
    df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = LinearRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),

    ("named_inputs", "train input data")

], ids=["missing_output", "missing_input"])
def test_linear_regression_no_code_success(selector, drop, get_arguments):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = LinearRegressionModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize('par', ['alpha', 'max_iter'])
def test_linear_regression_invalid_params_fail(get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        LinearRegressionModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {LinearRegressionOperation}" in str(val_err.value)


def test_linear_regression_invalid_l1_ratio_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'l1_ratio': -1})
    with pytest.raises(ValueError) as val_err:
        LinearRegressionModelOperation(**arguments)
    assert f"Parameter 'l1_ratio' must be 0<=x<=1 for task" \
           f" {LinearRegressionOperation}" in str(val_err.value)


def test_linear_regression_invalid_random_state_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'random_state': -1})
    with pytest.raises(ValueError) as val_err:
        LinearRegressionModelOperation(**arguments)
    assert f"Parameter 'random_state' must be x>=0 for task" \
           f" {LinearRegressionOperation}" in str(val_err.value)


def test_linear_regression_missing_label_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop('label')
    with pytest.raises(ValueError) as val_err:
        LinearRegressionModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task" \
           " RegressionModelOperation" in str(
        val_err.value)
