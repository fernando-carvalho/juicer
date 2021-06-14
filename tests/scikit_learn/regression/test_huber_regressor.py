import pytest
from sklearn.linear_model import HuberRegressor

from juicer.scikit_learn.regression_operation import \
    HuberRegressorModelOperation, \
    HuberRegressorOperation
from tests.scikit_learn import util


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


# HuberRegressor
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),
    ({'epsilon': 1.6}, {'epsilon': 1.6}),
    ({'max_iter': 5}, {'max_iter': 5}),
    ({'alpha': 0.2}, {'alpha': 0.2}),
    ({'tol': 0.5}, {'tol': 0.5}),
    ({'fit_intercept': 0}, {'fit_intercept': False})
], ids=["default_params", 'epsilon_param', 'max_iter_param', 'alpha_param',
        'tol_param', 'fit_intercept_param'])
def test_huber_regressor_params_success(get_arguments, get_columns, get_df,
                                        operation_par, algorithm_par):
    df = get_df.copy()
    test_df = get_df.copy()

    arguments = get_arguments
    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = HuberRegressorModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    x_train = util.get_X_train_data(test_df, get_columns)
    y = util.get_label_data(test_df, [get_columns[0]])

    model_1 = HuberRegressor(epsilon=1.35, max_iter=100,
                             alpha=0.0001, tol=1e-05, fit_intercept=True,
                             warm_start=False)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()

    assert str(result['regressor_model']) == str(model_1)
    assert result['out'].equals(test_df)


def test_huber_regressor_prediction_param_success(get_arguments, get_df):
    df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = HuberRegressorModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),

    ("named_inputs", "train input data")

], ids=["missing_output", "missing_input"])
def test_huber_regressor_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = HuberRegressorModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
@pytest.mark.parametrize('par', ['max_iter', 'alpha'])
def test_huber_regressor_invalid_max_iter_param_fail(par, get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        HuberRegressorModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {HuberRegressorOperation}" in str(val_err.value)


def test_huber_regressor_invalid_epsilon_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'epsilon': 0.5})
    with pytest.raises(ValueError) as val_err:
        HuberRegressorModelOperation(**arguments)
    assert f"Parameter 'epsilon' must be x>1.0 for task" \
           f" {HuberRegressorOperation}" in str(val_err.value)
