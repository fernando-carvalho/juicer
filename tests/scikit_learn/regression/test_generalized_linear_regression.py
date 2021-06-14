import pytest
from sklearn.linear_model import LinearRegression

from juicer.scikit_learn.regression_operation import \
    GeneralizedLinearRegressionModelOperation, \
    GeneralizedLinearRegressionOperation
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
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }


# GeneralizedLinearRegression:
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({'fit_intercept': 0, 'copy_X': 0},
     {'fit_intercept': False, 'copy_X': False}),

    ({'fit_intercept': 1, 'copy_X': 0},
     {'fit_intercept': True, 'copy_X': False}),

    ({'fit_intercept': 1, 'copy_X': 0, 'normalize': 1},
     {'fit_intercept': True, 'copy_X': False, 'normalize': True}),

    ({'fit_intercept': 1, 'copy_X': 1}, {'fit_intercept': True, 'copy_X': True}),

    ({'fit_intercept': 0, 'copy_X': 0, 'n_jobs': 4},
     {'copy_X': False, 'n_jobs': 4})
], ids=['default_params', "fit_intercept_param", "normalize_param",
        "copy_x_param", "n_jobs_param"])
def test_generalized_linear_regression_success(get_columns, get_arguments,
                                               get_df,
                                               algorithm_par,
                                               operation_par):
    df = get_df.copy()
    test_df = get_df.copy()
    arguments = get_arguments

    arguments['parameters'].update(operation_par)
    util.add_minimum_ml_args(arguments)
    instance = GeneralizedLinearRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    model_1 = LinearRegression(
        fit_intercept=False, copy_X=False,
        n_jobs=1)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()

    assert str(result['regressor_model']) == str(model_1)
    assert result['out'].equals(test_df)


def test_generalized_linear_regression_alias_param_success(get_arguments,
                                                           get_df):
    df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = GeneralizedLinearRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),

    ("named_inputs", "input data")

], ids=["missing_output", "missing_input"])
def test_generalized_linear_regression_no_code_success(selector, drop,
                                                       get_arguments):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = GeneralizedLinearRegressionModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_generalized_linear_regression_missing_label_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop('label')
    with pytest.raises(ValueError) as val_err:
        GeneralizedLinearRegressionModelOperation(**arguments)
    print(val_err)
    assert "Parameters 'features' and 'label' must be informed for task" \
           " RegressionModelOperation" in str(val_err.value)


def test_generalized_linear_regression_invalid_n_jobs_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'n_jobs': -2})
    with pytest.raises(ValueError) as val_err:
        GeneralizedLinearRegressionModelOperation(**arguments)

    assert f"Parameter 'n_jobs' must be x>=-1 for task" \
           f" {GeneralizedLinearRegressionOperation}" in str(val_err.value)
