import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from juicer.scikit_learn.regression_operation import \
    IsotonicRegressionModelOperation, IsotonicRegressionOperation
from tests.scikit_learn import util


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

@pytest.fixture
def get_columns():
    return ['sepallength']


@pytest.fixture
def get_df(get_columns):
    return util.iris(get_columns)


@pytest.fixture
def get_arguments(get_columns):
    return {
        'parameters': {'features': get_columns,
                       'label': get_columns,
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }


# IsotonicRegression:
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({'isotonic': False}, {'increasing': False}),

    ({'isotonic': True}, {'increasing': True}),

    ({'isotonic': False, 'y_min': 2.0, 'y_max': 4.0},
     {'isotonic': False, 'y_min': 2.0, 'y_max': 4.0}),

    ({'isotonic': False, 'out_of_bounds': 'clip'},
     {'increasing': False, 'out_of_bounds': 'clip'})

], ids=["default_params", "isotonic_param", "y_min_and_y_max_param",
        'out_of_bounds_param'])
def test_isotonic_regression_params_success(get_arguments, get_columns, get_df,
                                            operation_par, algorithm_par):
    df = get_df.copy()
    test_df = get_df.copy()

    arguments = get_arguments
    arguments['parameters'].update(operation_par)
    util.add_minimum_ml_args(arguments)
    instance = IsotonicRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = IsotonicRegression(y_min=None, y_max=None, increasing=False,
                                 out_of_bounds='nan')

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    x_train = util.get_X_train_data(test_df, get_columns)
    y = util.get_label_data(test_df, get_columns)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert str(result['regressor_model']) == str(model_1)
    assert result['out'].equals(test_df)


def test_isotonic_regression_prediction_param_success(get_arguments,
                                                      get_df):
    df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update({'isotonic': False, 'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = IsotonicRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[1] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),

    ("named_inputs", "train input data")

], ids=["missing_output", "missing_input"])
def test_isotonic_regression_no_code_success(selector, drop, get_arguments):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = IsotonicRegressionModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_isotonic_regression_invalid_size_features_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'features': ['sepallength', 'sepalwidth'],
                                    'isotonic': False})
    with pytest.raises(ValueError) as val_err:
        IsotonicRegressionModelOperation(**arguments)
    assert f"Parameter 'features' must be x<2 for task" \
           f" {IsotonicRegressionOperation}" in \
           str(val_err.value)


def test_isotonic_regression_invalid_y_min_y_max_params_values_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'y_min': 2, 'y_max': 1, 'isotonic': False})
    with pytest.raises(ValueError) as val_err:
        IsotonicRegressionModelOperation(**arguments)
    assert "Parameter 'y_min' must be less than or equal" \
           f" to 'y_max' for " \
           f"task {IsotonicRegressionOperation}" in str(val_err.value)
