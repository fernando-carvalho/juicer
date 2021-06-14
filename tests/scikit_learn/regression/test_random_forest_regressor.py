import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from juicer.scikit_learn.regression_operation import \
    RandomForestRegressorModelOperation, RandomForestRegressorOperation
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


# RandomForestRegressor:
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),
    ({'n_estimators': 200}, {'n_estimators': 200}),
    ({'max_features': 'sqrt'}, {'max_features': 'sqrt'}),
    ({'max_depth': 5}, {'max_depth': 5}),
    ({'min_samples_split': 6}, {'min_samples_split': 6}),
    ({'min_samples_leaf': 8}, {'min_samples_leaf': 8}),
    ({'criterion': 'mae'}, {'criterion': 'mae'}),
    ({'min_weight_fraction_leaf': 0.5}, {'min_weight_fraction_leaf': 0.5}),
    ({'max_leaf_nodes': 5}, {'max_leaf_nodes': 5}),
    ({'min_impurity_decrease': 0.5}, {'min_impurity_decrease': 0.5}),

    ({'bootstrap': False, 'oob_score': False},
     {'bootstrap': False, 'oob_score': False}),

    ({'n_jobs': 3}, {'n_jobs': 3}),
    ({'random_state': 2002}, {'random_state': 2002})

], ids=["default_params", 'n_estimators_param', 'max_features_param',
        'max_depth_param', 'min_samples_split_param', 'min_samples_leaf_param',
        'criterion_param', 'min_weight_fraction_leaf_param',
        'max_leaf_nodes_params', 'min_impurity_decrease_params',
        'bootstrap_and_oob_score_params', 'n_jobs_param', 'random_state_param'])
def test_random_forest_regressor_success(get_arguments, get_columns, get_df,
                                         operation_par, algorithm_par):
    df = get_df.copy()
    test_df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = RandomForestRegressorModelOperation(**arguments)

    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()

    assert np.allclose(result['out'], test_df, atol=1)
    assert str(result['regressor_model']) == str(model_1)


def test_random_forest_regressor_prediction_param_success(get_arguments,
                                                          get_df):
    df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = RandomForestRegressorModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),
    ('named_inputs', 'train input data')
], ids=["missing_output", "missing_input"])
def test_random_forest_regressor_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    util.add_minimum_ml_args(arguments)
    instance = RandomForestRegressorModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
@pytest.mark.parametrize('par', ['n_estimators', 'min_samples_split',
                                 'min_samples_leaf'])
def test_random_forest_regressor_multiple_params_fail(par, get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {RandomForestRegressorOperation}" in str(val_err.value)


def test_random_forest_regressor_invalid_n_jobs_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'n_jobs': -3})
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorModelOperation(**arguments)
    assert f"Parameter 'n_jobs' must be x >= -1 for task" \
           f" {RandomForestRegressorOperation}" in str(val_err.value)


def test_random_forest_regressor_invalid_max_depth_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'max_depth': -5})
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorModelOperation(**arguments)
    assert f"Parameter 'max_depth' must be x>0 or None for task" \
           f" {RandomForestRegressorOperation}" in str(val_err.value)


def test_random_forest_regressor_invalid_random_state_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'random_state': -1})
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorModelOperation(**arguments)
    assert f"Parameter 'random_state' must be x>=0 or None for task" \
           f" {RandomForestRegressorOperation}" in str(val_err.value)


def test_random_forest_regressor_invalid_min_weight_fraction_leaf_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'min_weight_fraction_leaf': 5})
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorModelOperation(**arguments)
    assert "Parameter 'min_weight_fraction_leaf' must be x >= 0 and x" \
           f" <= 0.5 for task {RandomForestRegressorOperation}" in str(
        val_err.value)


def test_random_forest_regressor_invalid_min_impurity_decrease_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'min_impurity_decrease': -1})
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorModelOperation(**arguments)
    assert "Parameter 'min_impurity_decrease' must be x>=0 or None for task" \
           f" {RandomForestRegressorOperation}" in str(val_err.value)
