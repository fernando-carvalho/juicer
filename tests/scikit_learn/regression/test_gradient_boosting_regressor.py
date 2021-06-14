import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from juicer.scikit_learn.regression_operation import \
    GradientBoostingRegressorModelOperation, GradientBoostingRegressorOperation
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


# GradientBoostingRegressor:
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),
    ({'learning_rate': 2.0}, {'learning_rate': 2.0}),
    ({'n_estimators': 200}, {'n_estimators': 200}),
    ({'max_depth': 6}, {'max_depth': 6}),
    ({'min_samples_split': 6}, {'min_samples_split': 6}),
    ({'min_samples_leaf': 3}, {'min_samples_leaf': 3}),
    ({'max_features': 'auto'}, {'max_features': 'auto'}),
    ({'criterion': 'mse'}, {'criterion': 'mse'}),
    ({'min_weight_fraction_leaf': 0.5}, {'min_weight_fraction_leaf': 0.5}),
    ({'max_leaf_nodes': 2}, {'max_leaf_nodes': 2}),
    ({'min_impurity_decrease': 0.5}, {'min_impurity_decrease': 0.5}),
    ({'random_state': 2002}, {'random_state': 2002}),
    ({'loss': 'huber'}, {'loss': 'huber'}),
    ({'subsample': 0.5}, {'subsample': 0.5}),
    ({'alpha': 0.6}, {'alpha': 0.6}),
    ({'cc_alpha': 0.5}, {'ccp_alpha': 0.5}),
    ({'validation_fraction': 0.3}, {'validation_fraction': 0.3}),
    ({'n_iter_no_change': 5}, {'n_iter_no_change': 5}),
    ({'tol': 0.01}, {'tol': 0.01})
], ids=["default_params", "learning_rate_param", "n_estimators_param",
        "max_depth_param", 'min_samples_split_param', 'min_samples_leaf_param',
        'max_features_param', 'criterion_param',
        'min_weight_fraction_leaf_param', 'max_leaf_nodes_param',
        'min_impurity_decrease_param', 'random_state_param', 'loss_param',
        'subsample_param', 'alpha_param', 'cc_alpha_param',
        'validation_fraction_param', 'n_iter_no_change_param', 'tol_param'])
def test_gradient_boosting_regressor_params_success(get_arguments,
                                                    get_columns,
                                                    get_df,
                                                    operation_par,
                                                    algorithm_par):
    df = get_df.copy()
    test_df = get_df.copy()
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = GradientBoostingRegressorModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert np.allclose(result['out'], test_df, atol=1)
    assert str(result['regressor_model']) == str(model_1)


def test_gradient_boosting_regressor_prediction_param_success(get_columns,
                                                              get_df,
                                                              get_arguments):
    df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = GradientBoostingRegressorModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),

    ("named_inputs", "train input data")

], ids=["missing_output", "missing_input"])
def test_gradient_boosting_regressor_no_code_success(get_arguments, selector,
                                                     drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = GradientBoostingRegressorModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
@pytest.mark.parametrize('par', ['learning_rate',
                                 'n_estimators',
                                 'min_samples_split',
                                 'min_samples_leaf',
                                 'max_depth'])
def test_gradient_boosting_regressor_invalid_params_1_fail(par, get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        GradientBoostingRegressorModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {GradientBoostingRegressorOperation}" in str(val_err.value)


@pytest.mark.parametrize('par', ['random_state', 'n_iter_no_change'])
def test_gradient_boosting_regressor_invalid_params_2_fail(
        par, get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        GradientBoostingRegressorModelOperation(**arguments)
    assert f"Parameter '{par}' must be x >= 0 or None for task" \
           f" {GradientBoostingRegressorOperation}" in str(val_err.value)


@pytest.mark.parametrize('par', ['cc_alpha', 'min_impurity_decrease'])
def test_gradient_boosting_regressor_cc_alpha_min_impurity_params_fail(
        get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        GradientBoostingRegressorModelOperation(**arguments)
    assert f"Parameter '{par}' must be x >= 0 for task" \
           f" {GradientBoostingRegressorOperation}" in str(val_err.value)


def test_gradient_boosting_regressor_invalid_validation_fraction_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'validation_fraction': -1})
    with pytest.raises(ValueError) as val_err:
        GradientBoostingRegressorModelOperation(**arguments)
    assert f"Parameter 'validation_fraction' must be 0 <= x <= 1 for task" \
           f" {GradientBoostingRegressorOperation}" in str(val_err.value)


def test_gradient_boosting_regressor_invalid_subsample_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'subsample': -1})
    with pytest.raises(ValueError) as val_err:
        GradientBoostingRegressorModelOperation(**arguments)
    assert f"Parameter 'subsample' must be 0 < x <= 1 for task" \
           f" {GradientBoostingRegressorOperation}" in str(val_err.value)


def test_gradient_boosting_regressor_invalid_min_wight_fraction_leaf_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'min_weight_fraction_leaf': -1})
    with pytest.raises(ValueError) as val_err:
        GradientBoostingRegressorModelOperation(**arguments)
    assert "Parameter 'min_weight_fraction_leaf' must be 0 <= x <= 0.5 for task" \
           in str(val_err.value)
