import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from juicer.scikit_learn.classification_operation import \
    GBTClassifierModelOperation, GBTClassifierOperation
from tests.scikit_learn import util
from tests.scikit_learn.util import get_label_data, get_X_train_data


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
                       'multiplicity': {'train input data': 0},
                       'label': [get_columns[0]]},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# GBTClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({"random_state": 1}, {"random_state": 1}),

    ({'learning_rate': 0.2, "random_state": 1},
     {'learning_rate': 0.2, "random_state": 1}),

    ({'n_estimators': 102, "random_state": 1},
     {'n_estimators': 102, "random_state": 1}),

    ({"max_depth": 4, "random_state": 1},
     {"max_depth": 4, "random_state": 1}),

    ({'min_samples_split': 3, "random_state": 1},
     {'min_samples_split': 3, "random_state": 1}),

    ({'min_samples_leaf': 2, "random_state": 1},
     {'min_samples_leaf': 2, "random_state": 1}),

    ({'subsample': 0.9, "random_state": 1},
     {'subsample': 0.9, "random_state": 1}),

    ({'criterion': 'mae', "random_state": 1},
     {'criterion': 'mae', "random_state": 1}),

    ({'min_weight_fraction_leaf': 0.1, "random_state": 1},
     {'min_weight_fraction_leaf': 0.1, "random_state": 1}),

    ({'init': '"zero"', "random_state": 1},
     {'init': 'zero', "randFm_state": 1}),

    ({"random_state": 1, 'max_features': 'auto'},
     {"random_state": 1, 'max_features': 'auto'}),

    ({'max_leaf_nodes': 2, 'random_state': 1},
     {'max_leaf_nodes': 2, 'random_state': 1}),

    ({'validation_fraction': 0.2, 'random_state': 1},
     {'validation_fraction': 0.2, 'random_state': 1}),

    ({'n_iter_no_change': 4, 'random_state': 1},
     {'n_iter_no_change': 4, 'random_state': 1}),

    ({'tol': 1e-5, 'random_state': 1}, {'tol': 1e-5, 'random_state': 1})

], ids=["default_params", "learning_rate_param", "n_estimators_param",
        "max_depth_param", "min_samples_split_param", "min_samples_leaf_param",
        "subsample_param", "criterion_param", "min_weight_fraction_leaf_param",
        "init_param", "max_features_param", "max_leaf_nodes_param",
        "validation_fraction_param", "n_iter_no_change_param", "tol_param", ])
def test_gbt_classifier_params_success(get_arguments, get_df, get_columns,
                                       operation_par, algorithm_par):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_loss_param_success(get_arguments, get_columns):
    df = pd.DataFrame(
        [[5, 3], [4, 3],
         [4, 3], [4, 3],
         [5, 3], [5, 3],
         [4, 3], [5, 3],
         [4, 2], [4, 3]], columns=get_columns[0:2]).copy()
    test_df = df.copy()

    arguments = get_arguments
    arguments['parameters'].update({'features': get_columns[0:2],
                                    'label': [get_columns[0]],
                                    'loss': 'exponential',
                                    'random_state': 1})
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, get_columns[0:2])
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='exponential',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_min_impurity_decrease_param_success(get_columns,
                                                            get_arguments):
    df = util.iris(get_columns[0:2]).astype(np.int64())
    test_df = util.iris(get_columns[0:2]).astype(np.int64())
    arguments = get_arguments

    arguments['parameters'].update({"features": get_columns[0:2],
                                    "random_state": 1,
                                    "min_impurity_decrease": 0.2})

    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, get_columns[0:2])
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.2, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_prediction_param_success(get_arguments, get_df,
                                                 get_columns):
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df.astype(np.int64())})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_gbt_classifier_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
@pytest.mark.parametrize("par", [
    'min_samples_split',
    'min_samples_leaf',
    'learning_rate',
    'n_estimators',
    'max_depth'
])
def test_gbt_classifier_multiple_invalid_params_fail(par, get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task {GBTClassifierOperation}" in str(
        val_err.value)


def test_gbt_classifier_invalid_max_leafs_nodes_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'max_leaf_nodes': -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert f"Parameter 'max_leaf_nodes' must be None or x > 1 for task" \
           f" {GBTClassifierOperation}" in str(val_err.value)


def test_gbt_classifier_invalid_n_iter_no_change_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'n_iter_no_change': -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert f"Parameter 'n_iter_no_change' must be None or x > 0 for task" \
           f" {GBTClassifierOperation}" in str(val_err.value)


def test_gbt_classifier_invalid_validation_fraction_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'validation_fraction': -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert f"Parameter 'validation_fraction' must be 0 <= x =< 1 for task" \
           f" {GBTClassifierOperation}" in str(val_err.value)


def test_gbt_classifier_invalid_subsample_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'subsample': -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert f"Parameter 'subsample' must be 0 < x =< 1 for task" \
           f" {GBTClassifierOperation}" in str(val_err.value)


def test_gbt_classifier_invalid_min_weight_fraction_leaf_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'min_weight_fraction_leaf': -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert f"Parameter 'min_weight_fraction_leaf' must be 0.0" \
           f" <= x =< 0.5 for task" \
           f" {GBTClassifierOperation}" in str(val_err.value)
