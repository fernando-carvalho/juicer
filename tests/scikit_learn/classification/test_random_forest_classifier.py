from tests.scikit_learn import util
from tests.scikit_learn.util import get_label_data, get_X_train_data
from juicer.scikit_learn.classification_operation import \
    RandomForestClassifierModelOperation, RandomForestClassifierOperation, \
    ClassificationModelOperation
from sklearn.ensemble import RandomForestClassifier
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


# RandomForestClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
# ({"seed": 1}, {"random_state": 1}),
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({"seed": 1}, {"random_state": 1}),

    ({"seed": 1, 'n_estimators': 101}, {"random_state": 1, 'n_estimators': 101}),

    ({"seed": 1, 'max_depth': 2}, {"random_state": 1, 'max_depth': 2}),

    ({"seed": 1, 'min_samples_split': 3},
     {"random_state": 1, 'min_samples_split': 3}),

    ({"seed": 1, "min_samples_leaf": 2},
     {"random_state": 1, "min_samples_leaf": 2}),

    ({"seed": 1, 'criterion': 'entropy'},
     {"random_state": 1, 'criterion': 'entropy'}),

    ({"seed": 1, 'min_weight_fraction_leaf': 0.3},
     {"random_state": 1, 'min_weight_fraction_leaf': 0.3}),

    ({"seed": 1, 'max_features': 2}, {"random_state": 1, 'max_features': 2}),

    ({"seed": 1, 'max_leaf_nodes': 2}, {"random_state": 1, 'max_leaf_nodes': 2}),

    ({"seed": 1, 'min_impurity_decrease': 0.5},
     {"random_state": 1, 'min_impurity_decrease': 0.5}),

    ({"seed": 1, 'bootstrap': 0}, {"random_state": 1, 'bootstrap': False}),

    ({"seed": 1, 'oob_score': 1}, {"random_state": 1, 'oob_score': True}),

    ({"seed": 1, 'n_jobs': -1}, {"random_state": 1, 'n_jobs': -1}),

    ({"seed": 1, 'ccp_alpha': 0.5}, {"random_state": 1, 'ccp_alpha': 0.5}),

    ({"seed": 1, 'max_samples': 1},
     {"random_state": 1, 'max_samples': 1 / 100.0}),

], ids=["default_params", "n_estimators_param", "max_depth_param",
        "min_samples_split", "min_samples_leaf_param", "criterion_param",
        'min_weight_fraction_leaf_param', "max_features_param",
        'max_leaf_nodes_param', 'min_impurity_decrease_param',
        "bootstrap_param", 'oob_score_param', 'n_jobs_param', 'ccp_alpha_param',
        "max_samples_param"])
def test_random_forest_classifier_params_success(get_arguments, get_df,
                                                 get_columns,
                                                 operation_par, algorithm_par):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments
    arguments['parameters'].update(operation_par)

    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_prediction_param_success(get_df, get_columns,
                                                           get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df.copy().astype(np.int64())})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_random_forest_classifier_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_random_forest_classifier_missing_label_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop('label')
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task" \
           " ClassificationModelOperation" in str(val_err.value)


def test_random_forest_classifier_missing_features_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop('features')
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert "Parameters 'features' must be informed for task" \
           " RandomForestClassifierOperation" in str(val_err.value)


def test_random_forest_classifier_invalid_min_weight_fraction_leaf_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'min_weight_fraction_leaf': -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert f"Parameter 'min_weight_fraction_leaf' must be x>=0.0 and x<=0.5 for" \
           f" task {RandomForestClassifierOperation}" in str(val_err.value)


def test_random_forest_classifier_invalid_max_samples_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({"max_samples": -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert "Parameter 'max_samples' must be x>0 and x<100, or empty, " \
           f"case you want to use a fully sample for task" \
           f" {RandomForestClassifierOperation}" in str(val_err.value)


def test_random_forest_classifier_invalid_max_features_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'max_features': -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert f"Parameter 'max_features' must be x>=0  for task" \
           f" {RandomForestClassifierOperation}" in str(val_err.value)


@pytest.mark.parametrize('par',
                         ['ccp_alpha',
                          'min_impurity_decrease']
                         )
def test_random_forest_classifier_invalid_params_fail(
        get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>=0 for task" \
           f" {RandomForestClassifierOperation}" in str(val_err.value)


@pytest.mark.parametrize('par', ['min_samples_split',
                                 'min_samples_leaf',
                                 'n_estimators'])
def test_random_forest_classifier_invalid_params_2_fail(
        par,
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {RandomForestClassifierOperation}" in str(val_err.value)
