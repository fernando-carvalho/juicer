import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from juicer.scikit_learn.classification_operation import \
    DecisionTreeClassifierOperation, DecisionTreeClassifierModelOperation
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

# DecisionTreeClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),

    ({'max_depth': 2}, {'max_depth': 2}),

    ({'min_samples_split': 3}, {'min_samples_split': 3}),

    ({'min_samples_leaf': 2}, {'min_samples_leaf': 2}),

    ({'min_weight': 0.1}, {'min_weight_fraction_leaf': 0.1}),

    ({'random_state': 2002}, {'random_state': 2002}),

    ({'criterion': 'entropy'}, {'criterion': 'entropy'}),

    ({'splitter': 'random'}, {'splitter': 'random'}),

    ({'max_features': 2}, {'max_features': 2}),

    ({'max_leaf_nodes': 5}, {'max_leaf_nodes': 5}),

    ({'min_impurity_decrease': 0.5}, {'min_impurity_decrease': 0.5}),

    ({'class_weight': '"balanced"'}, {'class_weight': "balanced"})

], ids=["default_params", "max_depth_param", "min_samples_split_param",
        "min_samples_leaf_param", "min_weight_param", "random_state_param",
        "criterion_param", "splitter_param", "max_features_param",
        "max_leaf_nodes_param", "min_impurity_decrease_param",
        "class_weight_param"])
def test_decision_tree_classifier_params_success(get_arguments, get_df,
                                                 get_columns,
                                                 operation_par, algorithm_par):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_prediction_param_success(get_columns,
                                                           get_arguments,
                                                           get_df):
    arguments = get_arguments

    arguments['parameters'].update({"prediction": "success"})

    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df.astype(np.int64())})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_outputs", "missing_inputs"])
def test_decision_tree_classifier_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


@pytest.mark.parametrize("par", ["min_samples_split", "min_samples_leaf",
                                 "max_depth"])
# # # # # # # # # # Fail # # # # # # # # # #
def test_decision_tree_classifier_invalid_params_fail(
        par, get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        DecisionTreeClassifierModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {DecisionTreeClassifierOperation}" in str(val_err)


def test_decision_tree_classifier_invalid_min_weight_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'min_weight': -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        DecisionTreeClassifierModelOperation(**arguments)
    assert f"Parameter 'min_weight' must be x>=0 or x<=0.5 for task" \
           f" {DecisionTreeClassifierOperation}" in str(val_err)


def test_decision_tree_classifier_min_impurity_decrease_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'min_impurity_decrease': -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        DecisionTreeClassifierModelOperation(**arguments)
    assert f"Parameter 'min_impurity_decrease' must be x>=0 for task " \
           f"{DecisionTreeClassifierOperation}" in str(val_err)
