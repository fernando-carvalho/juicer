import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from juicer.scikit_learn.classification_operation import KNNClassifierOperation, \
    KNNClassifierModelOperation
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


# KNNClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),
    ({'n_neighbors': 6}, {'n_neighbors': 6}),
    ({'weights': 'distance'}, {'weights': 'distance'}),
    ({'algorithm': 'brute'}, {'algorithm': 'brute'}),
    ({'leaf_size': 31}, {'leaf_size': 31}),
    ({"p": 3}, {"p": 3}),
    ({"metric": "braycurtis"}, {"metric": "braycurtis"}),
    ({'n_jobs': -1}, {'n_jobs': -1})

], ids=["default_params", "n_neighbors_param", "classifier_weights_param",
        "classifier_algorithm_param", "leaf_size_param", "p_param",
        "metric_param", "n_jobs_param"])
def test_knn_classifier_params_success(get_columns, get_df, get_arguments,
                                       operation_par, algorithm_par):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_metric_params_param_success(get_df, get_columns,
                                                    get_arguments):
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth']).copy()
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'metric_params': {'1': '1'}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params={'1': '1'}, n_jobs=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_prediction_param_success(get_columns, get_df,
                                                 get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})

    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df.astype(np.int64())})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_knn_classifier_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = KNNClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_knn_classifier_invalid_invalid_n_neighbors_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'n_neighbors': -1})
    with pytest.raises(ValueError) as val_err:
        KNNClassifierOperation(**arguments)
    assert f"Parameter 'n_neighbors' must be x>0 for task" \
           f" {KNNClassifierOperation}" in str(val_err.value)


def test_knn_classifier_invalid_p_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'p': -1})
    with pytest.raises(ValueError) as val_err:
        KNNClassifierOperation(**arguments)
    assert f"Parameter 'p' must be x>1 when parameter 'metric_params'" \
           f" is 'minkowski' for task {KNNClassifierOperation}" in str(
        val_err.value)
