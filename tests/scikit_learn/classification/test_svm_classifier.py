from tests.scikit_learn import util
from tests.scikit_learn.util import get_label_data, get_X_train_data
from juicer.scikit_learn.classification_operation \
    import SvmClassifierModelOperation, SvmClassifierOperation
from sklearn.svm import SVC
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


# SvmClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),
    ({"c": 1.1}, {"C": 1.1}),
    ({"kernel": "linear"}, {"kernel": "linear"}),
    ({"degree": 4}, {"degree": 4}),
    ({"tol": 1e-4}, {"tol": 1e-4}),
    ({'max_iter': 2}, {'max_iter': 2}),
    ({'seed': 2002}, {"random_state": 2002}),
    ({'gamma': 'auto'}, {'gamma': 'auto'}),
    ({'coef0': 0.5}, {'coef0': 0.5}),
    ({'shrinking': 0}, {'shrinking': False}),
    ({'probability': 1}, {'probability': True}),
    ({'decision_function_shape': 'ovo'}, {'decision_function_shape': 'ovo'})
], ids=["default_params", "c_param", "kernel_param", "degree_param",
        "tol_param", "max_iter_param", "seed_param", "gamma_param",
        "coef0_param", 'shrinking_param', 'probability_param',
        'decision_function_shape_param'])
def test_svm_classifier_params_success(get_arguments, get_df, get_columns,
                                       operation_par, algorithm_par):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_prediction_param_success(get_arguments, get_df):
    arguments = get_arguments
    arguments['parameters'].update({"prediction": "success"})
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df.astype(np.int64())})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_svm_classifier_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_svm_classifier_missing_label_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop("label")
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        SvmClassifierModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task" \
           " ClassificationModelOperation" in str(val_err.value)


def test_svm_classifier_missing_features_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop("features")
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        SvmClassifierModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task" \
           " ClassificationModelOperation" in str(val_err.value)


@pytest.mark.parametrize('par', ['degree', 'c'])
def test_svm_classifier_multiple_invalid_params_fail(par, get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        SvmClassifierModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {SvmClassifierOperation}" in str(val_err.value)
