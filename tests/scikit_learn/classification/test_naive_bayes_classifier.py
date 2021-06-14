from tests.scikit_learn import util
from tests.scikit_learn.util import get_X_train_data, get_label_data
from juicer.scikit_learn.classification_operation import \
    NaiveBayesClassifierModelOperation, NaiveBayesClassifierOperation
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
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


# NaiveBayesClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),
    ({"alpha": 1.1}, {"alpha": 1.1}),
    ({"fit_prior": 0}, {"fit_prior": False}),
], ids=["default_params", "alpha_param", "fit_prior_param"])
def test_naive_bayes_classifier_multinomialnb_params_success(get_columns,
                                                             get_arguments,
                                                             get_df,
                                                             operation_par,
                                                             algorithm_par):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = MultinomialNB(alpha=1.0,
                            class_prior=None, fit_prior=True)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_multinomialnb_class_prior_param_success(
        get_arguments, get_columns):
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=get_columns[0:2]).copy()
    test_df = df.copy()

    arguments = get_arguments
    arguments['parameters'].update({'features': get_columns[0:2],
                                    'label': [get_columns[0]],
                                    'class_prior': "0.1, 0.1, 0.1, 0.1, 0.1,"
                                                   " 0.1, 0.1, 0.1, 0.1, 0.1"})
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns[0:2])
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = MultinomialNB(alpha=1.0,
                            class_prior=[0.1, 0.1, 0.1, 0.1, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1],
                            fit_prior=True)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_type_gaussiannb_var_smoothing_param_success(
        get_arguments, get_df, get_columns):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments

    arguments['parameters'].update({"type": "GuassianNB", "var_smoothing": 1e-8})

    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = GaussianNB(priors=None, var_smoothing=1e-8)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_type_gaussiannb_priors_param_success(
        get_arguments, get_columns):
    df = util.iris(get_columns[0:2], size=10).astype(np.int64())
    test_df = df.copy()

    arguments = get_arguments
    arguments['parameters'].update({'type': 'GaussianNB', 'priors': "0.5, 0.5",
                                    'features': get_columns[0:2],
                                    'label': [get_columns[0]]})
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns[0:2])
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = GaussianNB(priors=[0.5, 0.5],
                         var_smoothing=1e-09)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_type_bernoulli_binarize_param_success(get_df,
                                                                      get_arguments,
                                                                      get_columns):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())

    arguments = get_arguments
    arguments['parameters'].update({'type': 'Bernoulli', 'binarize': 10})

    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = BernoulliNB(alpha=1.0,
                          class_prior=None, fit_prior=True,
                          binarize=10.0)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_prediction_param_success(get_columns,
                                                         get_arguments, get_df
                                                         ):
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df.astype(np.int64())})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_naive_bayes_classifier_no_code_success(selector, drop,
                                                get_arguments):
    arguments = get_arguments
    arguments[selector].pop(drop)
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_naive_bayes_classifier_invalid_alpha_param_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'alpha': -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        NaiveBayesClassifierModelOperation(**arguments)
    assert f"Parameter 'alpha' must be x>0 for task" \
           f" {NaiveBayesClassifierOperation}" in str(val_err.value)
