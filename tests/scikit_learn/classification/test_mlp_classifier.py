import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier

from juicer.scikit_learn.classification_operation \
    import MLPClassifierModelOperation, MLPClassifierOperation
from tests.scikit_learn import util
from tests.scikit_learn.util import get_label_data, get_X_train_data


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
                       'label': [get_columns[0]],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }


# MLPClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
#     ({"seed": 1}, {"random_state": 1})
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({"seed": 1}, {"random_state": 1}),

    ({"seed": 1, 'hidden_layer_sizes': '(1, 101)'},
     {"random_state": 1, 'hidden_layer_sizes': (1, 101)}),

    ({"seed": 1, 'activation': 'tanh'},
     {"random_state": 1, 'activation': 'tanh'}),

    ({"seed": 1, 'solver': 'lbfgs'}, {"random_state": 1, 'solver': 'lbfgs'}),

    ({"seed": 1, 'alpha': 0.0002}, {"random_state": 1, 'alpha': 0.0002}),

    ({"seed": 1, 'max_iter': 201}, {"random_state": 1, 'max_iter': 201}),

    ({"seed": 1, 'tol': 1e-5}, {"random_state": 1, 'tol': 1e-5}),

    ({"seed": 1, 'batch_size': 201}, {"random_state": 1, 'batch_size': 201}),

    ({"seed": 1, 'solver': 'sgd', 'learning_rate': 'adaptive'},
     {"random_state": 1, 'solver': 'sgd', 'learning_rate': 'adaptive'}),

    ({"seed": 1, 'learning_rate_init': 0.002},
     {"random_state": 1, 'learning_rate_init': 0.002}),

    ({"seed": 1, 'solver': 'sgd', 'power_t': 0.6},
     {"random_state": 1, 'solver': 'sgd', 'power_t': 0.6}),

    ({"seed": 1, 'shuffle': 0}, {"random_state": 1, 'shuffle': False}),

    ({"seed": 1, 'solver': 'sgd', 'momentum': 0.8},
     {"random_state": 1, 'solver': 'sgd', 'momentum': 0.8}),

    ({"seed": 1, 'solver': 'sgd', 'nesterovs_momentum': 0},
     {"random_state": 1, 'solver': 'sgd', 'nesterovs_momentum': False}),

    ({"seed": 1, 'early_stopping': 1},
     {"random_state": 1, 'early_stopping': True}),

    ({"seed": 1, 'solver': 'sgd', 'early_stopping': 1,
      'validation_fraction': 0.2},
     {"random_state": 1, 'solver': 'sgd', 'early_stopping': True,
      'validation_fraction': 0.2}),

    ({"seed": 1, 'beta1': 0.8, 'beta2': 0.988},
     {"random_state": 1, 'beta_1': 0.8, 'beta_2': 0.988}),

    ({"seed": 1, 'epsilon': 1e-7}, {"random_state": 1, 'epsilon': 1e-7}),

    ({"seed": 1, 'n_iter_no_change': 11},
     {"random_state": 1, 'n_iter_no_change': 11}),

    ({"seed": 1, 'solver': 'lbfgs', 'max_fun': 15050},
     {"random_state": 1, 'solver': 'lbfgs', 'max_fun': 15050})

], ids=["default_params", "hidden_layer_sizes_param", "activation_param",
        "solver_param", "alpha_param", "max_iter_param", "tol_param",
        "batch_size_param", "learning_rate_param", "learning_rate_init_param",
        "power_t_param", "shuffle_param", "momentum_param",
        "nesterovs_momentum_param", "early_stopping_param",
        "validation_fraction_param", "beta_1_beta_2_params", "epsilon_param",
        "n_iter_no_change_param", "max_fun_param"])
def test_mlp_classifier_params_success(get_arguments, get_df, get_columns,
                                       operation_par, algorithm_par):
    df = get_df.copy().astype(np.int64())
    test_df = get_df.copy().astype(np.int64())
    arguments = get_arguments
    arguments['parameters'].update(operation_par)

    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_prediction_param_success(get_arguments, get_df):
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df.astype(np.int64())})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_mlp_classifier_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
@pytest.mark.parametrize('par', ["tol", "max_iter", "learning_rate_init",
                                 "n_iter_no_change", "max_fun"])
def test_mlp_classifier_invalid_params_fail(par, get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})

    if par == "max_fun":
        arguments['parameters'].update({'solver': 'lbfgs'})

    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter '{par}' must be x > 0 for task " \
           f"{MLPClassifierOperation}" in str(val_err)


def test_mlp_classifier_invalid_alpha_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({"alpha": -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'alpha' must be x >= 0 for task " \
           f"{MLPClassifierOperation}" in str(val_err.value)


def test_mlp_classifier_invalid_hidden_layers_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'hidden_layer_sizes': '(1)'})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'hidden_layer_sizes' must be a tuple with the size of" \
           f" each layer for task {MLPClassifierOperation}" in str(val_err.value)


def test_mlp_classifier_invalid_momentum_validation_fration_params_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({"solver": "sgd", "momentum": -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'momentum' must be x between 0 and 1 for task " \
           f"{MLPClassifierOperation}" in str(val_err.value)


def test_mlp_classifier_invalid_validation_fraction_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update(
        {"early_stopping": 1, "validation_fraction": -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'validation_fraction' must be x between 0 and 1 for task " \
           f"{MLPClassifierOperation}" in str(val_err.value)


@pytest.mark.parametrize('par', ['beta1', 'beta2'])
def test_mlp_classifier_invalid_beta_1_beta_2_params_fail(par, get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'solver': 'adam', par: -1})
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter '{par}' must be in [0, 1) for task" \
           f" {MLPClassifierOperation}" in str(val_err.value)
