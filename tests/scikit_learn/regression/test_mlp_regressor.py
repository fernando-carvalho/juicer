import pytest
from sklearn.neural_network import MLPRegressor

from juicer.scikit_learn.regression_operation import MLPRegressorModelOperation, \
    MLPRegressorOperation
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


# MLPRegressor
#
#
# # # # # # # # # # Success # # # # # # # # # #
#
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({'random_state': 1}, {'random_state': 1}),

    ({'random_state': 1, 'layer_sizes': '(10, 200, 10)'},
     {'random_state': 1, 'hidden_layer_sizes': (10, 200, 10)}),

    ({'random_state': 1, 'activation': 'identity'},
     {'random_state': 1, 'activation': 'identity'}),

    ({'random_state': 1, 'activation': 'logistic'},
     {'random_state': 1, 'activation': 'logistic'}),

    ({'random_state': 1, 'activation': 'tanh'},
     {'random_state': 1, 'activation': 'tanh'}),

    ({'random_state': 1, 'solver': 'lbfgs'},
     {'random_state': 1, 'solver': 'lbfgs'}),

    ({'random_state': 1, 'solver': 'sgd'},
     {'random_state': 1, 'solver': 'sgd'}),

    ({'random_state': 1, 'alpha': 0.01},
     {'random_state': 1, 'alpha': 0.01}),

    ({'random_state': 1, 'max_iter': 10},
     {'random_state': 1, 'max_iter': 10}),

    ({'random_state': 1, 'tol': 0.01},
     {'random_state': 1, 'tol': 0.01}),

    ({'random_state': 1, 'batch_size': 20},
     {'random_state': 1, 'batch_size': 20}),

    ({'random_state': 1, 'solver': 'sgd', 'learning_rate': 'adaptive'},
     {'random_state': 1, 'solver': 'sgd', 'learning_rate': 'adaptive'}),

    ({'random_state': 1, 'solver': 'adam', 'learning_rate_init': 0.2},
     {'random_state': 1, 'solver': 'adam', 'learning_rate_init': 0.2}),

    ({'random_state': 1, 'solver': 'sgd', 'power_t': 0.2},
     {'random_state': 1, 'solver': 'sgd', 'power_t': 0.2}),

    ({'random_state': 1, 'solver': 'adam', 'shuffle': 0, },
     {'random_state': 1, 'solver': 'adam', 'shuffle': False, }),

    ({'random_state': 1, 'n_iter_no_change': 1},
     {'random_state': 1, 'n_iter_no_change': 1}),

    ({'random_state': 1, 'solver': 'sgd', 'momentum': 0.1},
     {'random_state': 1, 'solver': 'sgd', 'momentum': 0.1}),

    ({'random_state': 1, 'solver': 'sgd', 'nesterovs_momentum': 0},
     {'random_state': 1, 'solver': 'sgd', 'nesterovs_momentum': False}),

    ({'random_state': 1, 'early_stopping': 1, 'validation_fraction': 0.2},
     {'random_state': 1, 'early_stopping': True, 'validation_fraction': 0.2}),

    ({'random_state': 1, 'beta_1': 0.5, 'beta_2': 0.6},
     {'random_state': 1, 'beta_1': 0.5, 'beta_2': 0.6}),

    ({'random_state': 1, 'epsilon': 0.1}, {'random_state': 1, 'epsilon': 0.1}),

], ids=['default_params', "layer_sizes_param", "activation_identity_param",
        "activation_logistic_param", "tanh_logistic_param", 'solver_lgfgs_param',
        'solver_sgd_param', 'alpha_param', "max_iter_param", "tol_param",
        "batch_size_param", "learning_rate_param", "learning_rate_init_param",
        "power_t_param", "shuffle_param", "n_iter_no_change_param",
        'momentum_param', 'nesterovs_momentum_param',
        'early_stopping_and_validation_fraction_params', "beta_1_beta_2_params",
        'epsilon_param'])
def test_mlp_regressor_params_success(get_columns, get_arguments, get_df,
                                      operation_par,
                                      algorithm_par):
    df = get_df.copy()
    test_df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update(operation_par)

    util.add_minimum_ml_args(arguments)
    instance = MLPRegressorModelOperation(**arguments)

    result = util.execute(util.get_complete_code(instance), {'df': df})

    x_train = get_X_train_data(test_df, get_columns)
    y = get_label_data(test_df, [get_columns[0]])

    model_1 = MLPRegressor(random_state=1, hidden_layer_sizes=(1, 100, 1),
                           activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto')

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()

    assert str(result['regressor_model']) == str(model_1)
    assert result['out'].equals(test_df)


def test_mlp_regressor_prediction_param_success(get_df,
                                                get_arguments):
    df = get_df.copy()
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})
    util.add_minimum_ml_args(arguments)
    instance = MLPRegressorModelOperation(**arguments)

    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[4] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),

    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_mlp_regressor_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    util.add_minimum_ml_args(arguments)
    instance = MLPRegressorModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_mlp_regressor_invalid_layer_sizes_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'layer_sizes': (1, 2)})
    with pytest.raises(ValueError) as val_err:
        MLPRegressorModelOperation(**arguments)

    assert "Parameter 'layer_sizes' must be a tuple with the size of each" \
           f" layer for task {MLPRegressorOperation}" in str(val_err)


@pytest.mark.parametrize('par', ['alpha', 'max_iter', 'tol'])
def test_mlp_regressor_multiple_invalid_params_fail(get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        MLPRegressorModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>=0 for task" \
           f" {MLPRegressorOperation}" in str(val_err.value)


def test_mlp_regressor_invalid_momentum_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'momentum': -1})
    with pytest.raises(ValueError) as val_err:
        MLPRegressorModelOperation(**arguments)
    assert "Parameter 'momentum' must be x between" \
           f" 0 and 1 for task {MLPRegressorOperation}" in str(val_err.value)


@pytest.mark.parametrize('par', ['beta_1', 'beta_2'])
def test_mlp_regressor_invalid_beta_1_param_fail(get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        MLPRegressorModelOperation(**arguments)
    assert f"Parameter '{par}' must be in [0, 1) for task" \
           f" {MLPRegressorOperation}" in str(val_err.value)
