from textwrap import dedent
import re
from juicer.operation import Operation
from juicer.util.template_util import *
from juicer.scikit_learn.util import get_X_train_data, get_label_data
from juicer.scikit_learn.model_operation import AlgorithmOperation


class ClassificationModelOperation(Operation):

    LABEL_ATTRIBUTE_PARAM = 'label'
    FEATURES_ATTRIBUTE_PARAM = 'features'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) >= 1 and any(
                [len(self.named_outputs) >= 1, self.contains_results()])

        if not self.has_code and len(self.named_outputs) > 0:
            raise ValueError(
                    _('Model is being used, but at least one input is missing'))

        if any([self.FEATURES_ATTRIBUTE_PARAM not in parameters,
                self.LABEL_ATTRIBUTE_PARAM not in parameters]):
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                    self.FEATURES_ATTRIBUTE_PARAM, self.LABEL_ATTRIBUTE_PARAM,
                    self.__class__.__name__))

        self.label = parameters.get(self.LABEL_ATTRIBUTE_PARAM)
        self.features = parameters.get(self.FEATURES_ATTRIBUTE_PARAM)
        self.prediction = parameters.get(self.PREDICTION_ATTRIBUTE_PARAM,
                                         'prediction')

        self.model = self.named_outputs.get('model',
                                            'model_task_{}'.format(self.order))
        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.metrics_code = ""
        self.transpiler_utils.add_custom_function(
                'get_X_train_data', get_X_train_data)
        self.transpiler_utils.add_custom_function(
                'get_label_data', get_label_data)

        self.perform_cross_validation = parameters.get(
                'apply_cross_validation') in [True, '1', 1]
        if self.perform_cross_validation:
            self.transpiler_utils.add_import("from sklearn.model_selection "
                                             "import cross_val_score, KFold")
            self.cross_validation_metric = \
                parameters.get('metric_cross_validation', 'f1_weighted')
            self.kfold = int(parameters.get('folds', 3))

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['train input data'] > 1 else ""
        if self.perform_cross_validation:
            fit_code = "avg_score = cross_val_score(algorithm, X, y, " \
                       "cv={folds}, scoring='{metric}').mean()" \
                .format(folds=self.kfold, seed=None,
                        metric=self.cross_validation_metric)
            score = '["Average score in cross-validation ({k}-fold)", ' \
                    'avg_score],'.format(k=self.kfold)
        else:
            score = ""
            fit_code = ""

        code = """
        X = get_X_train_data({input}, {features})
        y = get_label_data({input}, {label})
        classification_model = algorithm.fit(X, y)
        {fit_code}
        {output} = {input}{copy_code}
        prediction = classification_model.predict(X).tolist()
        {output}['{predCol}'] = prediction
        {model} = classification_model

        display_text = {display_text}
        if display_text:
            metric_rows = [{score}{metrics_append}]

            if metric_rows:
                metrics_content = SimpleTableReport(
                    'table table-striped table-bordered w-auto', [],
                    metric_rows, title='{metrics}')

                emit_event('update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=metrics_content.generate(),
                    type='HTML', title='{metrics}',
                    task={{'id': '{task_id}' }},
                    operation={{'id': {operation_id} }},
                    operation_id={operation_id})
            """.format(model=self.model, label=self.label, copy_code=copy_code,
                       input=self.named_inputs['train input data'],
                       score=score,
                       features=self.features, output=self.output,
                       fit_code=fit_code,
                       predCol=self.prediction,
                       metrics_append=self.metrics_code,
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title=_("Clustering result"),
                       summary=gettext('Summary'),
                       metrics=gettext('Metrics'),
                       display_text=self.parameters['task']['forms'].get(
                               'display_text', {}).get('value') in (1, '1'))

        return dedent(code)


class ClassificationOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs, algorithm):
        model_in_ports = {
            'train input data': named_inputs.get('train input data'),
            'algorithm': 'algorithm'}

        model = ClassificationModelOperation(
                parameters, model_in_ports, named_outputs)
        super(ClassificationOperation, self).__init__(
                parameters, named_inputs, named_outputs, model, algorithm)
        model.metrics_code = algorithm.get_output_metrics_code()


class DecisionTreeClassifierModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = DecisionTreeClassifierOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(DecisionTreeClassifierModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class GBTClassifierModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = GBTClassifierOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(GBTClassifierModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class KNNClassifierModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = KNNClassifierOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(KNNClassifierModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class LogisticRegressionModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = LogisticRegressionOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(LogisticRegressionModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class MLPClassifierModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = MLPClassifierOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(MLPClassifierModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class NaiveBayesClassifierModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = NaiveBayesClassifierOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(NaiveBayesClassifierModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class PerceptronClassifierModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = PerceptronClassifierOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(PerceptronClassifierModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class RandomForestClassifierModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = RandomForestClassifierOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(RandomForestClassifierModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class SvmClassifierModelOperation(ClassificationOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = SvmClassifierOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(SvmClassifierModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class DecisionTreeClassifierOperation(Operation):
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    MIN_WEIGHT_PARAM = 'min_weight'
    SEED_PARAM = 'random_state'
    CRITERION_PARAM = 'criterion'
    SPLITTER_PARAM = 'splitter'
    MAX_FEATURES_PARAM = 'max_features'
    MAX_LEAF_NODES_PARAM = 'max_leaf_nodes'
    MIN_IMPURITY_DECREASE_PARAM = 'min_impurity_decrease'
    CLASS_WEIGHT_PARAM = 'class_weight'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:
            self.min_split = int(parameters.get(self.MIN_SPLIT_PARAM, 2) or 2)
            self.min_leaf = int(parameters.get(self.MIN_LEAF_PARAM, 1) or 1)
            max_depth_ = parameters.get(self.MAX_DEPTH_PARAM, None)
            self.max_depth = None if max_depth_ is None else int(max_depth_)
            self.min_weight = float(parameters.get(
                    self.MIN_WEIGHT_PARAM, 0.0)) or 0.0
            self.seed = parameters.get(self.SEED_PARAM, None) or None
            self.criterion = parameters.get(
                    self.CRITERION_PARAM, 'gini') or 'gini'
            self.splitter = parameters.get(
                    self.SPLITTER_PARAM, 'best') or 'best'
            max_features_ = parameters.get(self.MAX_FEATURES_PARAM, None)
            self.max_features = max_features_
            self.max_leaf_nodes = parameters.get(
                    self.MAX_LEAF_NODES_PARAM, None) or None
            self.min_impurity_decrease = float(parameters.get(
                    self.MIN_IMPURITY_DECREASE_PARAM, 0) or 0)
            self.class_weight = parameters.get(
                    self.CLASS_WEIGHT_PARAM, None) or None

            vals = [self.min_split, self.min_leaf]
            atts = [self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import(
                    'from sklearn.tree import DecisionTreeClassifier')
            self.input_treatment()

    def input_treatment(self):
        if self.min_weight < 0 or self.min_weight > 0.5:
            raise ValueError(
                _("Parameter '{}' must be x>=0 or x<=0.5 for task {}").format(
                    self.MIN_WEIGHT_PARAM, self.__class__))

        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError(
                _("Parameter '{}' must be x>0 for task {}").format(
                    self.MAX_DEPTH_PARAM, self.__class__))

        if self.min_split <= 1:
            raise ValueError(
                _("Parameter '{}' must be x>1 for task {}").format(
                    self.MIN_SPLIT_PARAM, self.__class__))

        if self.min_impurity_decrease < 0:
            raise ValueError(
                _("Parameter '{}' must be x>=0 for task {}").format(
                    self.MIN_IMPURITY_DECREASE_PARAM, self.__class__))

        if self.max_leaf_nodes is not None and self.max_leaf_nodes != '0':
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        else:
            self.max_leaf_nodes = None

        if self.seed is not None and self.seed != '0':
            self.seed = int(self.seed)
        else:
            self.seed = None

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{max_features}', classification_model.max_features_], 
        ['{n_classes}', classification_model.n_classes_],
        ['{n_outputs}', classification_model.n_outputs_]
        """.format(max_features=gettext('Number of features'),
                   n_classes=gettext('Number of classes'),
                   n_outputs=gettext('The number of outputs'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = DecisionTreeClassifier(max_depth={max_depth}, 
            min_samples_split={min_split}, 
            min_samples_leaf={min_leaf}, 
            min_weight_fraction_leaf={min_weight}, 
            random_state={seed}, criterion='{criterion}', 
            splitter='{splitter}', max_features={max_features},
            max_leaf_nodes={max_leaf_nodes}, 
            min_impurity_decrease={min_impurity_decrease}, 
            class_weight={class_weight})
        """.format(min_split=self.min_split,
                   min_leaf=self.min_leaf,
                   min_weight=self.min_weight,
                   seed=self.seed,
                   max_depth=self.max_depth,
                   criterion=self.criterion,
                   splitter=self.splitter,
                   max_features=self.max_features,
                   max_leaf_nodes=self.max_leaf_nodes,
                   min_impurity_decrease=self.min_impurity_decrease,
                   class_weight=self.class_weight)
        return code


class GBTClassifierOperation(Operation):
    LEARNING_RATE_PARAM = 'learning_rate'
    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    LOSS_PARAM = 'loss'
    SEED_PARAM = 'random_state'
    SUBSAMPLE_PARAM = 'subsample'
    CRITERION_PARAM = 'criterion'
    MIN_WEIGHT_FRACTION_LEAF_PARAM = 'min_weight_fraction_leaf'
    MIN_IMPURITY_DECREASE_PARAM = 'min_impurity_decrease'
    INIT_PARAM = 'init'
    MAX_FEATURES_PARAM = 'max_features'
    MAX_LEAF_NODES_PARAM = 'max_leaf_nodes'
    VALIDATION_FRACTION_PARAM = 'validation_fraction'
    N_ITER_NO_CHANGE_PARAM = 'n_iter_no_change'
    TOL_PARAM = 'tol'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'

    LOSS_PARAM_DEV = 'deviance'
    LOSS_PARAM_EXP = 'exponential'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:
            self.max_depth = int(parameters.get(self.MAX_DEPTH_PARAM, 3) or 3)
            self.min_split = int(parameters.get(self.MIN_SPLIT_PARAM, 2) or 2)
            self.min_leaf = int(parameters.get(self.MIN_LEAF_PARAM, 1) or 1)
            self.n_estimators = int(parameters.get(
                    self.N_ESTIMATORS_PARAM, 100) or 100)
            self.learning_rate = float(parameters.get(
                    self.LEARNING_RATE_PARAM, 0.1)) or 0.1
            self.loss = \
                parameters.get(self.LOSS_PARAM, self.LOSS_PARAM_DEV) or \
                self.LOSS_PARAM_DEV
            self.seed = parameters.get(self.SEED_PARAM, None) or None
            self.subsample = float(
                parameters.get(self.SUBSAMPLE_PARAM, 1.0)) or 1.0
            self.criterion = parameters.get(
                    self.CRITERION_PARAM, 'friedman_mse') or 'friedman_mse'
            self.min_weight_leaf = float(parameters.get(
                    self.MIN_WEIGHT_FRACTION_LEAF_PARAM, 0) or 0)
            self.min_impurity_decrease = float(parameters.get(
                    self.MIN_IMPURITY_DECREASE_PARAM, 0) or 0)
            self.init = parameters.get(self.INIT_PARAM, 'None') or 'None'
            max_features_ = parameters.get(self.MAX_FEATURES_PARAM, None)
            self.max_features = None \
                if max_features_ is None else "'" + max_features_ + "'"
            max_leaf_nodes_ = parameters.get(self.MAX_LEAF_NODES_PARAM, None)
            self.max_leaf_nodes = None if max_leaf_nodes_ is None else int(
                    max_leaf_nodes_)
            self.validation_fraction = float(
                    parameters.get(self.VALIDATION_FRACTION_PARAM, 0.1) or 0.1)
            n_iter_no_change_ = parameters.get(self.N_ITER_NO_CHANGE_PARAM,
                                               None)
            self.n_iter_no_change = None if n_iter_no_change_ is None else int(
                    n_iter_no_change_)
            self.tol = float(parameters.get(self.TOL_PARAM, 1e-4) or 1e-4)

            vals = [self.min_split, self.min_leaf, self.learning_rate,
                    self.n_estimators, self.max_depth]
            atts = [self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM,
                    self.LEARNING_RATE_PARAM, self.N_ESTIMATORS_PARAM,
                    self.MAX_DEPTH_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import(
                    "from sklearn.ensemble import GradientBoostingClassifier")
            self.input_treatment()

    def input_treatment(self):
        if self.max_leaf_nodes is not None and self.max_leaf_nodes <= 1:
            raise ValueError(
                _("Parameter '{}' must be None or x > 1 for task {}").format(
                    self.MAX_LEAF_NODES_PARAM, self.__class__))

        if self.seed is not None and self.seed != '0':
            self.seed = int(self.seed)
        else:
            self.seed = None

        if self.n_iter_no_change is not None and self.n_iter_no_change <= 0:
            raise ValueError(
                _("Parameter '{}' must be None or x > 0 for task {}").format(
                    self.N_ITER_NO_CHANGE_PARAM, self.__class__))

        if self.validation_fraction < 0 or self.validation_fraction > 1:
            raise ValueError(
                _("Parameter '{}' must be 0 <= x =< 1 for task {}").format(
                    self.VALIDATION_FRACTION_PARAM, self.__class__))

        if self.subsample > 1.0 or self.subsample <= 0.0:
            raise ValueError(
                _("Parameter '{}' must be 0 < x =< 1 for task {}").format(
                    self.SUBSAMPLE_PARAM, self.__class__))

        if self.min_weight_leaf > 0.5 or self.min_weight_leaf < 0.0:
            raise ValueError(
                _("Parameter '{}' must be 0.0 <= x =< 0.5 for task {}").format(
                    self.MIN_WEIGHT_FRACTION_LEAF_PARAM, self.__class__))

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{n_est}', classification_model.n_estimators_], 
        ['{n_features}', classification_model.n_features_],
        ['{n_classes}', classification_model.n_classes_],
        ['{max}', classification_model.max_features_]
        """.format(n_features=gettext('The number of data features'),
                   n_est=gettext('The number of estimators'),
                   n_classes=gettext('Number of classes'),
                   max=gettext('The inferred value of max_features'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = GradientBoostingClassifier(loss='{loss}', 
            learning_rate={learning_rate}, 
            n_estimators={n_estimators}, min_samples_split={min_split},
            max_depth={max_depth}, min_samples_leaf={min_leaf}, 
            random_state={seed}, subsample={subsample}, 
            criterion='{criterion}', 
            min_weight_fraction_leaf={min_weight_fraction_leaf}, 
            min_impurity_decrease={min_impurity_decrease}, init={init},
            max_features={max_features},
            max_leaf_nodes={max_leaf_nodes}, warm_start=False, 
            validation_fraction={validation_fraction}, 
            n_iter_no_change={n_iter_no_change}, tol={tol})
        """.format(loss=self.loss,
                   n_estimators=self.n_estimators,
                   min_leaf=self.min_leaf,
                   min_split=self.min_split,
                   learning_rate=self.learning_rate,
                   max_depth=self.max_depth,
                   seed=self.seed,
                   subsample=self.subsample,
                   criterion=self.criterion,
                   min_weight_fraction_leaf=self.min_weight_leaf,
                   min_impurity_decrease=self.min_impurity_decrease,
                   init=self.init,
                   max_features=self.max_features,
                   max_leaf_nodes=self.max_leaf_nodes,
                   validation_fraction=self.validation_fraction,
                   n_iter_no_change=self.n_iter_no_change,
                   tol=self.tol)
        return code


class KNNClassifierOperation(Operation):

    K_PARAM = 'n_neighbors'
    WEIGHTS_PARAM = 'weights'
    ALGORITHM_PARAM = 'algorithm'
    LEAF_SIZE_PARAM = 'leaf_size'
    P_PARAM = 'p'
    METRIC_PARAM = 'metric'
    METRIC_PARAMS_PARAM = 'metric_params'
    N_JOBS_PARAM = 'n_jobs'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:
            self.n_neighbors = int(self.parameters.get(self.K_PARAM, 5)) or 5
            self.weights = self.parameters.get(self.WEIGHTS_PARAM,
                                               'uniform') or 'uniform'
            self.algorithm = self.parameters.get(self.ALGORITHM_PARAM,
                                                 'auto') or 'auto'
            self.leaf_size = int(
                self.parameters.get(self.LEAF_SIZE_PARAM, 30) or 30)
            self.p = int(self.parameters.get(self.P_PARAM, 2) or 2)
            self.metric = self.parameters.get(self.METRIC_PARAM,
                                              'minkowski') or 'minkowski'
            self.metric_params = self.parameters.get(self.METRIC_PARAMS_PARAM,
                                                     None) or None
            self.n_jobs = self.parameters.get(self.N_JOBS_PARAM, None) or None

            if self.n_neighbors <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                                self.K_PARAM, self.__class__))

            self.transpiler_utils.add_import(
                    "from sklearn.neighbors import KNeighborsClassifier")
            self.input_treatment()

    def input_treatment(self):
        if self.n_jobs is not None:
            self.n_jobs = int(self.n_jobs)
        else:
            self.n_jobs = None

        if self.p <= 1 and self.metric in ['minkowski', 'wminkowski']:
            raise ValueError(
                        _("Parameter '{}' must be x>1 when parameter '{}' "
                          "is 'minkowski' for task {}").format(
                                self.P_PARAM, self.METRIC_PARAMS_PARAM,
                                self.__class__))

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{n_samples}', classification_model.n_samples_fit_]
        """.format(n_samples=gettext('Number of samples in the fitted data'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = KNeighborsClassifier(n_neighbors={n_neighbors}, 
            weights='{weights}', algorithm='{algorithm}', 
            leaf_size={leaf_size}, p={p}, metric='{metric}', 
            metric_params={metric_params}, n_jobs={n_jobs})
        """.format(n_neighbors=self.n_neighbors, weights=self.weights,
                   algorithm=self.algorithm, leaf_size=self.leaf_size,
                   p=self.p, metric=self.metric,
                   metric_params=self.metric_params, n_jobs=self.n_jobs)
        return code


class LogisticRegressionOperation(Operation):
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'
    TOLERANCE_PARAM = 'tol'
    REGULARIZATION_PARAM = 'regularization'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'random_state'
    SOLVER_PARAM = 'solver'
    PENALTY_PARAM = 'penalty'
    DUAL_PARAM = 'dual'
    FIT_INTERCEPT_PARAM = 'fit_intercept'
    INTERCEPT_SCALING_PARAM = 'intercept_scaling'
    MULTI_CLASS_PARAM = 'multi_class'
    N_JOBS_PARAM = 'n_jobs'
    L1_RATIO_PARAM = 'l1_ratio'

    SOLVER_PARAM_NEWTON = 'newton-cg'
    SOLVER_PARAM_LBFGS = 'lbfgs'
    SOLVER_PARAM_LINEAR = 'liblinear'
    SOLVER_PARAM_SAG = 'sag'
    SOLVER_PARAM_SAGa = 'saga'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.has_code:
            if self.LABEL_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.LABEL_PARAM,
                    self.__class__))
            else:
                self.label = parameters.get(self.LABEL_PARAM, None)

            if self.FEATURES_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM,
                    self.__class__))
            else:
                self.features = parameters.get(self.FEATURES_PARAM, None)

            self.prediction_column = parameters.get(
                    self.PREDICTION_PARAM, 'prediction')

            self.tol = float(self.parameters.get(
                    self.TOLERANCE_PARAM, 0.0001) or 0.0001)
            if self.tol <= 0:
                raise ValueError(
                    _("Parameter '{}' must be x>0 for task {}").format(
                        self.TOLERANCE_PARAM, self.__class__))

            self.regularization = float(self.parameters.get(
                    self.REGULARIZATION_PARAM, 1.0)) or 1.0
            self.max_iter = int(self.parameters.get(
                    self.MAX_ITER_PARAM, 100)) or 100

            seed_ = self.parameters.get(self.SEED_PARAM, None)
            self.seed = int(seed_) if seed_ is not None else 'None'

            self.solver = self.parameters.get(
                    self.SOLVER_PARAM, self.SOLVER_PARAM_LINEAR)\
                or self.SOLVER_PARAM_LINEAR

            self.penalty = parameters.get(self.PENALTY_PARAM, 'l2')

            self.dual = int(parameters.get(self.DUAL_PARAM, 0)) == 1
            self.fit_intercept = int(parameters.get(
                    self.FIT_INTERCEPT_PARAM, 1)) == 1
            self.intercept_scaling = float(parameters.get(
                    self.INTERCEPT_SCALING_PARAM, 1.0))
            if self.fit_intercept and self.intercept_scaling <= 0 \
                    and self.solver == 'liblinear':
                raise ValueError(
                        _("Parameter '{}' must be x>0 for task {}").format(
                            self.INTERCEPT_SCALING_PARAM, self.__class__))

            n_jobs_ = parameters.get(self.N_JOBS_PARAM, None)
            if n_jobs_ is not None:
                self.n_jobs = int(n_jobs_)
                if self.n_jobs <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    self.N_JOBS_PARAM, self.__class__))
            else:
                self.n_jobs = 'None'

            self.multi_class = parameters.get(
                    self.MULTI_CLASS_PARAM, 'ovr') or 'ovr'

            vals = [self.regularization, self.max_iter]
            atts = [self.REGULARIZATION_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            solver_dict = {
                'newton-cg': ['l2', 'none'],
                'lbfgs': ['l2', 'none'],
                'liblinear': ['l1', 'l2'],
                'sag': ['l2', 'none'],
                'saga': ['l2', 'none', 'l1', 'elasticnet']
            }
            if self.penalty not in solver_dict[self.solver]:
                raise ValueError(
                        _(
                            "For '{}' solver, the penalty type must be in {} "
                            "for task {}").format(
                                self.solver, str(solver_dict[self.solver]),
                                self.__class__))

            if self.solver == 'newton-cg' and self.dual == True:
                raise ValueError(
                        _(
                            "For '{}' solver supports only dual=False for "
                            "task {}").format(
                                self.solver, self.__class__))

            if self.solver == 'liblinear' and self.multi_class == 'multinomial':
                raise ValueError(
                        _(
                            'Parameter "{}" does not support {'
                            '}="multinomial"').format(
                                self.SOLVER_PARAM, self.MULTI_CLASS_PARAM,
                                self.__class__))

            l1_ratio_param_ = parameters.get(self.L1_RATIO_PARAM, None)
            if l1_ratio_param_ is not None:
                self.l1_ratio = float(l1_ratio_param_)
                if self.penalty == 'elasticnet' and \
                        (self.l1_ratio < 0 or self.l1_ratio > 1):
                    raise ValueError(
                    _("Parameter 'l1_ratio' must be 0 <= x <= 1 for task {}")
                    .format(self.__class__))
            elif self.penalty == 'elasticnet' and l1_ratio_param_ is None:
                raise ValueError(
                    _("Parameter 'l1_ratio' must be 0 <= x <= 1 for task {}")
                    .format(self.__class__))
            else:
                self.l1_ratio = 'None'

            self.transpiler_utils.add_import(
                    "from sklearn.linear_model import LogisticRegression")

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{coef}', classification_model.coef_],
        ['{intercept}', classification_model.intercept_],
        ['{n_iter}', classification_model.n_iter_]
        """.format(coef=gettext('Coefficient'),
                   intercept=gettext('Intercept'),
                   n_iter=gettext('Actual number of iterations'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = LogisticRegression(tol={tol}, C={C}, 
            max_iter={max_iter}, solver='{solver}', random_state={seed}, 
            penalty='{penalty}',  dual={dual}, 
            fit_intercept={fit_intercept}, 
            intercept_scaling={intercept_scaling},
            multi_class='{multi_class}', n_jobs={n_jobs}, 
            l1_ratio={l1_ratio})
        """.format(tol=self.tol, C=self.regularization,
                   max_iter=self.max_iter, seed=self.seed,
                   solver=self.solver, penalty=self.penalty,
                   dual=self.dual, fit_intercept=self.fit_intercept,
                   intercept_scaling=self.intercept_scaling,
                   multi_class=self.multi_class,
                   n_jobs=self.n_jobs, l1_ratio=self.l1_ratio)
        return code


class MLPClassifierOperation(Operation):

    HIDDEN_LAYER_SIZES_PARAM = 'hidden_layer_sizes'
    ACTIVATION_PARAM = 'activation'
    SOLVER_PARAM = 'solver'
    ALPHA_PARAM = 'alpha'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'seed'
    BATCH_SIZE_PARAM = 'batch_size'
    LEARNING_RATE_PRAM = 'learning_rate'
    LEARNING_RATE_INIT_PRAM = 'learning_rate_init'
    POWER_T_PARAM = 'power_t'
    SHUFFLE_PARAM = 'shuffle'
    MOMENTUM_PARAM = 'momentum'
    NESTEROVS_MOMENTUM_PARAM = 'nesterovs_momentum'
    EARLY_STOPPING_PARAM = 'early_stopping'
    VALIDATION_FRACTION_PARAM = 'validation_fraction'
    BETA_1_PARAM = 'beta1'
    BETA_2_PARAM = 'beta2'
    EPSILON_PARAM = 'epsilon'
    N_ITER_NO_CHANGE_PARAM = 'n_iter_no_change'
    MAX_FUN_PARAM = 'max_fun'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'

    SOLVER_PARAM_ADAM = 'adam'
    SOLVER_PARAM_LBFGS = 'lbfgs'
    SOLVER_PARAM_SGD = 'sgd'

    ACTIVATION_PARAM_ID = 'identity'
    ACTIVATION_PARAM_LOG = 'logistic'
    ACTIVATION_PARAM_TANH = 'tanh'
    ACTIVATION_PARAM_RELU = 'relu'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:
            self.add_functions_required = ""
            self.hidden_layers = parameters.get(self.HIDDEN_LAYER_SIZES_PARAM,
                                                '(100,1)') or '(100,1)'
            self.hidden_layers = \
                self.hidden_layers.replace("(", "").replace(")", "")
            self.activation = parameters.get(
                    self.ACTIVATION_PARAM,
                    self.ACTIVATION_PARAM_RELU) or self.ACTIVATION_PARAM_RELU
            self.solver = parameters.get(
                    self.SOLVER_PARAM,
                    self.SOLVER_PARAM_ADAM) or self.SOLVER_PARAM_ADAM
            self.alpha = float(parameters.get(self.ALPHA_PARAM,
                                              0.0001)) or 0.0001
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 200)) or 200
            self.tol = float(parameters.get(self.TOLERANCE_PARAM,
                                            0.0001)) or 0.0001
            self.seed = parameters.get(self.SEED_PARAM, None) or None

            self.batch_size = parameters.get(self.BATCH_SIZE_PARAM, 'auto') \
                or 'auto'
            self.learning_rate = parameters.get(self.LEARNING_RATE_PRAM,
                                                'constant') or 'constant'
            self.learning_rate_init = float(parameters.get(
                    self.LEARNING_RATE_INIT_PRAM, 0.001) or 0.001)
            self.power_t = float(parameters.get(self.POWER_T_PARAM, 0.5) or 0.5)
            self.shuffle = int(parameters.get(self.SHUFFLE_PARAM, 1)) == 1
            self.momentum = float(parameters.get(
                    self.MOMENTUM_PARAM, 0.9) or 0.9)
            self.nesterovs_momentum = int(parameters.get(
                    self.NESTEROVS_MOMENTUM_PARAM, 1)) == 1
            self.early_stopping = int(parameters.get(
                    self.EARLY_STOPPING_PARAM, 0)) == 1
            self.validation_fraction = float(parameters.get(
                    self.VALIDATION_FRACTION_PARAM, 0.1) or 0.1)
            self.beta1 = float(parameters.get(self.BETA_1_PARAM, 0.9) or 0.9)
            self.beta2 = float(parameters.get(
                    self.BETA_2_PARAM, 0.999) or 0.999)
            self.epsilon = float(parameters.get(self.EPSILON_PARAM,
                                                1e-8) or 1e-8)
            self.n_iter_no_change = int(parameters.get(
                    self.N_ITER_NO_CHANGE_PARAM, 10) or 10)
            self.max_fun = int(parameters.get(
                    self.MAX_FUN_PARAM, 15000) or 15000)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM,
                                                  'prediction')

            self.transpiler_utils.add_import(
                    "from sklearn.neural_network import MLPClassifier")
            self.input_treatment()

    def input_treatment(self):
        if self.tol <= 0:
            raise ValueError(
                _("Parameter '{}' must be x > 0 for task {}").format(
                    self.TOLERANCE_PARAM, self.__class__))

        if self.max_iter <= 0:
            raise ValueError(
                _("Parameter '{}' must be x > 0 for task {}").format(
                    self.MAX_ITER_PARAM, self.__class__))

        if self.alpha < 0:
            raise ValueError(
                _("Parameter '{}' must be x >= 0 for task {}").format(
                    self.ALPHA_PARAM, self.__class__))

        if self.seed is not None:
            self.seed = int(self.seed)

        if self.batch_size != 'auto':
            self.batch_size = int(self.batch_size)
        else:
            self.batch_size = "'" + self.batch_size + "'"

        if not bool(re.match('(\d+,)+\d*', self.hidden_layers)):
            raise ValueError(
                    _("Parameter '{}' must be a tuple with the size "
                      "of each layer for task {}").format(
                            self.HIDDEN_LAYER_SIZES_PARAM, self.__class__))

        self.hidden_layers = tuple([int(i) for i in
                                    self.hidden_layers.replace(' ', '').split(
                                        ',')])
        functions_required = ["""hidden_layer_sizes={hidden_layers}""".format(
            hidden_layers=self.hidden_layers)]

        self.activation = """activation='{activation}'""".format(
                activation=self.activation)
        functions_required.append(self.activation)

        self.solver_ = """solver='{solver}'""".format(solver=self.solver)
        functions_required.append(self.solver_)

        self.alpha = """alpha={alpha}""".format(alpha=self.alpha)
        functions_required.append(self.alpha)

        self.max_iter = """max_iter={max_iter}""".format(max_iter=self.max_iter)
        functions_required.append(self.max_iter)

        self.tol = """tol={tol}""".format(tol=self.tol)
        functions_required.append(self.tol)

        self.seed = """random_state={seed}""".format(seed=self.seed)
        functions_required.append(self.seed)

        if self.solver != 'lbfgs':
            self.batch_size = """batch_size={batch_size}""" \
                .format(batch_size=self.batch_size)
            functions_required.append(self.batch_size)

        if self.solver == 'sgd':
            self.learning_rate = """learning_rate='{learning_rate}'"""\
                .format(learning_rate=self.learning_rate)
            functions_required.append(self.learning_rate)

            self.nesterovs_momentum = \
                """nesterovs_momentum={nesterovs_momentum}"""\
                .format(nesterovs_momentum=self.nesterovs_momentum)
            functions_required.append(self.nesterovs_momentum)

            self.power_t = """power_t={power_t}""".format(power_t=self.power_t)
            functions_required.append(self.power_t)

            if self.momentum < 0 or self.momentum > 1:
                raise ValueError(
                    _("Parameter '{}' must be x between 0 and 1 for task {}")
                    .format(self.MOMENTUM_PARAM, self.__class__))
            else:
                self.momentum = """momentum={momentum}""".format(
                    momentum=self.momentum)
                functions_required.append(self.momentum)

        if self.solver == 'sgd' or self.solver == 'adam':
            if self.learning_rate_init <= 0:
                raise ValueError(
                    _("Parameter '{}' must be x > 0 for task {}").format(
                        self.LEARNING_RATE_INIT_PRAM, self.__class__))
            else:
                self.learning_rate_init = \
                    """learning_rate_init={learning_rate_init}"""\
                    .format(learning_rate_init=self.learning_rate_init)
                functions_required.append(self.learning_rate_init)

            self.shuffle = """shuffle={shuffle}""".format(shuffle=self.shuffle)
            functions_required.append(self.shuffle)

            self.early_stopping = """early_stopping={early_stopping}""".format(
                early_stopping=self.early_stopping)
            functions_required.append(self.early_stopping)

            if self.n_iter_no_change <= 0:
                raise ValueError(
                        _("Parameter '{}' must be x > 0 for task {}").format(
                                self.N_ITER_NO_CHANGE_PARAM, self.__class__))
            else:
                self.n_iter_no_change = \
                    """n_iter_no_change={n_iter_no_change}"""\
                    .format(n_iter_no_change=self.n_iter_no_change)
                functions_required.append(self.n_iter_no_change)

        if self.early_stopping:
            if self.validation_fraction < 0 or self.validation_fraction > 1:
                raise ValueError(
                        _(
                            "Parameter '{}' must be x between 0 and 1 for "
                            "task {}").format(
                                self.VALIDATION_FRACTION_PARAM, self.__class__))
            else:
                self.validation_fraction = \
                    """validation_fraction={validation_fraction}"""\
                        .format(validation_fraction=self.validation_fraction)
                functions_required.append(self.validation_fraction)

        if self.solver == 'adam':
            if self.beta1 < 0 or self.beta1 >= 1:
                raise ValueError(
                    _("Parameter '{}' must be in [0, 1) for task {}").format(
                        self.BETA_1_PARAM, self.__class__))
            else:
                self.beta1 = """beta_1={beta1}""".format(beta1=self.beta1)
                functions_required.append(self.beta1)

            if self.beta2 < 0 or self.beta2 >= 1:
                raise ValueError(
                    _("Parameter '{}' must be in [0, 1) for task {}").format(
                        self.BETA_2_PARAM, self.__class__))
            else:
                self.beta2 = """beta_2={beta2}""".format(beta2=self.beta2)
                functions_required.append(self.beta2)

            self.epsilon = """epsilon={epsilon}""".format(epsilon=self.epsilon)
            functions_required.append(self.epsilon)

        if self.solver == 'lbfgs':
            if self.max_fun <= 0:
                raise ValueError(
                    _("Parameter '{}' must be x > 0 for task {}").format(
                        self.MAX_FUN_PARAM, self.__class__))
            else:
                self.max_fun = """max_fun={}""".format(self.max_fun)
                functions_required.append(self.max_fun)

        self.add_functions_required = ',\n    '.join(functions_required)

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{loss}', classification_model.loss_],
        ['{best_loss}', classification_model.best_loss_],
        ['{t}', classification_model.t_],
        ['{n_iter}', classification_model.n_iter_],
        ['{n_layers}', classification_model.n_layers_],
        ['{activation}', classification_model.out_activation_],
        ['{n_output}', classification_model.n_outputs_]
        """.format(loss=gettext('Current loss'),
                   best_loss=gettext('The minimum loss'),
                   t=gettext('The number of training samples'),
                   n_iter=gettext('Actual number of iterations'),
                   n_layers=gettext('Number of layers'),
                   n_output=gettext('Number of outputs'),
                   activation=gettext('Output activation function'),
                   )
        return code

    def generate_code(self):
        """Generate code."""

        code = """
        algorithm = MLPClassifier({add_functions_required})
        """.format(add_functions_required=self.add_functions_required)
        return code


class NaiveBayesClassifierOperation(Operation):

    ALPHA_PARAM = 'alpha'
    CLASS_PRIOR_PARAM = 'class_prior'
    FIT_PRIOR_PARAM = 'fit_prior'
    VAR_SMOOTHING_PARAM = 'var_smoothing'
    PRIORS_PARAM = 'priors'
    BINARIZE_PARAM = 'binarize'
    MODEL_TYPE_PARAM = 'type'
    PREDICTION_PARAM = 'prediction'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'

    MODEL_TYPE_PARAM_B = 'Bernoulli'
    MODEL_TYPE_PARAM_G = 'GaussianNB'
    MODEL_TYPE_PARAM_M = 'Multinomial'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:
            self.class_prior = parameters.get(self.CLASS_PRIOR_PARAM, 'None') \
                               or 'None'
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 1.0) or 1.0)
            self.fit_prior = int(parameters.get(self.FIT_PRIOR_PARAM, 1))
            self.var_smoothing = float(parameters.get(self.VAR_SMOOTHING_PARAM,
                                                      1e-9) or 1e-9)
            self.priors = parameters.get(self.PRIORS_PARAM, 'None') or 'None'
            self.binarize = float(parameters.get(self.BINARIZE_PARAM, 0) or 0)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM,
                                                  'prediction')
            self.smoothing = float(parameters.get(self.ALPHA_PARAM, 1.0) or 1.0)
            self.model_type = parameters.get(
                    self.MODEL_TYPE_PARAM,
                    self.MODEL_TYPE_PARAM_M) or self.MODEL_TYPE_PARAM_M

            if self.model_type == self.MODEL_TYPE_PARAM_M:
                self.transpiler_utils.add_import(
                        "from sklearn.naive_bayes import MultinomialNB")
            elif self.model_type == self.MODEL_TYPE_PARAM_B:
                self.transpiler_utils.add_import(
                        "from sklearn.naive_bayes import BernoulliNB")
            else:
                self.transpiler_utils.add_import(
                        "from sklearn.naive_bayes import GaussianNB")
            self.input_treatment()

    def input_treatment(self):
        self.fit_prior = True if int(self.fit_prior) == 1 else False

        if self.class_prior != "None":
            self.class_prior = '[' + self.class_prior + ']'

        if self.priors != "None":
            self.priors = '[' + self.priors + ']'

        if self.smoothing <= 0 and \
           self.model_type in [self.MODEL_TYPE_PARAM_M,
                               self.MODEL_TYPE_PARAM_B]:
            raise ValueError(
                _("Parameter '{}' must be x>0 for task {}").format(
                    self.ALPHA_PARAM, self.__class__))

    def get_output_metrics_code(self):
        if self.model_type != self.MODEL_TYPE_PARAM_G:
            code = """
            ['{class_count}', classification_model.class_count_],
            ['{class_log_prior}', classification_model.class_log_prior_],
            ['{n_features}', classification_model.n_features_],
            ['{coef}', classification_model.coef_],
            ['{intercept}', classification_model.intercept_]
            """.format(coef=gettext('Coefficient'),
                       intercept=gettext('Intercept'),
                       class_count=gettext('Number of samples '
                                           'encountered for each class'),
                       class_log_prior=gettext('Smoothed empirical log '
                                               'probability'),
                       n_features=gettext('Number of features of each '
                                          'sample'))
        else:
            code = """
            ['{class_count}', classification_model.class_count_],
            ['{class_prior}', classification_model.class_prior_],
            ['{epsilon}', classification_model.epsilon_],
            ['{sigma}', classification_model.sigma_],
            ['{theta}', classification_model.theta_]
            """.format(epsilon=gettext('Absolute additive value to '
                                       'variances'),
                       sigma=gettext('Variance of each feature per class'),
                       class_count=gettext('Number of samples '
                                           'encountered for '
                                           'each class'),
                       class_prior=gettext('Probability of each class'),
                       theta=gettext('Mean of each feature per class'))

        return code

    def generate_code(self):
        """Generate code."""

        if self.model_type == self.MODEL_TYPE_PARAM_M:
            code = """
            algorithm = MultinomialNB(alpha={alpha}, class_prior={class_prior}, 
                fit_prior={fit_prior})
            """.format(class_prior=self.class_prior, alpha=self.alpha,
                       fit_prior=self.fit_prior)
        elif self.model_type == self.MODEL_TYPE_PARAM_B:
            code = """
            algorithm = BernoulliNB(alpha={alpha}, class_prior={class_prior}, 
                fit_prior={fit_prior}, binarize={binarize})
            """.format(alpha=self.alpha, class_prior=self.class_prior,
                       fit_prior=self.fit_prior, binarize=self.binarize)
        else:
            code = """
            algorithm = GaussianNB(priors={priors}, var_smoothing={smoothing})
            """.format(priors=self.priors, smoothing=self.var_smoothing)

        return code


class PerceptronClassifierOperation(Operation):

    ALPHA_PARAM = 'alpha'
    TOLERANCE_PARAM = 'tol'
    SHUFFLE_PARAM = 'shuffle'
    SEED_PARAM = 'seed'
    PENALTY_PARAM = 'penalty'
    MAX_ITER_PARAM = 'max_iter'
    FIT_INTERCEPT_PARAM = 'fit_intercept'
    ETA0_PARAM = 'eta0'
    N_JOBS_PARAM = 'n_jobs'
    EARLY_STOPPING_PARAM = 'early_stopping'
    VALIDATION_FRACTION_PARAM = 'validation_fraction'
    N_ITER_NO_CHANGE_PARAM = 'n_iter_no_change'
    CLASS_WEIGHT_PARAM = 'class_weight'
    PREDICTION_PARAM = 'prediction'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'

    PENALTY_PARAM_EN = 'elasticnet'
    PENALTY_PARAM_L1 = 'l1'
    PENALTY_PARAM_L2 = 'l2'
    PENALTY_PARAM_NONE = 'None'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:
            self.max_iter = int(parameters.get(
                    self.MAX_ITER_PARAM, 1000) or 1000)
            self.alpha = float(parameters.get(
                    self.ALPHA_PARAM, 0.0001) or 0.0001)
            self.tol = float(parameters.get(
                    self.TOLERANCE_PARAM, 0.001)) or 0.001
            self.shuffle = int(parameters.get(self.SHUFFLE_PARAM, 0) or 0)
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.penalty = parameters.get(
                    self.PENALTY_PARAM,
                    self.PENALTY_PARAM_NONE) or self.PENALTY_PARAM_NONE
            self.fit_intercept = int(parameters.get(
                    self.FIT_INTERCEPT_PARAM, 1) or 1)
            self.eta0 = float(parameters.get(self.ETA0_PARAM, 1) or 1)
            self.n_jobs = parameters.get(self.N_JOBS_PARAM, None) or None
            self.early_stopping = int(parameters.get(
                    self.EARLY_STOPPING_PARAM, 0) or 0)
            self.validation_fraction = float(parameters.get(
                    self.VALIDATION_FRACTION_PARAM, 0.1) or 0.1)
            self.n_iter_no_change = int(parameters.get(
                    self.N_ITER_NO_CHANGE_PARAM, 5) or 5)
            self.class_weight = parameters.get(
                    self.CLASS_WEIGHT_PARAM, None) or None

            vals = [self.max_iter, self.alpha]
            atts = [self.MAX_ITER_PARAM, self.ALPHA_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import(
                    "from sklearn.linear_model import Perceptron")
            self.input_treatment()

    def input_treatment(self):
        if self.n_jobs is not None and self.n_jobs != '0':
            self.n_jobs = int(self.n_jobs)
        else:
            self.n_jobs = None

        self.shuffle = True if int(self.shuffle) == 1 else False
        self.early_stopping = True if int(self.early_stopping) == 1 else False

        if self.validation_fraction < 0 or self.validation_fraction > 1:
            raise ValueError(
                _("Parameter '{}' must be 0 <= x =< 1 for task {}").format(
                    self.VALIDATION_FRACTION_PARAM, self.__class__))

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{t}', classification_model.t_],
        ['{coef}', classification_model.coef_],
        ['{intercept}', classification_model.intercept_],
        ['{n_iter}', classification_model.n_iter_]
        """.format(coef=gettext('Coefficient'),
                   intercept=gettext('Intercept'),
                   loss=gettext('Current loss'),
                   best_loss=gettext('The minimum loss'),
                   t=gettext('The number of training samples'),
                   n_iter=gettext('Actual number of iterations'),
                   n_layers=gettext('Number of layers'),
                   n_output=gettext('Number of outputs'),
                   activation=gettext('Output activation function'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = Perceptron(tol={tol}, alpha={alpha}, 
            max_iter={max_iter}, shuffle={shuffle}, random_state={seed}, 
            penalty='{penalty}', fit_intercept={fit_intercept}, 
            eta0={eta0}, n_jobs={n_jobs}, early_stopping={early_stopping}, 
            validation_fraction={validation_fraction}, 
            n_iter_no_change={n_iter_no_change}, 
            class_weight={class_weight}, warm_start=False)
        """.format(tol=self.tol,
                   alpha=self.alpha,
                   max_iter=self.max_iter,
                   shuffle=self.shuffle,
                   penalty=self.penalty,
                   seed=self.seed,
                   fit_intercept=self.fit_intercept,
                   eta0=self.eta0,
                   n_jobs=self.n_jobs,
                   early_stopping=self.early_stopping,
                   validation_fraction=self.validation_fraction,
                   n_iter_no_change=self.n_iter_no_change,
                   class_weight=self.class_weight)
        return code


class RandomForestClassifierOperation(Operation):
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'
    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    SEED_PARAM = 'seed'
    CRITERION_PARAM = 'criterion'
    MIN_WEIGHT_FRACTION_LEAF_PARAM = 'min_weight_fraction_leaf'
    MAX_FEATURES_PARAM = 'max_features'
    MAX_LEAF_NODES_PARAM = 'max_leaf_nodes'
    MIN_IMPURITY_DECREASE_PARAM = 'min_impurity_decrease'
    BOOTSTRAP_PARAM = 'bootstrap'
    OOB_SCORE_PARAM = 'oob_score'
    N_JOBS_PARAM = 'n_jobs'
    CCP_ALPHA_PARAM = 'ccp_alpha'
    MAX_SAMPLES_PARAM = 'max_samples'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:

            if self.FEATURES_PARAM not in parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                        self.FEATURES_PARAM,
                        self.__class__.__name__))
            else:
                self.features = parameters.get(self.FEATURES_PARAM, None)

            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.min_split = int(parameters.get(self.MIN_SPLIT_PARAM, 2) or 2)
            self.min_leaf = int(parameters.get(self.MIN_LEAF_PARAM, 1) or 1)

            self.max_depth = self.__positive_or_none_param(parameters,
                                                           self.MAX_DEPTH_PARAM)
            self.n_estimators = int(parameters.get(self.N_ESTIMATORS_PARAM,
                                                   10) or 10)

            self.criterion = parameters.get(self.CRITERION_PARAM, 'gini')

            self.min_weight_fraction_leaf = float(
                    parameters.get(self.MIN_WEIGHT_FRACTION_LEAF_PARAM, 0.0))
            if self.min_weight_fraction_leaf < 0.0 or \
                    self.min_weight_fraction_leaf > 0.5:
                raise ValueError(
                    _("Parameter '{}' must be x>=0.0 and x<=0.5 for task {}"
                      ).format(self.MIN_WEIGHT_FRACTION_LEAF_PARAM,
                               self.__class__))

            self.max_features = \
                self.__positive_or_none_param(parameters,
                                              self.MAX_FEATURES_PARAM)
            self.max_leaf_nodes = \
                self.__positive_or_none_param(parameters,
                                              self.MAX_LEAF_NODES_PARAM)

            self.min_impurity_decrease = float(parameters.get(
                    self.MIN_IMPURITY_DECREASE_PARAM, 0.0))

            self.bootstrap = int(parameters.get(self.BOOTSTRAP_PARAM, 1)) == 1
            self.oob_score = int(parameters.get(self.OOB_SCORE_PARAM, 0)) == 1

            self.n_jobs = parameters.get(self.N_JOBS_PARAM, None)

            self.ccp_alpha = float(parameters.get(self.CCP_ALPHA_PARAM, 0.0))

            max_samples_ = parameters.get(self.MAX_SAMPLES_PARAM, None)
            if max_samples_ is not None:
                max_samples_ = float(max_samples_)
                if max_samples_ <= 0.0 or max_samples_ >= 100.0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 and x<100, or empty, "
                              "case you want to use a fully sample for "
                              "task {}").format(self.MAX_SAMPLES_PARAM,
                                                self.__class__))
                else:
                    self.max_samples = max_samples_ / 100.0
            else:
                self.max_samples = 'None'

            if self.max_features != 'None' and \
                    (self.max_features <= 0 or
                     self.max_features > len(self.features)):
                raise ValueError(
                        _("Parameter '{}' must be x>0 and "
                          "x < n_features for task {}").format(
                                self.MAX_FEATURES_PARAM, self.__class__))

            vals = [self.min_impurity_decrease, self.ccp_alpha]
            atts = [self.MIN_IMPURITY_DECREASE_PARAM, self.CCP_ALPHA_PARAM]
            for var, att in zip(vals, atts):
                if var < 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>=0 for task {}").format(
                                    att, self.__class__))

            vals = [self.min_split, self.min_leaf, self.n_estimators]
            atts = [self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM,
                    self.N_ESTIMATORS_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import(
                    "from sklearn.ensemble import RandomForestClassifier")

    def __positive_or_none_param(self, parameters, param_name):
        param = parameters.get(param_name, None)
        if param is not None:
            returned_param = int(param)
            if returned_param < 0:
                raise ValueError(
                        _("Parameter '{}' must be x>=0  for task {}").format(
                                param_name, self.__class__))
        else:
            returned_param = 'None'
        return returned_param

    def get_output_metrics_code(self):
        code = """
        ['{n_classes}', classification_model.n_classes_],
        ['{n_features}', classification_model.n_features_],
        ['{n_outputs}', classification_model.n_outputs_],
        """.format(n_classes=gettext('The classes labels '),
                   n_features=gettext('The number of features'),
                   n_outputs=gettext('The number of outputs'))

        if self.oob_score:
            code += """
            ['{oob_score}', classification_model.oob_score_]
            """.format(oob_score=gettext('Out-of-bag score'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = RandomForestClassifier(n_estimators={n_estimators}, 
            max_depth={max_depth},  min_samples_split={min_split}, 
            min_samples_leaf={min_leaf}, random_state={seed},
            criterion='{criterion}', 
            min_weight_fraction_leaf={min_weight_fraction_leaf},
            max_features={max_features}, max_leaf_nodes={max_leaf_nodes}, 
            min_impurity_decrease={min_impurity_decrease}, n_jobs={n_jobs}, 
            bootstrap={bootstrap}, oob_score={oob_score}, 
            ccp_alpha={ccp_alpha}, max_samples={max_samples})
        """.format(n_estimators=self.n_estimators, max_depth=self.max_depth,
                   min_split=self.min_split, min_leaf=self.min_leaf,
                   seed=self.seed, criterion=self.criterion,
                   min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                   max_features=self.max_features, n_jobs=self.n_jobs,
                   max_leaf_nodes=self.max_leaf_nodes,
                   min_impurity_decrease=self.min_impurity_decrease,
                   bootstrap=self.bootstrap, oob_score=self.oob_score,
                   ccp_alpha=self.ccp_alpha, max_samples=self.max_samples)

        return code


class SvmClassifierOperation(Operation):
    PENALTY_PARAM = 'c'
    KERNEL_PARAM = 'kernel'
    DEGREE_PARAM = 'degree'
    TOLERANCE_PARAM = 'tol'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'
    PREDICTION_PARAM = 'prediction'
    GAMMA_PARAM = 'gamma'
    COEF0_PARAM = 'coef0'
    SHRINKING_PARAM = 'shrinking'
    PROBABILITY_PARAM = 'probability'
    CACHE_SIZE_PARAM = 'cache_size'
    DECISION_FUNCTION_SHAPE_PARAM = 'decision_function_shape'

    KERNEL_PARAM_LINEAR = 'linear'
    KERNEL_PARAM_RBF = 'rbf'
    KERNEL_PARAM_POLY = 'poly'
    KERNEL_PARAM_SIG = 'sigmoid'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:

            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, -1))
            self.tol = float(parameters.get(self.TOLERANCE_PARAM, 0.001) or
                             0.001)
            self.tol = abs(float(self.tol))
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
            self.degree = int(parameters.get(self.DEGREE_PARAM, 3) or 3)
            self.kernel = parameters.get(
                    self.KERNEL_PARAM,
                    self.KERNEL_PARAM_RBF) or self.KERNEL_PARAM_RBF
            self.c = float(parameters.get(self.PENALTY_PARAM, 1.0) or 1.0)

            self.gamma = parameters.get(self.GAMMA_PARAM, 'scale')

            self.coef0 = float(parameters.get(self.COEF0_PARAM, 0.0) or 0.0)
            self.shrinking = int(parameters.get(self.SHRINKING_PARAM, 1)) == 1
            self.probability = int(parameters.get(
                    self.PROBABILITY_PARAM, 0)) == 1
            self.decision_function_shape = parameters.get(
                    self.DECISION_FUNCTION_SHAPE_PARAM, 'ovr') or 'ovr'

            vals = [self.degree, self.c]
            atts = [self.DEGREE_PARAM, self.PENALTY_PARAM,
                    self.CACHE_SIZE_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import("from sklearn.svm import SVC")

    def get_output_metrics_code(self):
        code = """
        ['{intercept}', classification_model.intercept_],
        ['{class_weight}', classification_model.class_weight_],
        ['{classes}', classification_model.classes_]
        """.format(intercept=gettext('Intercept'),
                   classes=gettext('The classes labels'),
                   class_weight=gettext('Multipliers of parameter C '
                                        'for each class'))
        if self.kernel == self.KERNEL_PARAM_LINEAR:
            code += """
            ['{coef}', classification_model.coef_],
            """.format(coef=gettext('Coefficient'))
        return code

    def generate_code(self):
        """Generate code."""

        code = """
        algorithm = SVC(tol={tol}, C={c}, max_iter={max_iter}, 
                       degree={degree}, kernel='{kernel}', random_state={seed},
                       gamma='{gamma}', coef0={coef0}, probability={prob},
                       shrinking={shrinking}, 
                       decision_function_shape='{decision_func_shape}',
                       class_weight=None)
        """.format(tol=self.tol, c=self.c, max_iter=self.max_iter,
                   degree=self.degree, kernel=self.kernel, seed=self.seed,
                   gamma=self.gamma, coef0=self.coef0, prob=self.probability,
                   shrinking=self.shrinking,
                   decision_func_shape=self.decision_function_shape)
        return code
