from textwrap import dedent
from juicer.operation import Operation
from itertools import izip_longest


# noinspection PyAbstractClass
class RegressionOperation(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_ATTR_PARAM = 'prediction'

    __slots__ = ('label', 'features', 'prediction')

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.label = self.features = self.prediction = None
        self.output = named_outputs.get(
            'algorithm', 'regression_algorithm_{}'.format(self.order))

    def read_common_params(self, parameters):
        if not all([self.LABEL_PARAM in parameters,
                    self.FEATURES_PARAM in parameters]):
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.FEATURES_PARAM, self.LABEL_PARAM,
                self.__class__))
        else:
            self.label = parameters.get(self.LABEL_PARAM)[0]
            self.features = parameters.get(self.FEATURES_PARAM)[0]
            self.prediction = parameters.get(self.PREDICTION_ATTR_PARAM)[0]
            self.output = self.named_outputs.get(
                'algorithm', 'regression_algorithm_{}'.format(self.order))

    def get_output_names(self, sep=', '):
        return self.output

    def get_data_out_names(self, sep=','):
        return ''


class RegressionModelOperation(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_COL_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 2,
                             self.contains_results()])

        if self.has_code:
            self.algorithm = self.named_inputs['algorithm']
            self.input = self.named_inputs['train input data']

            if not all([self.FEATURES_PARAM in parameters,
                        self.LABEL_PARAM in parameters]):
                msg = _("Parameters '{}' and '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM, self.LABEL_PARAM,
                    self.__class__.__name__))

            self.features = parameters[self.FEATURES_PARAM]
            self.label = parameters[self.LABEL_PARAM]
            self.prediction = parameters.get(self.PREDICTION_COL_PARAM,
                                             'prediction') or 'prediction'
            self.model = self.named_outputs.get(
                'model', 'model_{}'.format(self.order))
            self.output = self.named_outputs.get(
                'output data', 'out_task_{}'.format(self.order))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):

        #TODO:  np.ravel() if isotonic

        if self.has_code:
            code = """
            algorithm = {algorithm}
            {output_data} = {input}
            X_train = {input}['{features}'].values.tolist()
            y = {input}['{label}'].values.tolist()
            {model} = algorithm.fit(X_train, y)
            {output_data}['{prediction}'] = algorithm.predict(X_train).tolist()
            """.format(model=self.model, algorithm=self.algorithm,
                       input=self.named_inputs['train input data'],
                       output_data=self.output, prediction=self.prediction,
                       label=self.label[0], features=self.features[0])

            return dedent(code)


class LinearRegressionOperation(RegressionOperation):
    MAX_ITER_PARAM = 'max_iter'
    WEIGHT_COL_PARAM = 'weight'
    REG_PARAM = 'reg_param'
    ELASTIC_NET_PARAM = 'elastic_net'

    SOLVER_PARAM = 'solver'

    TYPE_SOLVER_AUTO = 'auto'
    TYPE_SOLVER_NORMAL = 'normal'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.LinearRegression'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            # self.read_common_params(parameters)

            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10) or 10
            self.reg_param = parameters.get(self.REG_PARAM, 0.1) or 0.0
            self.weight_col = parameters.get(self.WEIGHT_COL_PARAM, None)

            self.solver = self.parameters.get(self.SOLVER_PARAM,
                                              self.TYPE_SOLVER_AUTO)
            self.elastic = self.parameters.get(self.ELASTIC_NET_PARAM,
                                               0.0) or 0.0

    def generate_code(self):
        if self.has_code:
            code = dedent("""
            from sklearn.linear_model import ElasticNet
            {output} = ElasticNet(alpha={reg_param}, l1_ratio={elastic},
             max_iter={max_iter})""").format(
                output=self.output,
                max_iter=self.max_iter,
                reg_param=self.reg_param,
                elastic=self.elastic,
            )
            return code
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class RandomForestRegressorOperation(RegressionOperation):

    MAX_DEPTH_PARAM = 'max_depth'
    MAX_BINS_PARAM = 'max_bins'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    NUM_TREES_PARAM = 'num_trees'
    FEATURE_SUBSET_STRATEGY_PARAM = 'feature_subset_strategy'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.RandomForestRegressor'
        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 5) or 5
            self.max_bins = parameters.get(self.MAX_BINS_PARAM, 32) or 32
            self.min_info_gain = parameters.get(self.MIN_INFO_GAIN_PARAM,
                                                0.0) or 0.0
            self.num_trees = parameters.get(self.NUM_TREES_PARAM, 20) or 20
            self.feature_subset_strategy = parameters.get(
                self.FEATURE_SUBSET_STRATEGY_PARAM, 'auto')
            # self.feature_subset_strategy = 'auto', 'sqrt', log2', None

    def generate_code(self):
        code = dedent("""
            from sklearn.ensemble import RandomForestRegressor
            {output} = RandomForestRegressor(
                max_depth={max_depth},
                min_samples_split={max_bins},
                min_impurity_decrease={min_info_gain},
                n_estimators={num_trees},
                max_features='{feature_subset_strategy}')
            """).format(output=self.output,
                        max_depth=self.max_depth,
                        max_bins=self.max_bins,
                        min_info_gain=self.min_info_gain,
                        num_trees=self.num_trees,
                        feature_subset_strategy=self.feature_subset_strategy)
        return code


class IsotonicRegressionOperation(RegressionOperation):
    """
        Only univariate (single feature) algorithm supported
    """
    WEIGHT_COL_PARAM = 'weight'
    ISOTONIC_PARAM = 'isotonic'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.IsotonicRegression'
        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.weight_col = parameters.get(self.WEIGHT_COL_PARAM, None)
            self.isotonic = parameters.get(
                self.ISOTONIC_PARAM, True) in (1, '1', 'true', True)

    def generate_code(self):
        code = dedent("""
        from sklearn.isotonic import IsotonicRegression
        {output} = IsotonicRegression(increasing={isotonic})
        """).format(output=self.output,
                    isotonic=self.isotonic, )
        return code


