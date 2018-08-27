from textwrap import dedent
from juicer.operation import Operation


class ClassificationModelOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'label' not in parameters and 'features' not in parameters:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format('label',  'features', self.__class__))

        self.label = parameters['label'][0]
        self.features = parameters['features'][0]
        self.predCol = parameters.get('prediction', 'prediction')
        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format('train input data',  'algorithm', self.__class__))

        self.model = self.named_outputs.get('model',
                                            'model_tmp{}'.format(self.order))

        self.perform_transformation = 'output data' in self.named_outputs
        if not self.perform_transformation:
            self.output = 'task_{}'.format(self.order)
        else:
            self.output = self.named_outputs['output data']
            self.prediction = self.parameters.get('prediction', 'prediction')

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        code = """
            X = {input}['{features}'].values.tolist()
            y = {input}['{label}'].values.tolist()
            {model} = {algorithm}.fit(X, y)
            """.format(model=self.model, label=self.label,
                       input=self.named_inputs['train input data'],
                       algorithm=self.named_inputs['algorithm'],
                       features=self.features)

        if self.perform_transformation:
            code += """
            {OUT} = {IN}
            
            {OUT}['{predCol}'] = {model}.predict(X).tolist()
            """.format(predCol=self.predCol, OUT=self.output, model=self.model,
                       IN=self.named_inputs['train input data'])
        else:
            code += """
            {output} = None
            """.format(output=self.output)

        return dedent(code)


class DecisionTreeClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))

        self.min_split = self.parameters.get('min_samples_split', 2)
        self.min_leaf = self.parameters.get('min_samples_leaf', 1)
        self.max_depth = self.parameters.get('max_depth', None)
        self.min_weight = self.parameters.get('min_weight', 0.0)
        self.seed = self.parameters.get('seed', None)

        self.min_split = self.min_split if not self.min_split == '' else 2
        self.min_leaf = self.min_leaf if not self.min_leaf == '' else 1
        self.max_depth = self.max_depth if not self.max_depth == '' else None
        self.min_weight = self.min_weight if not self.min_weight == '' else 0.0
        self.seed = self.seed if not self.seed == '' else None

    def generate_code(self):
        """Generate code."""
        code = """
        from sklearn.tree import DecisionTreeClassifier
        {output} = DecisionTreeClassifier(max_depth={max_depth}, 
        min_samples_split={min_split}, min_samples_leaf={min_leaf}, 
        min_weight_fraction_leaf={min_weight}, random_state={seed})
        """.format(output=self.output, min_split=self.min_split,
                   min_leaf=self.min_leaf, min_weight=self.min_weight,
                   seed=self.seed, max_depth=self.max_depth)
        return dedent(code)


class GBTClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))

        self.min_split = self.parameters.get('min_samples_split', 2)
        self.min_leaf = self.parameters.get('min_samples_leaf', 1)
        self.n_estimators = self.parameters.get('n_estimators', 100)
        self.learning_rate = self.parameters.get('learning_rate', 0.1)
        self.loss = self.parameters.get('loss', 'deviance')

        self.min_split = self.min_split if not self.min_split == '' else 2
        self.min_leaf = self.min_leaf if not self.min_leaf == '' else 1

    def generate_code(self):
        """Generate code."""
        code = """
        from sklearn.ensemble import GradientBoostingClassifier
        {output} = GradientBoostingClassifier(loss='{loss}',
        learning_rate={learning_rate}, n_estimators={n_estimators},
        min_samples_split={min_split}, min_samples_leaf={min_leaf})
        """.format(output=self.output, loss=self.loss,
                   n_estimators=self.n_estimators, min_leaf=self.min_leaf,
                   min_split=self.min_split, learning_rate=self.learning_rate)
        return dedent(code)


class KNNClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'k' not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('k', self.__class__))

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))

    def generate_code(self):
        """Generate code."""
        code = """
        from sklearn.neighbors import KNeighborsClassifier
        {output} = KNeighborsClassifier(n_neighbors={K})
        """.format(K=self.parameters['k'], output=self.output)
        return dedent(code)


class LogisticRegressionOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))

        self.tol = self.parameters.get('tol', 0.0001)
        self.C = self.parameters.get('regularization', 1.0)
        self.max_iter = self.parameters.get('max_iter', 100)
        self.seed = self.parameters.get('seed', None)
        self.solver = self.parameters.get('solver', 'liblinear')

    def generate_code(self):
        """Generate code."""
        code = """
            from sklearn.linear_model import LogisticRegression
            {output} = LogisticRegression(tol={tol}, C={C}, max_iter={max_iter},
            solver='{solver}', random_state={seed})
            """.format(tol=self.tol,
                       C=self.C,
                       max_iter=self.max_iter,
                       seed=self.seed,
                       solver=self.solver,
                       output=self.output)
        return dedent(code)


class NaiveBayesClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))

        self.weight_attr = False  # FIXME
        self.thresholds = self.parameters.get('thresholds', 0.0)
        self.smoothing = self.parameters.get('smoothing', 1.0)
        self.model_type = self.parameters.get('model_type', 'multinomial')

    def generate_code(self):
        """Generate code."""
        if self.model_type == 'multinomial':
            code = """
        from sklearn.naive_bayes import MultinomialNB
        {output} = MultinomialNB(alpha={smoothing})
        """.format(output=self.output, smoothing= self.smoothing)
        else:
            code = """
        from sklearn.naive_bayes import BernoulliNB
        {output} = BernoulliNB(alpha={smoothing}, binarize={thresholds})
        """.format(output=self.output, smoothing= self.smoothing,
                   thresholds=self.thresholds)

        return dedent(code)


class PerceptronClassifierOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))

        self.max_iter = self.parameters.get('max_iter', 1000)
        self.alpha = self.parameters.get('alpha', 0.0001)
        self.tol = self.parameters.get('tol', 0.001)
        self.shuffle = self.parameters['shuffle']
        self.seed = self.parameters.get('seed', None)
        self.penalty = self.parameters.get('penalty', None)

        self.shuffle = self.shuffle if not self.shuffle == '' else False
        self.seed = self.seed if not self.seed == '' else None

    def generate_code(self):
        """Generate code."""
        code = """
        from sklearn.linear_model import Perceptron
        {output} = Perceptron(tol={tol}, alpha={tol}, max_iter={max_iter},
        shuffle={shuffle}, random_state={seed}, penalty='{penalty}')
        """.format(tol=self.tol,
                   alpha=self.alpha,
                   max_iter=self.max_iter,
                   shuffle=self.shuffle,
                   penalty=self.penalty,
                   seed=self.seed,
                   output=self.output)
        return dedent(code)


class RandomForestClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))

        self.seed = self.parameters.get('seed', None)
        self.min_split = self.parameters.get('min_samples_split', 2)
        self.min_leaf = self.parameters.get('min_samples_leaf', 1)
        self.max_depth = self.parameters.get('max_depth', 'None')
        self.n_estimators = self.parameters.get('n_estimators', 10)

        self.min_split = self.min_split if not self.min_split == '' else 2
        self.min_leaf = self.min_leaf if not self.min_leaf == '' else 1
        self.max_depth = self.max_depth if not self.max_depth == '' else None
        self.seed = self.seed if not self.seed == '' else None

    def generate_code(self):
        """Generate code."""
        code = """
        from sklearn.ensemble import RandomForestClassifier
        {output} = RandomForestClassifier(n_estimators={n_estimators}, 
        max_depth={max_depth},  min_samples_split={min_split}, 
        min_samples_leaf={min_leaf}, random_state={seed})
        """.format(output=self.output, n_estimators=self.n_estimators,
                   max_depth=self.max_depth, min_split=self.min_split,
                   min_leaf=self.min_leaf, seed=self.seed)

        return dedent(code)


class SvmClassifierOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))

        self.max_iter = self.parameters.get('max_iter', 100)
        self.threshold = self.parameters.get('threshold', 1e-3)
        self.tol = self.parameters.get('tol', 1.0)
        self.standardization = False  # FIXME
        self.weight_attr = False  # FIXME

    def generate_code(self):
        """Generate code."""
        code = """
        from sklearn.svm import LinearSVC
        {output} = LinearSVC(tol={threshold}, C={tol}, max_iter={maxIters})
        """.format(tol=self.tol,
                   threshold=self.threshold,
                   maxIters=self.max_iter,
                   output=self.output)
        return dedent(code)
