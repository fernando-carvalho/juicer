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
        self.has_import = ""
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

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': True,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 'ml_algorithm': False,  # if its a machine learning algorithm
                 'ml_model': True
                 }

        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
            ClassifModel, settings = {algorithm}
            settings['label'] = '{label}'
            settings['features'] = '{features}'
            model = ClassifModel.fit({input}, settings, numFrag)
            {model} = [ClassifModel, model]
            """.format(model=self.model, label=self.label,
                       input=self.named_inputs['train input data'],
                       algorithm=self.named_inputs['algorithm'],
                       features=self.features)

        if self.perform_transformation:
            code += """
            settings['predCol'] = '{predCol}'
            {OUT} = ClassifModel.transform_serial({IN}, model, settings)
            """.format(predCol=self.predCol, OUT=self.output,
                       IN=self.named_inputs['train input data'])
        else:
            code += """
            {output} = None
            """.format(output=self.output)

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
        self.has_import = "from functions.ml.classification.Knn.knn "\
                          "import KNN\n"

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 'ml_algorithm': True,  # if its a machine learning algorithm
                 }

        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        ClassifModel = KNN()
        settings = dict()
        settings['K'] = {K}
        {output} = [ClassifModel, settings]
        """.format(K=self.parameters['k'], output=self.output)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
            ClassifModel = KNN()
            settings = dict()
            settings['K'] = {K}
            {output} = [ClassifModel, settings]
            """.format(K=self.parameters['k'], output=self.output)
        return dedent(code)


class LogisticRegressionOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        attributes = ['coef_alpha', 'coef_lr', 'coef_threshold', 'max_iter']
        for att in attributes:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(att, self.__class__))

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))
        self.has_import = "from functions.ml.classification." \
                          "LogisticRegression.logisticRegression " \
                          "import logisticRegression\n"

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 'ml_algorithm': True,  # if its a machine learning algorithm
                 }

        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return dedent(code)


    def generate_code(self):
        """Generate code."""
        code = """
            ClassifModel = logisticRegression()
            settings = dict()
            settings['alpha'] = {alpha}
            settings['iters'] = {maxIters}
            settings['threshold'] = {threshold}
            settings['regularization'] = {regularization}
            {output} = [ClassifModel, settings]
            """.format(alpha=self.parameters['coef_alpha'],
                       regularization=self.parameters['coef_lr'],
                       threshold=self.parameters['coef_threshold'],
                       maxIters=self.parameters['max_iter'],
                       output=self.output)
        return dedent(code)


class NaiveBayesClassifierOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))
        self.has_import = "from functions.ml.classification.NaiveBayes." \
                          "naivebayes import GaussianNB\n"

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 'ml_algorithm': True,  # if its a machine learning algorithm
                 }

        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return dedent(code)


    def generate_code(self):
        """Generate code."""
        code = """
            ClassificationModel = GaussianNB()
            {output} = [ClassificationModel, dict()]
            """.format(output=self.output)
        return dedent(code)


class SvmClassifierOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        attributes = ['coef_lambda', 'coef_lr', 'coef_threshold', 'max_iter']
        for att in attributes:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(att, self.__class__))

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))
        self.has_import = "from functions.ml.classification.Svm.svm "\
                          "import SVM\n"

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 'ml_algorithm': True,  # if its a machine learning algorithm
                 }

        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return dedent(code)


    def generate_code(self):
        """Generate code."""
        code = """
            ClassificationModel = SVM()
            settings = dict()
            settings['coef_lambda'] = {coef_lambda}
            settings['coef_lr'] = {coef_lr}
            settings['coef_threshold'] = {coef_threshold}
            settings['coef_maxIters'] = {coef_maxIters}

            {output} = [ClassificationModel, settings]
            """.format(coef_lambda=self.parameters['coef_lambda'],
                       coef_lr=self.parameters['coef_lr'],
                       coef_threshold=self.parameters['coef_threshold'],
                       coef_maxIters=self.parameters['max_iter'],
                       output=self.output)
        return dedent(code)
