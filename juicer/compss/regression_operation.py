# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class RegressionModelOperation(Operation):

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

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': True,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': True,  # if need to be executed as a first task
                 'algorithm_ml': False,  # if its a machine learning algorithm
                 }

        return flags

    def generate_code(self):
        """Generate code."""
        code = """
            regressor, settings = {algorithm}
            settings['label'] = '{label}'
            settings['features'] = '{features}'
            model = regressor.fit({input}, settings, numFrag)
            {model} = [regressor, model]
            """.format(output=self.output, model=self.model,
                       input=self.named_inputs['train input data'],
                       algorithm=self.named_inputs['algorithm'],
                       label=self.label, features=self.features)

        if self.perform_transformation:
            code += """
            settings['predCol'] = '{predCol}'
            {output} = regressor.transform({input}, model, settings, numFrag)
            """.format(predCol=self.predCol, output=self.output,
                       input=self.named_inputs['train input data'])
        else:
            code += 'task_{} = None'.format(self.order)

        return dedent(code)


class LinearRegressionOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = True
        self.output = named_outputs.get('algorithm',
                                        'algorithm_tmp{}'.format(self.order))
        self.maxIters = parameters.get('max_iter', 100)
        self.alpha = parameters.get('alpha', 0.001)
        self.has_import = "from functions.ml.regression.linearRegression." \
                          "linearRegression import linearRegression\n"

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': True,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 'algorithm_ml': True,  # if its a machine learning algorithm
                 }

        return flags

    def generate_code(self):
        """Generate code."""
        code = """
            regression_model = linearRegression()
            settings = dict()
            settings['alpha'] = {alpha}
            settings['max_iter'] = {it}
            settings['option'] = 'SDG'
            {output} = [regression_model, settings]
            """.format(alpha=self.alpha, it=self.maxIters, output=self.output)
        return dedent(code)
