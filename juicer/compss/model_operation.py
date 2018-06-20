# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class ApplyModel(Operation):
    """ApplyModel.

    REVIEW: 2017-10-20
    OK - Juicer / Tahiti / implementation
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'features' not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                .format('features', self.__class__))

        self.prediction = parameters.get('prediction', 'prediction')

        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(self.order))

        self.has_import = ""
        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input data',  'model', self.__class__))


    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 'apply_model': True,  # if its a machine learning algorithm
                 }

        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['features'] = '{features}'
        settings['predCol'] = '{predCol}'
        conf.append(settings)
        """.format(IN=self.named_inputs['input data'],
                   features=self.parameters['features'][0],
                   predCol=self.prediction)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        algorithm, model = {model}
        {OUT} = algorithm.transform({IN}, model, conf_X, numFrag)
        """.format(IN=self.named_inputs['input data'],
                   model=self.named_inputs['model'],
                   OUT=self.output)
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm, model = {model}
        settings = dict()
        settings['features'] = '{features}'
        settings['predCol'] = '{predCol}'

        {OUT} = algorithm.transform({IN}, model, settings, numFrag)
        """.format(IN=self.named_inputs['input data'],
                   model=self.named_inputs['model'],
                   features=self.parameters['features'][0],
                   predCol=self.prediction, OUT=self.output)
        return dedent(code)


class EvaluateModelOperation(Operation):
    """EvaluateModelOperation.

    REVIEW: 2017-10-20
    ??? - Juicer ?? / Tahiti ok/ implementation ok

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        for att in ['label_attribute', 'prediction_attribute', 'metric']:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(att, self.__class__))

        self.true_col = self.parameters['label_attribute'][0]
        self.pred_col = self.parameters['prediction_attribute'][0]
        self.metric = self.parameters['metric']

        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format('input data',  'model', self.__class__))

        if self.metric in ['rmse', 'mse', 'mae']:
            self.modeltype = 'RegressionModelEvaluation'
            self.has_import = "from functions.ml.metrics." \
                              "RegressionModelEvaluation import *\n"
        else:
            self.modeltype = 'ClassificationModelEvaluation'
            self.has_import = \
                "from functions.ml.metrics.ClassificationModelEvaluation" \
                " import *\n"

        self.evaluated_out = \
            self.named_outputs.get('evaluated model',
                                   'evaluated_model{}'.format(self.order))
        tmp = 'evaluator{}'.format(self.order)
        self.evaluator = self.named_outputs.get("evaluator", tmp)

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.evaluated_out, self.evaluator])

    def generate_code(self):
        """Generate code."""
        code = """
            settings = dict()
            settings['pred_col'] = '{pred_col}'
            settings['test_col'] = '{true_col}'
            settings['metric'] = '{metric}'
            {evaluator} = {type}()
            {OUT} = {evaluator}.calculate({IN}, settings, numFrag)

            """.format(IN=self.named_inputs['input data'],
                       type=self.modeltype, metric=self.metric,
                       true_col=self.true_col, pred_col=self.pred_col,
                       OUT=self.evaluated_out, evaluator=self.evaluator)

        return dedent(code)


class LoadModelOperation(Operation):
    """LoadModel.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'name' not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('name', self.__class__))

        self.filename = parameters['name']
        self.output = named_outputs.get('output data',
                                        'output_data_{}'.format(self.order))

        self.has_code = len(named_outputs) > 0
        if self.has_code:
            self.has_import = 'from functions.ml.models ' \
                              'import LoadModelOperation\n'
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('output data', self.__class__))

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': True,  # if has only one stage
                 'keep_balance': True,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['filename'] = {filename}
        settings['storage'] = 'hdfs'
        conf.append(LoadModelOperation(settings))
        """.format(filename=self.filename)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {output} = conf_X
        """.format(output=self.output)
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        settings = dict()
        settings['filename'] = {filename}
        settings['storage'] = 'hdfs'
        {model} = LoadModelOperation(settings)
        """.format(model=self.output, filename=self.filename)
        return dedent(code)


class SaveModelOperation(Operation):
    """SaveModel.

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = 'model' in parameters
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('model', self.__class__))
        self.filename = named_inputs['model']
        self.overwrite = parameters.get('write_nome', 'OVERWRITE')
        if self.overwrite == 'OVERWRITE':
            self.overwrite = True
        else:
            self.overwrite = False

        if self.has_code:
            self.has_import = 'from functions.ml.models ' \
                              'import SaveModelOperation\n'

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
        flags = {'one_stage': True,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': True,  # if need to be executed as a first task
                 }
        #!Need to Check it
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['filename'] = {filename}
        settings['overwrite'] = {overwrite}
        settings['storage'] = 'hdfs'
        sucess = SaveModelOperation({IN}, settings)
        """.format(IN=self.named_inputs['model'],
                   filename=self.filename, overwrite=self.overwrite)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {output} = conf_X
        """.format(output=self.output)
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        settings = dict()
        settings['filename'] = {filename}
        settings['overwrite'] = {overwrite}
        settings['storage'] = 'hdfs'
        sucess = SaveModelOperation({IN}, settings)
        """.format(IN=self.named_inputs['model'],
                   filename=self.filename, overwrite=self.overwrite)
        return dedent(code)
