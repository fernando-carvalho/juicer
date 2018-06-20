# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class ClusteringModelOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'features' not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('features', self.__class__))

        self.features = parameters['features'][0]
        self.model = self.named_outputs.get('model',
                                            'model_tmp_{}'.format(self.output))

        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format('train input data',  'algorithm', self.__class__))

        self.perform_transformation = 'output data' in self.named_outputs
        if not self.perform_transformation:
            self.output = 'task_{}'.format(self.order)
        else:
            self.output = self.named_outputs['output data']
            self.prediction = self.parameters.get('prediction', 'prediction')

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        code = """
            cluster_model, settings = {algorithm}
            settings['features'] = '{features}'
            model = cluster_model.fit({input}, settings, numFrag)
            {model} = [cluster_model, model]
            """.format(model=self.model, features=self.features,
                       input=self.named_inputs['train input data'],
                       algorithm=self.named_inputs['algorithm'])

        if self.perform_transformation:
            code += """
            settings['predCol'] = '{predCol}'
            {OUT} = cluster_model.transform({IN}, model, settings, numFrag)
            """.format(OUT=self.output, model=self.model,
                       IN=self.named_inputs['train input data'],
                       predCol=self.predCol)
        else:
            code += """
            {output} = None
            """.format(output=self.output)

        return dedent(code)


class KMeansClusteringOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        attributes = ['number_of_clusters', 'max_iterations', 'init_mode']
        for att in attributes:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(att, self.__class__))

        self.K = parameters['number_of_clusters']
        self.maxIters = parameters['max_iterations']
        self.init_mode = parameters.get('init_mode', 'k-means||')
        self.epsilon = parameters.get('tolerance', 0.001)

        self.has_code = len(named_outputs) > 1

        if self.has_code:
            self.has_import = "from functions.ml.clustering.Kmeans.Kmeans " \
                              "import Kmeans\n"
            self.output = named_outputs.get('algorithm')


    def get_optimization_information(self):

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
            cluster_model = Kmeans()
            settings = dict()
            settings['k'] = {k}
            settings['maxIterations'] = {it}
            settings['epsilon'] = {ep}
            settings['initMode'] = '{init}'
            {output} = [cluster_model, settings]
            """.format(k=self.K, it=self.maxIters, ep=self.epsilon,
                       init=self.init_mode, output=self.output)
        return dedent(code)


class DBSCANClusteringOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.eps = parameters.get('eps', 0.001)
        self.minPts = parameters.get('minPts', 15)
        if 'features' not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('features', self.__class__))

        tmp = 'output_data_{}'.format(self.order)
        self.features = parameters['features'][0]
        self.predCol = self.parameters.get('prediction', 'prediction')
        self.output = self.named_outputs.get('output data', tmp)
        self.input = self.named_inputs['input data']

        self.has_code = len(named_inputs) == 1

        if self.has_code:
            self.has_import = "from functions.ml.clustering.DBSCAN.dbscan " \
                              "import DBSCAN\n"

    def get_optimization_information(self):
        # Not supported. DBSCAN doesnt have centroids to be used as model
        flags = {}
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
            cluster_model = DBSCAN()
            settings = dict()
            settings['feature'] = '{features}'
            settings['predCol'] = '{predCol}'
            settings['minPts'] = {pts}
            settings['eps'] = {eps}
            {output} = cluster_model.fit_predict({data_in}, settings, numFrag)
            """.format(pts=self.minPts, eps=self.eps, output=self.output,
                       features=self.features, predCol=self.predCol,
                       data_in=self.input)
        return dedent(code)
