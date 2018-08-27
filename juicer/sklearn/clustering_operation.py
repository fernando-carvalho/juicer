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
        X = {input}['{features}'].values.tolist()
        {model} = {algorithm}.fit(X)
        """.format(model=self.model, features=self.features,
                   input=self.named_inputs['train input data'],
                   algorithm=self.named_inputs['algorithm'])

        if self.perform_transformation:
            code += """
        y = cluster_model.predict({IN})
        {OUT} = {IN}
        {OUT}['{predCol}'] = y
        """.format(OUT=self.output, model=self.model,
                   IN=self.named_inputs['train input data'],
                   predCol=self.predCol)
        else:
            code += """
        {output} = None
        """.format(output=self.output)

        return dedent(code)


class GaussianMixtureClusteringOperation(Operation):
    K_PARAM = 'number_of_clusters'
    MAX_ITERATIONS_PARAM = 'max_iterations'
    TOLERANCE_PARAMETER = 'tolerance'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.number_of_clusters = parameters.get(self.K_PARAM, 10)
        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)
        self.tolerance = float(parameters.get(self.TOLERANCE_PARAMETER, 0.001))

        self.has_code = len(named_outputs) > 0

        self.output = named_outputs.get('algorithm')

    def generate_code(self):
        """Generate code."""
        code = """
        from sklearn.mixture import GaussianMixture
        {output} = GaussianMixture(n_components={k}, max_iter={iter}, tol={tol})
        """.format(k=self.number_of_clusters,
                   iter=self.max_iterations, tol=self.tolerance,
                   output=self.output)

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
        self.init_mode = parameters.get('init_mode', 'k-means++')
        if self.init_mode == "k-means||":
            self.init_mode = "k-means++"
        self.epsilon = parameters.get('tolerance', 0.001)
        self.type = parameters.get('type', "kmeans")
        self.has_code = len(named_outputs) >= 1

        self.output = named_outputs.get('algorithm')

    def generate_code(self):
        """Generate code."""
        if self.type == "kmeans":
            code = """
            from sklearn.cluster import KMeans
            {output} = KMeans(n_clusters={k}, init='{init}',
                max_iter={it}, tol={ep})
            """.format(k=self.K, it=self.maxIters, ep=self.epsilon,
                       init=self.init_mode, output=self.output)
        elif self.type == 'bisecting':
            code = ""
        return dedent(code)


class LdaClusteringOperation(Operation):
    NUMBER_OF_TOPICS_PARAM = 'number_of_topics'
    OPTIMIZER_PARAM = 'optimizer'
    MAX_ITERATIONS_PARAM = 'max_iterations'
    DOC_CONCENTRATION_PARAM = 'doc_concentration'
    TOPIC_CONCENTRATION_PARAM = 'topic_concentration'

    ONLINE_OPTIMIZER = 'online'
    EM_OPTIMIZER = 'em'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.output = named_outputs.get('algorithm')
        self.number_of_clusters = int(parameters.get(
            self.NUMBER_OF_TOPICS_PARAM, 10))
        self.optimizer = parameters.get(self.OPTIMIZER_PARAM,
                                        self.ONLINE_OPTIMIZER)
        if self.optimizer not in [self.ONLINE_OPTIMIZER, self.EM_OPTIMIZER]:
            raise ValueError(
                _('Invalid optimizer value {} for class {}').format(
                    self.optimizer, self.__class__))
        if self.optimizer == 'em':
            self.optimizer = 'batch'

        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)

        self.doc_concentration = parameters.get(self.DOC_CONCENTRATION_PARAM)
        if len(self.doc_concentration) == 0:
            self.doc_concentration = None

        self.topic_concentration = parameters.get(
            self.TOPIC_CONCENTRATION_PARAM)

        if len(self.topic_concentration) == 0:
            self.topic_concentration = None

        if self.doc_concentration:
            try:
                doc_concentration = [float(v) for v in
                                     str(self.doc_concentration).split(',') if
                                     v.strip()]
                print (doc_concentration)
            except Exception as e:
                raise ValueError(
                    _('Invalid document concentration: {}. It must be a single '
                      'decimal value or a list of decimal numbers separated by '
                      'comma.').format(
                        self.doc_concentration))
        self.has_code = len(named_outputs) > 0

    def generate_code(self):
        """Generate code."""
        code = """
        from sklearn.decomposition import LatentDirichletAllocation
        {output} = LatentDirichletAllocation(n_components={k}, 
        doc_topic_prior={doc}, topic_word_prior={topic}, 
        learning_method='{learning_method}', max_iter={iter})
        """.format(k=self.number_of_clusters, iter=self.max_iterations,
                   topic=self.topic_concentration,
                   doc=self.doc_concentration,
                   learning_method=self.optimizer,
                   output=self.output)

        return dedent(code)