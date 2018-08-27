from textwrap import dedent
from juicer.operation import Operation
from itertools import izip_longest


class FeatureAssemblerOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'attributes' not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                .format('attributes', self.__class__))

        self.alias = parameters.get("alias", 'FeatureField')
        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                     self.order))

        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):
        code = """
        cols = {cols}
        {output} = {input}
        {output}['{alias}'] = {input}[cols].values.tolist()
        """.format(output=self.output, alias=self.alias,
                   input=self.named_inputs['input data'],
                   cols=self.parameters['attributes'])

        return dedent(code)


class MaxAbsScalerOperation(Operation):
    """
    Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually
    such that the maximal absolute value of each feature in the training
    set will be 1.0. It does not shift/center the data, and thus does not
     destroy any sparsity.
    """
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if 'attribute' not in parameters:
            raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format('attribute', self.__class__))

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)
        self.alias = [s.strip()
                      for s in parameters.get("alias", []).split(',')]
        self.attributes = parameters['attribute']
        # Adjust alias in order to have the same number of aliases
        # as attributes by filling missing alias with the attribute
        # name sufixed by _indexed.
        if len(self.alias) > 0:
            self.alias = [x[1] or '{}_norm'.format(x[0]) for x in
                          izip_longest(self.attributes,
                                       self.alias[:len(self.attributes)])]

        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = scaler.fit_transform(X_train).tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attributes[0], alias=self.alias[0])
        return dedent(code)


class MinMaxScalerOperation(Operation):
    """
    Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually
    such that it is in the given range on the training set, i.e.
    between zero and one.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    """
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'attribute' not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                .format('attribute', self.__class__))

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)
        self.alias = [s.strip()
                      for s in parameters.get("alias", []).split(',')]
        self.attributes = parameters['attribute']
        # Adjust alias in order to have the same number of aliases
        # as attributes by filling missing alias with the attribute
        # name sufixed by _indexed.
        if len(self.alias) > 0:
            self.alias = [x[1] or '{}_norm'.format(x[0]) for x in
                          izip_longest(self.attributes,
                                       self.alias[:len(self.attributes)])]

        self.min = parameters.get('min', 0)
        self.max = parameters.get('max', 1)
        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=({min},{max}))
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = scaler.fit_transform(X_train).tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attributes[0], alias=self.alias[0],
                   min=self.min, max=self.max)
        return dedent(code)


class StandardScalerOperation(Operation):
    """
    Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Mean and standard
    deviation are then stored to be used on later data using the transform
    method.

    Standardization of a dataset is a common requirement for many machine
    learning estimators: they might behave badly if the individual feature
    do not more or less look like standard normally distributed data.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if 'attribute' not in parameters:
            raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format('attribute', self.__class__))

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)
        self.alias = [s.strip()
                      for s in parameters.get("alias", []).split(',')]
        self.attributes = parameters['attribute']
        # Adjust alias in order to have the same number of aliases
        # as attributes by filling missing alias with the attribute
        # name sufixed by _indexed.
        if len(self.alias) > 0:
            self.alias = [x[1] or '{}_norm'.format(x[0]) for x in
                          izip_longest(self.attributes,
                                       self.alias[:len(self.attributes)])]

        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = scaler.fit_transform(X_train).tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attributes[0], alias=self.alias[0])
        return dedent(code)


class OneHotEncoderOperation(Operation):
    """
    Encode categorical integer features using a one-hot aka one-of-K scheme.

    This encoding is needed for feeding categorical data to many
    scikit-learn estimators, notably linear models and SVMs with
    the standard kernels.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if 'attributes' not in parameters:
            raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format('attributes', self.__class__))

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)
        self.alias = [s.strip()
                      for s in parameters.get("alias", []).split(',')]
        self.attributes = parameters['attributes']
        # Adjust alias in order to have the same number of aliases
        # as attributes by filling missing alias with the attribute
        # name sufixed by _indexed.
        if len(self.alias) > 0:
            self.alias = [x[1] or '{}_norm'.format(x[0]) for x in
                          izip_longest(self.attributes,
                                       self.alias[:len(self.attributes)])]

        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = enc.fit_transform(X_train).toarray().tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attributes[0], alias=self.alias[0])
        return dedent(code)


class PCAOperation(Operation):
    """
    Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value
    Decomposition of the data to project it to a lower dimensional space.
    """
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if 'attribute' not in parameters:
            raise ValueError(
                    _("Parameters '{}' must be informed for task {}")
                    .format('attribute', self.__class__))

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)
        self.alias = [s.strip()
                      for s in parameters.get("alias", 'pca_feature').split(',')]
        self.attributes = parameters['attribute']
        # Adjust alias in order to have the same number of aliases
        # as attributes by filling missing alias with the attribute
        # name sufixed by _indexed.
        if len(self.alias) > 0:
            self.alias = [x[1] or '{}_norm'.format(x[0]) for x in
                          izip_longest(self.attributes,
                                       self.alias[:len(self.attributes)])]

        self.has_code = len(self.named_inputs) == 1
        self.n_comp = self.parameters['k']

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}
        from sklearn.decomposition import PCA
        pca = PCA(n_components={n_comp})
        X_train = {input}['{att}'].values.tolist()
        {output}['{alias}'] = pca.fit_transform(X_train).tolist()
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   att=self.attributes[0], alias=self.alias[0],
                   n_comp=self.n_comp)
        return dedent(code)