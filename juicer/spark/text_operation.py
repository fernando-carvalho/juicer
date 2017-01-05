# coding=utf-8
import json
from itertools import izip_longest
from textwrap import dedent

from juicer.spark.operation import Operation


class TokenizerOperation(Operation):
    TYPE_PARAM = 'type'
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'
    MINIMUM_SIZE = 'min_token_length'
    TYPE_SIMPLE = 'simple'
    TYPE_REGEX = 'regex'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_SIMPLE)
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_tokenized'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.expression_param = parameters.get(self.EXPRESSION_PARAM, r'\s+')
        self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
        self.has_code = len(self.inputs) > 0

    def generate_code(self):
        code = """
            col_alias = {3}
            tokenizers = [Tokenizer(inputCol=col, outputCol=alias)
                                for col, alias in col_alias]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=tokenizers)

            {2} = pipeline.fit({1}).transform({1})
        """.format(self.attributes, self.inputs[0], self.output,
                   json.dumps(zip(self.attributes, self.alias)))

        return dedent(code)


class RemoveStopWordsOperation(Operation):
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    STOP_WORD_LIST_PARAM = 'stop_word_list'
    STOP_WORD_ATTRIBUTE_PARAM = 'stop_word_attribute'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.stop_word_attribute = self.parameters.get(
            self.STOP_WORD_ATTRIBUTE_PARAM, 'stop_word')

        self.stop_word_list = [s.strip() for s in
                               self.parameters.get(self.STOP_WORD_LIST_PARAM,
                                                   '').split(',')]

        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_tokenized'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.has_code = len(self.inputs) > 0

    def generate_code(self):
        if len(self.inputs) != 2:
            code = "sw = {}".format(json.dumps(self.stop_word_list))
        else:
            code = "sw = [stop[0].strip() for stop in {}.collect()]".format(
                self.named_inputs['stop words'])

        code += dedent("""
            col_alias = {3}
            removers = [StopWordsRemover(inputCol=col, outputCol=alias,
                            stopWords=sw)for col, alias in col_alias]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=removers)
            {2} = pipeline.fit({1}).transform({1})
        """.format(self.attributes, self.named_inputs['input data'],
                   self.output, json.dumps(zip(self.attributes, self.alias)), ))
        return code


class WordToVectorOperation(Operation):
    TYPE_PARAM = 'type'
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    VOCAB_SIZE_PARAM = 'vocab_size'
    MINIMUM_DF_PARAM = 'minimum_df'
    MINIMUM_TF_PARAM = 'minimum_tf'

    TYPE_COUNT = 'count'
    TYPE_WORD2VEC = 'word2vec'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_COUNT)
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_tokenized'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.vocab_size = parameters.get(self.VOCAB_SIZE_PARAM, 1000)
        self.minimum_df = parameters.get(self.MINIMUM_DF_PARAM, 5)
        self.minimum_tf = parameters.get(self.MINIMUM_TF_PARAM, 1)
        self.has_code = len(self.inputs) > 0

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.named_outputs.get(
            'vocabulary', '{}_vocab'.format(self.inputs[0]))])

    def generate_code(self):
        if self.type == self.TYPE_COUNT:
            code = dedent("""
                col_alias = {3}
                vectorizers = [CountVectorizer(minTF={4}, minDF={5},
                               vocabSize={6}, binary=False, inputCol=col,
                               outputCol=alias) for col, alias in col_alias]""")
        elif self.type == self.TYPE_WORD2VEC:
            # @FIXME Implement
            code = ""
        else:
            raise ValueError(
                "Invalid type '{}' for task {}".format(self.type,
                                                       self.__class__))

        code += dedent("""
            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=vectorizers)
            model = pipeline.fit({1})
            {2} = model.transform({1})
            """)
        vocab_out = self.named_outputs.get('vocabulary',
                                           '{}_vocab'.format(self.inputs[0]))
        code += dedent("""
            {} = dict([(col_alias[i][1], v.vocabulary)
                    for i, v in enumerate(model.stages)])""".format(vocab_out))

        code = code.format(self.attributes, self.inputs[0],
                           self.named_outputs['output data'],
                           json.dumps(zip(self.attributes, self.alias)),
                           self.minimum_tf, self.minimum_df, self.vocab_size)

        return code


class GenerateNGramsOperation(Operation):
    """ Generates N-Grams from word vectors """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    N_PARAM = 'n'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.n = int(self.parameters.get(self.N_PARAM, 2))
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_tokenized'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.has_code = len(self.inputs) > 0

    def generate_code(self):
        code = dedent("""
            col_alias = {alias}
            n_gramers = [NGram(n={n}, inputCol=col,
                           outputCol=alias) for col, alias in col_alias]
            # Use Pipeline to process all attributes once
            print '=' * 20
            {input}.show()
            print '=' * 20
            pipeline = Pipeline(stages=n_gramers)
            model = pipeline.fit({input})
            {output} = model.transform({input})
            """).format(alias=json.dumps(zip(self.attributes, self.alias)),
                        n=self.n, input=self.inputs[0], output=self.output)

        return code