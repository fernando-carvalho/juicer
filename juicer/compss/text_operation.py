# -*- coding: utf-8 -*-
import ast
import pprint
from textwrap import dedent
from juicer.operation import Operation
from itertools import izip_longest


class TokenizerOperation(Operation):
    """TokenizerOperation.

    Tokenization is the process of taking text (such as a sentence) and
    breaking it into individual terms (usually words). A simple Tokenizer
    class provides this functionality.

    """

    TYPE_PARAM = 'type'
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'
    MINIMUM_SIZE = 'min_token_length'

    TYPE_SIMPLE = 'simple'
    TYPE_REGEX = 'regex'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_SIMPLE)
        if self.type not in [self.TYPE_REGEX, self.TYPE_SIMPLE]:
            raise ValueError(_('Invalid type for operation Tokenizer: {}')
                             .format(self.type))

        self.expression_param = parameters.get(self.EXPRESSION_PARAM, '\s+')

        self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as
        # attributes by filling missing alias with the attribute name
        # sufixed by _indexed.
        self.alias = [x[1] or '{}_tok'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.has_import = \
                "from functions.text.tokenizer import TokenizerOperation\n"

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def get_optimization_information(self):
        flags = {'one_stage': True,  # if has only one stage
                 'keep_balance': True,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': True,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['min_token_length'] = {min_token}
        settings['attributes'] = {att}
        settings['alias'] = {alias}""" \
        .format(min_token=self.min_token_lenght, att=self.attributes,
                alias=self.alias)

        if self.type == self.TYPE_SIMPLE:
            code += """
        settings['type'] = 'simple'"""
        else:
            code += """
        setting['expression'] = '{expression}'
        settings['type'] = 'regex'""".format(expression=self.expression_param)
        code += """
        conf.append(TokenizerOperation().preprocessing(settings))
        """
        return code

    def generate_optimization_code(self):
        """Generate code."""
        code = """
        {output} = TokenizerOperation().transform_serial({input}, conf_X)
        """.format(output=self.output,
                   input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
            settings = dict()
            settings['min_token_length'] = {min_token}
            settings['attributes'] = {att}
            settings['alias'] = {alias}"""\
            .format(min_token=self.min_token_lenght, att=self.attributes,
                    alias=self.alias)

        if self.type == self.TYPE_SIMPLE:
            code += """
                settings['type'] = 'simple'
            """
        else:
            code += """
                setting['expression'] = '{expression}'
                settings['type'] = 'regex'
            """.format(expression=self.expression_param)

        code += """
                {OUT} = TokenizerOperation().transform({IN}, settings, numFrag)
            """.format(OUT=self.output, IN=self.named_inputs['input data'])

        return dedent(code)


class RemoveStopWordsOperation(Operation):
    """RemoveStopWordsOperation.

    Stop words are words which should be excluded from the input,
    typically because the words appear frequently and don't carry
    as much meaning.
    """

    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    STOP_WORD_LIST_PARAM = 'stop_word_list'
    STOP_WORD_ATTRIBUTE_PARAM = 'stop_word_attribute'
    STOP_WORD_LANGUAGE_PARAM = 'stop_word_language'
    STOP_WORD_CASE_SENSITIVE_PARAM = 'sw_case_sensitive'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.stop_word_attribute = self.parameters.get(
            self.STOP_WORD_ATTRIBUTE_PARAM, "")

        self.stop_word_list = [s.strip() for s in
                               self.parameters.get(self.STOP_WORD_LIST_PARAM,
                                                   '').split(',')]

        self.alias = parameters.get(self.ALIAS_PARAM, 'tokenized_rm')

        self.sw_case_sensitive = self.parameters.get(
            self.STOP_WORD_CASE_SENSITIVE_PARAM, 'False')

        self.stopwords_input = self.named_inputs.get('stop words', [])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.has_code = 'input data' in self.named_inputs
        if self.has_code:
            self.has_import = "from functions.text.remove_stopwords "\
                              "import RemoveStopWordsOperation\n"

    def get_optimization_information(self):
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': True,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': True,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['attribute'] = {att}
        settings['alias'] = '{alias}'
        settings['col_words'] = '{att_stop}'
        settings['case-sensitive'] = {case}
        settings['news-stops-words'] = {stoplist}

        conf.append(RemoveStopWordsOperation().preprocessing({sw}, settings))
        """.format(att=self.attributes,
                   att_stop=self.stop_word_attribute,
                   alias=self.alias, case=self.sw_case_sensitive,
                   stoplist=self.stop_word_list,
                   sw=self.stopwords_input)
        return code

    def generate_optimization_code(self):
        """Generate code."""
        code = """
        {OUT} = RemoveStopWordsOperation().transform_serial(
        {IN}, settings, {sw})
        """.format(output=self.output,
                   input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
            settings = dict()
            settings['attribute'] = {att}
            settings['alias'] = '{alias}'
            settings['col_words'] = '{att_stop}'
            settings['case-sensitive'] = {case}
            settings['news-stops-words'] = {stoplist}

            {OUT} = RemoveStopWordsOperation().transform(
            {IN}, settings, {sw}, numFrag)
            """.format(att=self.attributes,
                       att_stop=self.stop_word_attribute,
                       alias=self.alias, case=self.sw_case_sensitive,
                       OUT=self.output, IN=self.named_inputs['input data'],
                       stoplist=self.stop_word_list, sw=self.stopwords_input)

        return dedent(code)


class WordToVectorOperation(Operation):
    """WordToVectorOperation.

    Can be used Bag of Words transformation or TF-IDF.
    """
    TYPE_PARAM = 'type'
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    VOCAB_SIZE_PARAM = 'vocab_size'
    MINIMUM_DF_PARAM = 'minimum_df'
    MINIMUM_TF_PARAM = 'minimum_tf'

    MINIMUM_VECTOR_SIZE_PARAM = 'minimum_size'
    MINIMUM_COUNT_PARAM = 'minimum_count'

    TYPE_COUNT = 'count'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.vocab_size = parameters.get(self.VOCAB_SIZE_PARAM, 1000)
        self.minimum_df = parameters.get(self.MINIMUM_DF_PARAM, 5)
        self.minimum_tf = parameters.get(self.MINIMUM_TF_PARAM, 1)
        self.minimum_size = parameters.get(self.MINIMUM_VECTOR_SIZE_PARAM, 3)
        self.minimum_count = parameters.get(self.MINIMUM_COUNT_PARAM, 0)

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_COUNT)

        if self.type == 'count':
            self.type = 'TF-IDF'
        else:
            self.type = "BoW"

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = parameters.get(self.ALIAS_PARAM, "")
        if len(self.alias) == 0:
            self.alias = 'features_{}'.format(self.type)

        self.input_data = self.named_inputs['input data']

        self.vocab = self.named_outputs.get('vocabulary', 'tmp')
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.has_import = "from functions.text.convert_words_to_vector " \
                              "import ConvertWordstoVectorOperation\n"

    def get_optimization_information(self):
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': True,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': True,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        params = dict()
        params['attributes'] = {attrib}
        params['alias'] = '{alias}'
        params['minimum_tf'] = {tf}
        params['minimum_df'] = {df}
        params['size'] = {size}
        params['mode'] = '{mode}'
        
        params, vocabulary = ConvertWordstoVectorOperation().preprocessing(
        {IN}, params, numFrag)
        conf.append([params, vocabulary])
        """.format(IN=self.input_data, size=self.vocab_size,
                   df=self.minimum_df, tf=self.minimum_tf,
                   attrib=self.attributes, alias=self.alias, mode=self.type)
        return code

    def generate_optimization_code(self):
        """Generate code."""
        code = """
        settings, vocabulary = conf_X
        {output} = ConvertWordstoVectorOperation()
        .transform_serial({input}, vocabulary, settings)
        """.format(output=self.output,
                   input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
            params = dict()
            params['attributes'] = {attrib}
            params['alias'] = '{alias}'
            params['minimum_tf'] = {tf}
            params['minimum_df'] = {df}
            params['size'] = {size}
            params['mode'] = '{mode}'
            params, {voc} = ConvertWordstoVectorOperation().preprocessing(
            {IN}, params, numFrag)
            {OUT} = ConvertWordstoVectorOperation().transform(
            {IN}, {voc}, params, numFrag)
            """.format(OUT=self.output, voc=self.vocab, IN=self.input_data,
                       size=self.vocab_size, df=self.minimum_df,
                       tf=self.minimum_tf, attrib=self.attributes,
                       alias=self.alias, mode=self.type)

        return dedent(code)
