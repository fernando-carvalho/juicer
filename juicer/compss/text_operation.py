# -*- coding: utf-8 -*-

import ast
import pprint
from textwrap import dedent
from juicer.operation import Operation
from itertools import izip_longest

class TokenizerOperation(Operation):
    """
    Tokenization is the process of taking text (such as a sentence) and
    breaking it into individual terms (usually words). A simple Tokenizer
    class provides this functionality.

    REVIEW: 2017-10-20
    OK - Juicer / Tahiti / implementation
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
            raise ValueError(
                _('Invalid type for operation Tokenizer: {}').format(self.type))

        self.expression_param = parameters.get(self.EXPRESSION_PARAM, '\s+')

        self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").\
                    format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_tok'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.has_import = \
                "from functions.text.Tokenizer import TokenizerOperation\n"

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):

        if self.type == self.TYPE_SIMPLE:
            code =  """
                settings = dict()
                settings['min_token_length'] = {min_token}
                settings['type'] = 'simple'
                settings['attributes'] = {att}
                settings['alias'] = {alias}
                {output} = TokenizerOperation({input}, settings, numFrag)
            """.format( output    = self.output,
                        min_token = self.min_token_lenght,
                        input     = self.named_inputs['input data'],
                        att = self.attributes,
                        alias = self.alias
                        )
        else:
            code =  """
                settings = dict()
                settings['min_token_length'] = {min_token}
                settings['type'] = 'regex'
                setting['expression'] = '{expression}'
                settings['attributes'] = {att}
                settings['alias'] = {alias}
                {output} = TokenizerOperation({input}, settings, numFrag)
            """.format( output    = self.output,
                        min_token = self.min_token_lenght,
                        expression = self.expression_param,
                        input     = self.named_inputs['input data'],
                        att = self.attributes,
                        alias = self.alias
                        )

        return dedent(code)

class RemoveStopWordsOperation(Operation):
    """
    Stop words are words which should be excluded from the input,
    typically because the words appear frequently and don’t carry
    as much meaning.

    REVIEW: 2017-10-20
    OK - Juicer / Tahiti / implementation
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
                _("Parameter '{}' must be informed for task {}").\
                    format(self.ATTRIBUTES_PARAM, self.__class__))

        self.stop_word_attribute = self.parameters.get(
            self.STOP_WORD_ATTRIBUTE_PARAM, '')

        self.stop_word_list = [s.strip() for s in
               self.parameters.get(self.STOP_WORD_LIST_PARAM,'').split(',')]

        self.alias = parameters.get(self.ALIAS_PARAM, 'tokenized_rm')

        self.sw_case_sensitive = self.parameters.get(
            self.STOP_WORD_CASE_SENSITIVE_PARAM, 'False')

        self.stopwords_input =  self.named_inputs.get('stop words', [])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.has_code = 'input data' in self.named_inputs
        if self.has_code:
            self.has_import = \
        "from functions.text.RemoveStopWords import RemoveStopWordsOperation\n"


    def generate_code(self):

        code =  """
            settings = dict()
            settings['attribute'] = {att}
            settings['alias']     = '{alias}'
            settings['attribute-stopwords'] = {att_stop}
            settings['case-sensitive']      = {case}
            settings['news-stops-words']    = {stopwords_list}

            {output}=RemoveStopWordsOperation({input}, settings, {sw}, numFrag)
            """.format(att      = self.attributes,
                       att_stop = self.stop_word_attribute,
                       alias    = self.alias,
                       case     = self.sw_case_sensitive,
                       output = self.output,
                       input  = self.named_inputs['input data'],
                       stopwords_list = self.stop_word_list,
                       sw = self.stopwords_input
                       )

        return dedent(code)

class WordToVectorOperation(Operation):
    """
    Can be used Bag of Words transformation or TF-IDF.

    REVIEW: 2017-10-20
    OK - Juicer / Tahiti / implementation
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
                _("Parameter '{}' must be informed for task {}").\
                    format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = parameters.get(self.ALIAS_PARAM,"")
        if len(self.alias) == 0:
            self.alias = 'features_{}'.format(self.type)

        self.input_data = self.named_inputs['input data']

        self.vocab = self.named_outputs.get('vocabulary', 'tmp')
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.has_import = "from functions.text.ConvertWordstoVector " \
                              "import ConvertWordstoVectorOperation\n"



    def generate_code(self):

        code =  """
            params = dict()
            params['attributes'] = {attrib}
            params['alias']      = '{alias}'
            params['minimum_tf'] = {tf}
            params['minimum_df'] = {df}
            params['size']       = {size}
            params['mode']       = '{mode}'
            {output}, {vocabulary} = ConvertWordstoVectorOperation({input}, params, numFrag)
            """.format(output    = self.output,
                       vocabulary= self.vocab,
                       input = self.input_data,
                       size  = self.vocab_size,
                       df    = self.minimum_df,
                       tf    = self.minimum_tf,
                       attrib = self.attributes,
                       alias  = self.alias,
                       mode  = self.type
                       )


        return dedent(code)