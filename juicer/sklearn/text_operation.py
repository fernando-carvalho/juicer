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
            raise ValueError(_('Invalid type for '
                               'operation Tokenizer: {}').format(self.type))

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

        self.expression_param = parameters.get(self.EXPRESSION_PARAM, '\s+')
        if len(self.expression_param) == 0:
            self.expression_param =  '\s+'

        self.has_code = len(self.named_inputs) == 1

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        """Generate code."""
        code = """
            min_token_length = {min_token}
            {OUT} = {IN}
            result = []
            for field in {IN}['{att}'].values:
                row = []
                toks = re.split('{expression}', field)
                for t in toks:
                    if len(t) > min_token_length:
                        row.append(t.lower())
                result.append(row)
                
            {OUT}['{alias}'] = result
            """.format(OUT=self.output, min_token=self.min_token_lenght,
                       IN=self.named_inputs['input data'],
                       att=self.attributes[0], alias=self.alias[0],
                       expression=self.expression_param)

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
            self.STOP_WORD_ATTRIBUTE_PARAM, '')

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
        self.lang = self.parameters.get('language', 'english')

    def generate_code(self):
        """Generate code."""
        code = """
        stop_words = []
        """
        if len(self.lang)>0:
            code += """
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = stopwords.words('{language}')
        """.format(language=self.lang)

        code += """
        {OUT} = {IN}
        tokens = {IN}['{att}'].values
        rows = []
        for row in tokens:
            filtered_sentence = []
            for w in row:
                if w not in stop_words:
                    filtered_sentence.append(w)
            rows.append(filtered_sentence)
        
        {OUT}['{alias}'] = rows
        """.format(att=self.attributes[0],
                   att_stop=self.stop_word_attribute,
                   alias=self.alias[0], case=self.sw_case_sensitive,
                   OUT=self.output, IN=self.named_inputs['input data'],
                   stoplist=self.stop_word_list, sw=self.stopwords_input)

        return dedent(code)


class GenerateNGramsOperation(Operation):
    """ Generates N-Grams from word vectors
    An n-gram is a sequence of n tokens (typically words) for some
    integer n. The NGram class can be used to transform input features
    into n-grams.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    N_PARAM = 'n'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)
        if self.N_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.N_PARAM, self.__class__))

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.n = int(self.parameters.get(self.N_PARAM, 2))
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _ngram.
        self.alias = [x[1] or '{}_ngram'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = dedent("""
            {output} = {input}
            
            from nltk.util import ngrams
            def word_grams(words, min={n}):
                max = len(words)
                s = []
                for n in range(min, max):
                    for ngram in ngrams(words, n):
                        s.append(' '.join(str(i.encode('utf-8')) for i in ngram))
                return s
            
            rows = {input}['{att}'].values
            grams = []
            for r in rows:
                grams.append(word_grams(r))
                
            {output}['{alias}'] = grams
            """).format(att=self.attributes[0], alias=self.alias[0],
                        n=self.n, input=input_data, output=self.output)

        return code

class WordToVectorOperation(Operation):
    """
    Word2Vec is an Estimator which takes sequences of words that
    represents documents and trains a Word2VecModel. The model is
    a Map(String, Vector) essentially, which maps each word to an
    unique fix-sized vector. The Word2VecModel transforms each
    documents into a vector using the average of all words in the
    document, which aims to other computations of documents such
    as similarity calculation consequencely.
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
    TYPE_WORD2VEC = 'word2vec'
    TYPE_HASHING_TF = 'hashing_tf'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_COUNT)
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _vect.
        self.alias = [x[1] or '{}_vect'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.vocab_size = parameters.get(self.VOCAB_SIZE_PARAM, 1000) or 1000
        self.minimum_df = parameters.get(self.MINIMUM_DF_PARAM, 1) or 1.0
        self.minimum_tf = parameters.get(self.MINIMUM_TF_PARAM, 1) or 1.0

        self.minimum_size = parameters.get(self.MINIMUM_VECTOR_SIZE_PARAM, 100)
        self.minimum_count = parameters.get(self.MINIMUM_COUNT_PARAM, 5)

        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))
        self.vocab = self.named_outputs.get('vocabulary',
                                            'vocab_task_{}'.format(self.order))
        self.output_model = self.named_outputs.get(
            'vector-model', 'vector_model_{}'.format(self.order))

        if self.type not in [self.TYPE_HASHING_TF, self.TYPE_WORD2VEC,
                             self.TYPE_COUNT]:
            raise ValueError(
                _("Invalid type '{}' for task {}").format(self.type,
                                                          self.__class__))

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.vocab, self.output_model])

    def generate_code(self):
        input_data = self.named_inputs['input data']

        # max_df is used for removing terms that appear too frequently
        if self.type == self.TYPE_COUNT:
            code = dedent("""
            from sklearn.feature_extraction.text import CountVectorizer
            corpus = {input}['{att}'].apply(lambda x: ' '.join(x))
            bow_transformer = CountVectorizer(min_df={min_df}, 
            binary=False, max_features={vocab_size})
                            
            bow_transformer.fit(corpus)
            {out} = {input}
            vector = bow_transformer.transform(corpus).toarray().tolist()
            {out}['{alias}'] = vector
            {model} = bow_transformer
            {vocab} = bow_transformer.get_feature_names()
            """.format(
                    input=input_data,
                    out=self.output,
                    att=self.attributes[0],
                    alias=self.alias[0],
                    min_df=self.minimum_df,
                    vocab_size=self.vocab_size,
                    model=self.output_model,
                    vocab=self.vocab))

        elif self.type == self.TYPE_HASHING_TF:
            code = dedent("""
            from sklearn.feature_extraction.text import HashingVectorizer
            corpus = {input}['{att}'].apply(lambda x: ' '.join(x))
            model = HashingVectorizer(binary=False, n_features={vocab_size})
                
            model.fit(corpus)
            {out} = {input}
            vector = model.transform(corpus).toarray().tolist()
            {out}['{alias}'] = vector
            {model} = model

            # There is no vocabulary in this type of transformer
            {vocab} = {{}}
            """.format(
                    input=input_data,
                    out=self.output,
                    att=self.attributes[0],
                    alias=self.alias[0],
                    vocab_size=self.vocab_size,
                    model=self.output_model,
                    vocab=self.vocab))
        elif self.type == self.TYPE_WORD2VEC:
            # @FIXME Check
            code = dedent("""
                """.format(
                self.attributes,
                input=input_data,
                out=self.output,
                aliases=json.dumps(zip(self.attributes, self.alias)),
                size=self.minimum_size, count=self.minimum_count,
                vocab=self.vocab,
                model=self.output_model
            ))

        else:
            raise ValueError(
                _("Invalid type '{}' for task {}").format(self.type,
                                                          self.__class__))
        return code
