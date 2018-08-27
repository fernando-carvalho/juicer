# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from textwrap import dedent

from juicer.operation import Operation


class FrequentItemSetOperation(Operation):
    """
    Finds frequent item sets in a data source composed by transactions.
    Current implementation uses FP-Growth.
    """
    MIN_SUPPORT_PARAM = 'min_support'
    ATTRIBUTE_PARAM = 'attribute'
    CONFIDENCE_PARAM = 'min_confidence'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.MIN_SUPPORT_PARAM not in parameters:
            raise ValueError(_(
                'Support must be informed for classifier {}').format(
                self.__class__))

        self.min_support = float(parameters.get(self.MIN_SUPPORT_PARAM))
        if self.min_support < .0001 or self.min_support > 1.0:
            raise ValueError('Support must be greater or equal '
                             'to 0.0001 and smaller than 1.0')

        self.output = self.named_outputs.get(
            'output data', 'freq_items_{}'.format(self.order))

        self.rules = self.named_outputs.get(
            'rules output', 'rules_{}'.format(self.order))

        self.attribute = parameters.get(self.ATTRIBUTE_PARAM)
        if not self.attribute:
            raise ValueError(
                'Missing parameter {}'.format(self.ATTRIBUTE_PARAM))
        self.attribute = self.attribute[0]

        self.has_code = len(self.named_inputs) == 1

        self.confidence = float(parameters.get(self.CONFIDENCE_PARAM, 0.9))

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.rules])

    def get_data_out_names(self, sep=','):
        return self.get_output_names(sep)

    def generate_code(self):
        code = u"""
            try:
                from pyspark.ml.fpm import FPGrowth
                algorithm = FPGrowth(itemsCol="{attr}",
                    minSupport={support}, minConfidence={confidence})

                # Evaluate if using cache is a good idea
                {input}.cache()
                size = float({input}.count())
                model = algorithm.fit({input})

                emit_event(name='update task', message='{model_trained}',
                           status='RUNNING', identifier='{task_id}')
                {output} = model.freqItemsets.withColumn('relative_support',
                    functions.col('freq') / size)
                {rules} = model.associationRules
            except Exception as e:
                sparkException = 'org.apache.spark.SparkException'
                if hasattr(e, 'java_exception') and \
                        e.java_exception.getClass().getName() == sparkException:
                    cause = e.java_exception.getCause()
                    not_unique = 'Items in a transaction must be unique'
                    if cause and cause.getMessage().find(not_unique) > -1:
                        raise ValueError('{not_unique}')
                    else:
                        raise
                else:
                    raise
        """.format(input=self.named_inputs['input data'],
                   support=self.min_support,
                   output=self.output,
                   attr=self.attribute,
                   task_id=self.parameters['task']['id'],
                   rules=self.rules,
                   not_unique=_('Items in a transaction must be unique'),
                   confidence=self.confidence,
                   model_trained=_('Model trained'))

        return dedent(code)


class AssociationRulesOperation(Operation):
    """
    AssociationRules constructs rules that have a single item as the consequent.
    Current implementation uses FP-Growth.
    """
    CONFIDENCE_PARAM = 'confidence'
    RULES_COUNT_PARAM = 'rules_count'
    ATTRIBUTE_PARAM = 'attribute'
    FREQ_PARAM = 'freq'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.confidence = float(parameters.get(self.CONFIDENCE_PARAM, 0.9))
            self.rules_count = parameters.get(self.RULES_COUNT_PARAM, 200)
            self.attribute = parameters.get(self.ATTRIBUTE_PARAM, ['items'])
            self.freq = parameters.get(self.FREQ_PARAM, ['freq'])

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        code = u"""
            try:
                ext_pkg = spark_session._jvm.br.ufmg.dcc.lemonade.ext.fpm
                associative_impl = ext_pkg.LemonadeAssociativeRules()
            except TypeError as te:
                if 'JavaPackage' in te.message:
                    raise ValueError('{required_pkg}')
                else:
                    raise
            items = {input}
            java_df = associative_impl.run(spark_session._jsparkSession,
                items._jdf, {confidence}, '{attr}', '{freq}')

            {output} = DataFrame(java_df, spark_session)
        """.format(input=self.named_inputs['input data'],
                   confidence=self.confidence,
                   output=self.output,
                   rules_count=self.rules_count,
                   attr=self.attribute[0],
                   freq=self.freq[0],
                   task_id=self.parameters['task']['id'],
                   model_trained=_('Model trained'),
                   required_pkg=_('Required Lemonade Spark Extensions '
                                  'not found in CLASSPATH.')
                   )

        return dedent(code)


class SequenceMiningOperation(Operation):
    """
    Sequential pattern mining algorithm (PrefixSpan)
    """
    MIN_SUPPORT_PARAM = 'min_support'
    MAX_PATTERN_LENGTH_PARAM = 'max_pattern_length'
    ATTRIBUTE_PARAM = 'attribute'
    FREQ_PARAM = 'freq'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.min_support = float(
                parameters.get(self.MIN_SUPPORT_PARAM, 0.1))
            self.max_length = parameters.get(self.MAX_PATTERN_LENGTH_PARAM, 10)
            self.attribute = parameters.get(self.ATTRIBUTE_PARAM, ['items'])
            self.freq = parameters.get(self.FREQ_PARAM, ['freq'])

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        code = dedent(u"""
            from pyspark.sql import DataFrame
            try:
                # noinspection PyProtectedMember
                ext_pkg = spark_session._jvm.br.ufmg.dcc.lemonade.ext.fpm
                prefix_span_impl = ext_pkg.LemonadePrefixSpan()
            except TypeError as te:
                if 'JavaPackage' in te.message:
                    raise ValueError('{required_pkg}')
                else:
                    raise
            sequences = {input}
            meta = json.loads(sequences.schema[str('{attr}')].json())
            if meta['type'] != 'array' and meta['type'][
                'elementType'] != 'array':
                elem_type = sequences.schema[str('{attr}')].dataType.elementType
                sequences = sequences.select(functions.udf(
                    lambda x: [[v] for v in x],
                        types.ArrayType(types.ArrayType(elem_type)))(
                        '{attr}').alias(
                        '{attr}'))
            # noinspection PyProtectedMember
            java_df = prefix_span_impl.run(spark_session._jsparkSession,
                sequences._jdf, {min_support}, {max_length}, '{attr}',
                '{freq}')

            {output} = DataFrame(java_df, spark_session)

        """.format(input=self.named_inputs['input data'],
                   min_support=self.min_support,
                   max_length=self.max_length,
                   output=self.output,
                   attr=self.attribute[0],
                   freq=self.freq[0],
                   required_pkg=_('Required Lemonade Spark Extensions '
                                  'not found in CLASSPATH.')
                   ))
        return code
