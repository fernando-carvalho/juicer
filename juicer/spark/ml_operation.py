# coding=utf-8
from __future__ import unicode_literals

import json
import logging
import string
from itertools import izip_longest
from textwrap import dedent

from juicer.operation import Operation, ReportOperation
from juicer.service import limonero_service
from juicer.service.limonero_service import query_limonero

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class SafeDict(dict):
    # noinspection PyMethodMayBeStatic
    def __missing__(self, key):
        return '{' + key + '}'  # pragma: no cover


class FeatureIndexerOperation(Operation):
    """
    A label indexer that maps a string attribute of labels to an ML attribute of
    label indices (attribute type = STRING) or a feature transformer that merges
    multiple attributes into a vector attribute (attribute type = VECTOR). All
    other attribute types are first converted to STRING and them indexed.
    """
    ATTRIBUTES_PARAM = 'attributes'
    TYPE_PARAM = 'indexer_type'
    ALIAS_PARAM = 'alias'
    MAX_CATEGORIES_PARAM = 'max_categories'

    TYPE_STRING = 'string'
    TYPE_VECTOR = 'vector'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_STRING)
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]

        self.max_categories = int(
            parameters.get(self.MAX_CATEGORIES_PARAM, 0) or 20)
        if not (self.max_categories >= 0):
            msg = _(
                "Parameter '{}' must be in range [x>=0] for task {}").format(
                self.MAX_CATEGORIES_PARAM, __name__)
            raise ValueError(msg)

        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name suffixed by _indexed.
        self.alias = [x[1] or '{}_indexed'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

    def generate_code(self):
        input_data = self.named_inputs['input data']
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        models = self.named_outputs.get('indexer models',
                                        'models_task_{}'.format(self.order))
        if self.type == self.TYPE_STRING:
            code = dedent("""
                col_alias = dict(tuple({alias}))
                indexers = [feature.StringIndexer(
                    inputCol=col, outputCol=alias, handleInvalid='keep')
                             for col, alias in col_alias.items()]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=indexers)
                {models} = dict([(c, indexers[i].fit({input})) for i, c in
                                 enumerate(col_alias.values())])

                # Spark ML 2.0.1 do not deal with null in indexer.
                # See SPARK-11569
                {input}_without_null = {input}.na.fill(
                    'NA', subset=col_alias.keys())

                {out} = pipeline.fit({input}_without_null).transform(
                    {input}_without_null)
            """.format(input=input_data, out=output, models=models,
                       alias=json.dumps(zip(self.attributes, self.alias),
                                        indent=None)))
        elif self.type == self.TYPE_VECTOR:
            code = dedent("""
                col_alias = dict({alias})
                indexers = [feature.VectorIndexer(maxCategories={max_categ},
                                inputCol=col, outputCol=alias)
                                    for col, alias in col_alias.items()]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=indexers)
                {models} = dict([(col, indexers[i].fit({input})) for i, col in
                                enumerate(col_alias.values())])
                labels = None

                # Spark ML 2.0.1 do not deal with null in indexer.
                # See SPARK-11569
                {input}_without_null = {input}.na.fill('NA',
                    subset=col_alias.keys())

                {out} = pipeline.fit({input}_without_null).transform(
                    {input}_without_null)
            """.format(input=input_data, out=output,
                       alias=json.dumps(zip(self.attributes, self.alias)),
                       max_categ=self.max_categories,
                       models=models))
        else:
            # Only if the field be open to type
            raise ValueError(
                _("Parameter type has an invalid value {}").format(self.type))

        code += dedent(
            """
            # Update metadata information (may be used in other operations)
            for k, v in col_alias.items():
                field_metadata = {out}.schema[str(v)].metadata
                if not 'lemon_attr' in field_metadata:
                    field_metadata['lemon_attr'] = {{}}
                field_metadata['lemon_attr']['indexed_from'] = k
                {out} = {out}.withColumn(
                    v, functions.col(v).alias('', metadata=field_metadata))

            # Store indexer models in cache. Some operations may need to use
            # them. If an indexer is created more than once for attributes with
            # the same name, the last executed will overwrite the previous one.
            # FIXME: evaluate how to handle this conflict.
            if 'indexer' not in cached_state:
                cached_state['indexers'] = {{}}
            for name, model in {models}.items():
                cached_state['indexers'][name] = model
        """.format(models=models, out=output))

        return code

    def get_output_names(self, sep=','):
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))
        models = self.named_outputs.get('indexer models',
                                        'models_task_{}'.format(self.order))
        return sep.join([output, models])

    def get_data_out_names(self, sep=','):
        return self.named_outputs.get('output data',
                                      'out_task_{}'.format(self.order))


class IndexToStringOperation(Operation):
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    ORIGINAL_NAMES_PARAM = 'original_names'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        if self.ORIGINAL_NAMES_PARAM in parameters:
            self.original_names = parameters.get(self.ORIGINAL_NAMES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ORIGINAL_NAMES_PARAM, self.__class__))
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]

        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_str'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]
        self.has_code = any(
            ['input data' in self.named_inputs, self.contains_results()])

    def generate_code(self):
        input_data = self.named_inputs['input data']
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))
        models = self.named_inputs.get('indexer models')
        if models is not None:
            labels = '{models}[original_names[i]].labels'.format(models=models)
        else:
            labels = '[]'

        code = """
            original_names = {original_names}
            col_alias = dict({alias})
            converter = [feature.IndexToString(inputCol=col,
                                               outputCol=alias,
                                               labels={labels})
                        for i, (col, alias) in enumerate(col_alias.items())]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=converter)
            {out} = pipeline.fit({input}).transform({input})
        """.format(input=input_data, out=output, labels=labels,
                   original_names=json.dumps(
                       [n.strip() for n in self.original_names]),
                   alias=json.dumps(zip(self.attributes, self.alias),
                                    indent=None))
        return dedent(code)


class OneHotEncoderOperation(Operation):
    """
    One hot encoding transforms categorical features to a format that works
    better with classification and regression algorithms.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]

        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name suffixed by _indexed.
        self.alias = [x[1] or '{}_onehotenc'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

    def generate_code(self):
        input_data = self.named_inputs['input data']
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        code = """
            col_alias = dict({aliases})
            encoders = [feature.OneHotEncoder(inputCol=col, outputCol=alias,
                            dropLast=True)
                        for col, alias in col_alias.items()]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=encoders)
            {out} = pipeline.fit({input}).transform({input})
            """.format(input=input_data, out=output,
                       aliases=json.dumps(zip(self.attributes, self.alias),
                                          indent=None))
        return dedent(code)


class FeatureAssemblerOperation(Operation):
    """
    A feature transformer that merges multiple attributes into a vector
    attribute.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.parameters = parameters
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.alias = parameters.get(self.ALIAS_PARAM, 'features')

        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])

    def generate_code(self):
        input_data = self.named_inputs['input data']
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        code = """
            assembler = feature.VectorAssembler(inputCols={0}, outputCol="{1}")
            {3}_without_null = {3}.na.drop(subset={0})
            {2} = assembler.transform({3}_without_null)
        """.format(json.dumps(self.attributes), self.alias, output,
                   input_data)

        return dedent(code)


class ApplyModelOperation(Operation):
    NEW_ATTRIBUTE_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
            [len(self.named_inputs) == 2, self.contains_results()])

        self.new_attribute = parameters.get(self.NEW_ATTRIBUTE_PARAM,
                                            'new_attribute')
        if not self.has_code and len(self.named_outputs) > 0:
            raise ValueError(
                _('Model is being used, but at least one input is missing'))

    def get_data_out_names(self, sep=','):
        return self.output

    def generate_code(self):
        input_data1 = self.named_inputs['input data']
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        model = self.named_inputs.get(
            'model', 'model_task_{}'.format(self.order))

        code = dedent("""
            # Depends on model params = {{'predictionCol': '{new_attr}'}}
            params = {{}}
            {out} = {in2}.transform({in1}, params)
            """.format(out=output, in1=input_data1, in2=model,
                       new_attr=self.new_attribute))

        return dedent(code)


class EvaluateModelOperation(Operation):
    PREDICTION_ATTRIBUTE_PARAM = 'prediction_attribute'
    LABEL_ATTRIBUTE_PARAM = 'label_attribute'
    METRIC_PARAM = 'metric'

    METRIC_TO_EVALUATOR = {
        'areaUnderROC': (
            'evaluation.BinaryClassificationEvaluator', 'rawPredictionCol'),
        'areaUnderPR': (
            'evaluation.BinaryClassificationEvaluator', 'rawPredictionCol'),
        'f1': ('evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'weightedPrecision': (
            'evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'weightedRecall': (
            'evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'accuracy': (
            'evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'rmse': ('evaluation.RegressionEvaluator', 'predictionCol'),
        'mse': ('evaluation.RegressionEvaluator', 'predictionCol'),
        'mae': ('evaluation.RegressionEvaluator', 'predictionCol'),
        'r2': ('evaluation.RegressionEvaluator', 'predictionCol'),
    }

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        # @FIXME: validate if metric is compatible with Model using workflow

        self.prediction_attribute = (parameters.get(
            self.PREDICTION_ATTRIBUTE_PARAM) or [''])[0]
        self.label_attribute = (parameters.get(
            self.LABEL_ATTRIBUTE_PARAM) or [''])[0]
        self.metric = parameters.get(self.METRIC_PARAM) or ''

        if all([self.prediction_attribute != '', self.label_attribute != '',
                self.metric != '']):
            pass
        else:
            msg = \
                _("Parameters '{}', '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.PREDICTION_ATTRIBUTE_PARAM, self.LABEL_ATTRIBUTE_PARAM,
                self.METRIC_PARAM, self.__class__))
        if self.metric in self.METRIC_TO_EVALUATOR:
            self.evaluator = self.METRIC_TO_EVALUATOR[self.metric][0]
            self.param_prediction_arg = self.METRIC_TO_EVALUATOR[self.metric][1]
        else:
            raise ValueError(_('Invalid metric value {}').format(self.metric))

        self.has_code = any([(
            (len(self.named_inputs) > 0 and len(self.named_outputs) > 0) or
            (self.named_outputs.get('evaluator') is not None) or
            ('input data' in self.named_inputs)
        ), self.contains_results()])

        self.model = self.named_inputs.get('model')
        self.model_out = self.named_outputs.get(
            'evaluated model', 'model_task_{}'.format(self.order))

        self.evaluator_out = self.named_outputs.get(
            'evaluator', 'evaluator_task_{}'.format(self.order))
        if not self.has_code and self.named_outputs.get(
                'evaluated model') is not None:
            raise ValueError(
                _('Model is being used, but at least one input is missing'))

        self.supports_cache = False

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return ''

    def generate_code(self):
        if not self.has_code:
            return ''
        else:
            display_text = self.parameters['task']['forms'].get(
                'display_text', {'value': 1}).get('value', 1) in (1, '1')
            display_image = self.parameters['task']['forms'].get(
                'display_image', {'value': 1}).get('value', 1) in (1, '1')

            code = [dedent("""
                emit = functools.partial(
                    emit_event, name='update task',
                    status='RUNNING', type='TEXT',
                    identifier='{task_id}',
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id},
                    task={{'id': '{task_id}'}},
                    title='{title}')
                metric_value = 0.0
                display_text = {display_text}
                display_image = {display_image}
                metric = '{metric}'
                label_col = '{label_attr}'
                prediction_col = str('{prediction_attr}')

                stages = []
                requires_pipeline = False
                if not dataframe_util.is_numeric({input}.schema, label_col):
                    emit(message=_('Label attribute is categorical, it will be '
                            'implicitly indexed as string.'),)
                    final_label = '{{}}_tmp'.format(label_col)
                    final_prediction = '{{}}_tmp'.format(prediction_col)

                    indexer = feature.StringIndexer(
                                inputCol=label_col, outputCol=final_label,
                                handleInvalid='keep')
                    label_indexer = indexer.fit({input})
                    {input} = label_indexer.transform({input})

                    {input} = label_indexer.transform({input},{{
                        label_indexer.inputCol: prediction_col,
                        label_indexer.outputCol: final_prediction
                    }})

                    label_col =  final_label
                    prediction_col = final_prediction

                """)]
            if self.metric in ['areaUnderROC', 'areaUnderPR']:
                self._get_code_for_area_metric(code)
            elif self.metric in ['f1', 'weightedPrecision', 'weightedRecall',
                                 'accuracy']:
                self._get_code_for_classification_metrics(code)
            elif self.metric in ['rmse', 'mae', 'mse']:
                self._get_code_for_regression_metrics(code)

            self._get_code_for_summary(code)

            # Common for all metrics!
            code.append(dedent("""
            {model_output} = ModelsEvaluationResultList(
                    [{model}], {model}, '{metric}', metric_value)

            {metric} = metric_value
            {model_output} = None
            """))

            code = "\n".join(code).format(
                model_output=self.model_out,
                model=self.model,
                input=self.named_inputs['input data'],
                metric=self.metric,
                evaluator_out=self.evaluator_out,
                task_id=self.parameters['task_id'],
                operation_id=self.parameters['operation_id'],
                title=_('Evaluation result'),
                display_text=display_text,
                display_image=display_image,
                prediction_attr=self.prediction_attribute,
                label_attr=self.label_attribute,
                headers=[_('Metric'), _('Value')],
                evaluator=self.evaluator,
                prediction_arg=self.param_prediction_arg,
            )
            return dedent(code)

    @staticmethod
    def _get_code_for_classification_metrics(code):
        """
        Generate code for other classification metrics besides those related to
        area.
        """
        code.append(dedent("""
            label_prediction = {input}.select(
                functions.col(prediction_col).cast('Double'),
                functions.col(label_col).cast('Double'))
            evaluator = MulticlassMetrics(label_prediction.rdd)
            if metric == 'f1':
                metric_value = evaluator.weightedFMeasure()
            elif metric == 'weightedPrecision':
                metric_value = evaluator.weightedPrecision
            elif metric == 'weightedRecall':
                metric_value = evaluator.weightedRecall
            elif metric == 'accuracy':
                metric_value = evaluator.accuracy

            if display_image:
                # Test if feature indexer is in global cache, because
                # strings must be converted into numbers in order tho
                # run algorithms, but they are cooler when displaying
                # results.
                all_labels = [l for l in {input}.schema[
                    str(label_col)].metadata.get(
                        'ml_attr', {{}}).get('vals', {{}}) if l[0] != '_']

                if not all_labels:
                    all_labels = sorted(
                        [x[0] for x in label_prediction.select(
                                label_col).distinct().collect()])

                content = ConfusionMatrixImageReport(
                    cm=evaluator.confusionMatrix().toArray(),
                    classes=all_labels,)

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content.generate(),
                    type='IMAGE', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})

            if display_text:
                headers = {headers}
                rows = [
                    ['F1', evaluator.weightedFMeasure()],
                    ['Weighted Precision', evaluator.weightedPrecision],
                    ['Weighted Recall', evaluator.weightedRecall],
                    ['Accuracy', evaluator.accuracy],
                ]

                content = SimpleTableReport(
                        'table table-striped table-bordered table-sm',
                        headers, rows,
                        title='{title}')

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content.generate(),
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
        """))

    @staticmethod
    def _get_code_for_area_metric(code):
        """
        Code for the evaluator when metric is related to the area
        """
        code.append(dedent("""
            evaluator = evaluation.BinaryClassificationEvaluator(
                {prediction_arg}=prediction_col,
                labelCol=label_col,
                metricName=metric)
            metric_value = evaluator.evaluate({input})
            if display_text:
                result = '<h4>{{}}: {{}}</h4>'.format('{metric}',
                    metric_value)

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=result,
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
        """))

    @staticmethod
    def _get_code_for_regression_metrics(code):
        """
        Code for the evaluator when metric is related to regression
        """
        code.append(dedent("""
            df = {input}
            if not isinstance({input}.schema[str(prediction_col)].dataType,
                (types.DoubleType, types.FloatType)):
                df = {input}.withColumn(prediction_col,
                    {input}[prediction_col].cast('double'))

            evaluator = evaluation.RegressionEvaluator(
                {prediction_arg}=prediction_col,
                labelCol=label_col,
                metricName=metric)
            metric_value = evaluator.evaluate(df)
            if display_text:
                result = '<h4>{{}}: {{}}</h4>'.format('{metric}',
                    metric_value)

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=result,
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
        """))

    @staticmethod
    def _get_code_for_summary(code):
        """
        Return code for model's summary (test if it is present)
        """
        i18n_dict = SafeDict(
            join_plot_title=_('Prediction versus Residual'),
            join_plot_x_title=_('Prediction'),
            join_plot_y_title=_('Residual'),

            plot_title=_('Actual versus Prediction'),
            plot_x_title=_('Actual'),
            plot_y_title=_('Prediction'),
        )
        partial_code = """
            summary = getattr({model}, 'summary', None)
            if summary:
                if summary.numInstances < 2000 and display_image:
                    predictions = [r[prediction_col] for r in
                        summary.predictions.collect()]

                    if isinstance({model}, LinearRegressionModel):
                        residuals_col = 'residuals'
                        df_residual = summary.residuals
                    else:
                        residuals_col = 'devianceResiduals'
                        df_residual = summary.residuals()
                        
                    residuals = [r[residuals_col] for r in
                        df_residual.collect()]
                    pandas_df = pd.DataFrame.from_records(
                        [
                            dict(prediction=x[0], residual=x[1])
                                for x in zip(predictions, residuals)
                        ]
                    )

                    report = SeabornChartReport()
                    emit_event(
                        'update task', status='COMPLETED',
                        identifier='{task_id}',
                        message=report.jointplot(pandas_df, 'prediction',
                            'residual', '{join_plot_title}',
                            '{join_plot_x_title}', '{join_plot_y_title}'),
                        type='IMAGE', title='{join_plot_title}',
                        task=dict(id='{task_id}'),
                        operation=dict(id={operation_id}),
                        operation_id={operation_id})

                    report2 = MatplotlibChartReport()

                    actual = []
                    predicted = []
                    for r in df.select([label_col, prediction_col]).collect():
                        actual.append(r[label_col])
                        predicted.append(r[prediction_col])

                    identity = range(int(max(actual[-1], predicted[-1])))
                    emit_event(
                         'update task', status='COMPLETED',
                        identifier='{task_id}',
                        message=report2.plot(
                            '{plot_title}',
                            '{plot_x_title}',
                            '{plot_y_title}',
                            identity, identity, 'r.',
                            actual, predicted,'b.',),
                        type='IMAGE', title='{join_plot_title}',
                        task=dict(id='{task_id}'),
                        operation=dict(id={operation_id}),
                        operation_id={operation_id})

                summary_rows = []
                for p in dir(summary):
                    if not p.startswith('_') and p != "cluster":
                        try:
                            summary_rows.append(
                                [p, getattr(summary, p)])
                        except Exception as e:
                            summary_rows.append([p, e.message])
                summary_content = SimpleTableReport(
                    'table table-striped table-bordered', [],
                    summary_rows,
                    title='Summary')
                emit_event('update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=summary_content.generate(),
                    type='HTML', title='{title}',
                    task=dict(id='{task_id}'),
                    operation=dict(id={operation_id}),
                    operation_id={operation_id})
        """
        code.append(
            dedent(string.Formatter().vformat(partial_code, (), i18n_dict)))


class ModelsEvaluationResultList:
    """ Stores a list of ModelEvaluationResult """

    def __init__(self, models, best, metric_name, metric_value):
        self.models = models
        self.best = best
        self.metric_name = metric_name
        self.metric_value = metric_value


class CrossValidationOperation(Operation):
    """
    Cross validation operation used to evaluate classifier results using as many
    as folds provided in input.
    """
    NUM_FOLDS_PARAM = 'folds'
    EVALUATOR_PARAM = 'evaluator'
    SEED_PARAM = 'seed'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction_attribute'
    LABEL_ATTRIBUTE_PARAM = 'label_attribute'

    METRIC_TO_EVALUATOR = {
        'areaUnderROC': (
            'evaluation.BinaryClassificationEvaluator', 'rawPredictionCol'),
        'areaUnderPR': (
            'evaluation.BinaryClassificationEvaluator', 'rawPredictionCol'),
        'f1': ('evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'weightedPrecision': (
            'evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'weightedRecall': (
            'evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'accuracy': (
            'evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'rmse': ('evaluation.RegressionEvaluator', 'predictionCol'),
        'mse': ('evaluation.RegressionEvaluator', 'predictionCol'),
        'mae': ('evaluation.RegressionEvaluator', 'predictionCol'),
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(self.named_inputs) == 2])

        if self.EVALUATOR_PARAM not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.EVALUATOR_PARAM, self.__class__))

        self.metric = parameters.get(self.EVALUATOR_PARAM)

        if self.metric in self.METRIC_TO_EVALUATOR:
            self.evaluator = self.METRIC_TO_EVALUATOR[self.metric][0]
            self.param_prediction_arg = self.METRIC_TO_EVALUATOR[self.metric][1]
        else:
            raise ValueError(_('Invalid metric value {}').format(self.metric))

        self.prediction_attr = parameters.get(self.PREDICTION_ATTRIBUTE_PARAM,
                                              'prediction')

        if self.LABEL_ATTRIBUTE_PARAM not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.LABEL_ATTRIBUTE_PARAM, self.__class__))

        self.label_attr = parameters.get(self.LABEL_ATTRIBUTE_PARAM)

        self.num_folds = parameters.get(self.NUM_FOLDS_PARAM, 3)
        self.seed = parameters.get(self.SEED_PARAM)

        self.output = self.named_outputs.get(
            'scored data', 'scored_data_task_{}'.format(self.order))
        self.models = self.named_outputs.get(
            'models', 'models_task_{}'.format(self.order))
        self.best_model = self.named_outputs.get(
            'best model', 'best_model_{}'.format(self.order))

        self.algorithm_port = self.named_inputs.get(
            'algorithm', 'algo_{}'.format(self.order))
        self.input_port = self.named_inputs.get(
            'input data', 'in_{}'.format(self.order))
        self.evaluator_port = self.named_inputs.get(
            'evaluator', 'eval_{}'.format(self.order))

    @property
    def get_inputs_names(self):
        return ', '.join(
            [self.algorithm_port, self.input_port, self.evaluator_port])

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.best_model, self.models])

    def get_data_out_names(self, sep=','):
        return sep.join([self.output])

    def generate_code(self):
        code = dedent("""
                grid_builder = tuning.ParamGridBuilder()
                estimator, param_grid, metric = {algorithm}

                for param_name, values in param_grid.items():
                    param = getattr(estimator, param_name)
                    grid_builder.addGrid(param, values)

                evaluator = {evaluator}(
                    {prediction_arg}='{prediction_attr}',
                    labelCol=label_col,
                    metricName='{metric}')

                estimator.setLabelCol(label_col)
                estimator.setPredictionCol('{prediction_attr}')

                cross_validator = tuning.CrossValidator(
                    estimator=estimator,
                    estimatorParamMaps=grid_builder.build(),
                    evaluator=evaluator, numFolds={folds})
                cv_model = cross_validator.fit({input_data})
                fit_data = cv_model.transform({input_data})
                {best_model} = cv_model.bestModel
                metric_result = evaluator.evaluate(fit_data)
                {output} = fit_data
                {models} = None
                """.format(algorithm=self.algorithm_port,
                           input_data=self.input_port,
                           evaluator=self.evaluator,
                           output=self.output,
                           best_model=self.best_model,
                           models=self.models,
                           prediction_arg=self.param_prediction_arg,
                           prediction_attr=self.prediction_attr,
                           label_attr=self.label_attr[0],
                           folds=self.num_folds,
                           metric=self.metric))

        # If there is an output needing the evaluation result, it must be
        # processed here (summarization of data results)
        needs_evaluation = 'evaluation' in self.named_outputs and False
        if needs_evaluation:
            eval_code = """
                grouped_result = fit_data.select(
                        evaluator.getLabelCol(), evaluator.getPredictionCol())\\
                        .groupBy(evaluator.getLabelCol(),
                                 evaluator.getPredictionCol()).count().collect()
                eval_{output} = {{
                    'metric': {{
                        'name': evaluator.getMetricName(),
                        'value': metric_result
                    }},
                    'estimator': {{
                        'name': estimator.__class__.__name__,
                        'predictionCol': evaluator.getPredictionCol(),
                        'labelCol': evaluator.getLabelCol()
                    }},
                    'confusion_matrix': {{
                        'data': json.dumps(grouped_result)
                    }},
                    'evaluator': evaluator
                }}

                emit_event('task result', status='COMPLETED',
                    identifier='{task_id}', message='Result generated',
                    type='TEXT', title='{title}',
                    task={{'id': '{task_id}' }},
                    operation={{'id': {operation_id} }},
                    operation_id={operation_id},
                    content=json.dumps(eval_{output}))

                """.format(output=self.output,
                           title='Evaluation result',
                           task_id=self.parameters['task_id'],
                           operation_id=self.parameters['operation_id'])
        else:
            eval_code = ''
        code = '\n'.join([code, dedent(eval_code)])

        return code


class ClassificationModelOperation(Operation):
    FEATURES_ATTRIBUTE_PARAM = 'features'
    LABEL_ATTRIBUTE_PARAM = 'label'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction'
    WEIGHTS_PARAM = 'weights'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 2,
                             self.contains_results()])
        self.has_code = self.has_code and 'algorithm' in self.named_inputs

        if self.has_code:
            if not all(
                    [self.FEATURES_ATTRIBUTE_PARAM in parameters,
                     self.LABEL_ATTRIBUTE_PARAM in parameters]):
                msg = _("Parameters '{}' and '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_ATTRIBUTE_PARAM, self.LABEL_ATTRIBUTE_PARAM,
                    self.__class__.__name__))

            self.label = parameters.get(self.LABEL_ATTRIBUTE_PARAM)[0]
            self.features = parameters.get(self.FEATURES_ATTRIBUTE_PARAM)
            self.prediction = parameters.get(self.PREDICTION_ATTRIBUTE_PARAM,
                                             'prediction')
            self.ensemble_weights = parameters.get(self.WEIGHTS_PARAM,
                                                   '1') or '1'

            self.model = named_outputs.get('model',
                                           'model_task_{}'.format(self.order))
        if not self.has_code and len(self.named_outputs) > 0:
            raise ValueError(
                _('Model is being used, but at least one input is missing'))

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=','):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:
            task = self.parameters.get('task', {})
            display_text = task.get('forms', {}).get(
                'display_text', {'value': 1}).get('value', 1) in (1, '1')

            code = """
            emit = functools.partial(
                emit_event, name='update task',
                status='RUNNING', type='TEXT',
                identifier='{task_id}',
                operation={{'id': {operation_id}}}, operation_id={operation_id},
                task={{'id': '{task_id}'}},
                title='{title}')

            alg, param_grid, metrics = {algorithm}

            # Clone the algorithm because it can be used more than once
            # and this may cause concurrency problems
            params = dict([(p.name, v) for p, v in
                alg.extractParamMap().items()])

            # Perceptron does not support some parameters
            if isinstance(alg, MultilayerPerceptronClassifier):
                del params['rawPredictionCol']

            algorithm_cls = globals()[alg.__class__.__name__]
            algorithm = algorithm_cls()
            algorithm.setParams(**params)

            algorithm.setPredictionCol('{prediction}')

            requires_pipeline = False
            stages = []
            features = {feat}
            keep_at_end = [c.name for c in {train}.schema]
            keep_at_end.append('{prediction}')

            to_assemble = []
            if len(features) > 1 and not isinstance(
                {train}.schema[str(features[0])].dataType, VectorUDT):
                emit(message='{msg0}')
                for f in features:
                    if not dataframe_util.is_numeric({train}.schema, f):
                        name = f + '_tmp'
                        to_assemble.append(name)
                        stages.append(feature.StringIndexer(
                            inputCol=f, outputCol=name, handleInvalid='keep'))
                    else:
                        to_assemble.append(f)

                # Remove rows with null (VectorAssembler doesn't support it)
                cond = ' AND '.join(['{{}} IS NOT NULL '.format(c)
                    for c in to_assemble])
                stages.append(SQLTransformer(
                    statement='SELECT * FROM __THIS__ WHERE {{}}'.format(cond)))

                final_features = 'features_tmp'
                stages.append(feature.VectorAssembler(
                    inputCols=to_assemble, outputCol=final_features))
                requires_pipeline = True
            else:
                final_features = features[0]

            requires_revert_label = False
            if not dataframe_util.is_numeric({train}.schema, '{label}'):
                emit(message='{msg1}')
                final_label = '{label}_tmp'
                stages.append(feature.StringIndexer(
                            inputCol='{label}', outputCol=final_label,
                            handleInvalid='keep'))
                requires_pipeline = True
                requires_revert_label = True
            else:
                final_label = '{label}'

            if requires_pipeline:
                algorithm.setLabelCol(final_label)
                algorithm.setFeaturesCol(final_features)
                if requires_revert_label:
                    algorithm.setPredictionCol('{prediction}_tmp')

                stages.append(algorithm)

                pipeline = Pipeline(stages=stages)
                {model} = pipeline.fit({train})

                last_stages = [{model}]
                if requires_revert_label:
                    # Map indexed label to original value
                    last_stages.append(IndexToString(inputCol='{prediction}_tmp',
                        outputCol='{prediction}',
                        labels={model}.stages[-2].labels))

                # Remove temporary columns
                sql = 'SELECT {{}} FROM __THIS__'.format(', '.join(keep_at_end))
                last_stages.append(SQLTransformer(statement=sql))

                pipeline = Pipeline(stages=last_stages)
                {model} = pipeline.fit({train})
            else:
                algorithm.setLabelCol(final_label)
                algorithm.setFeaturesCol(final_features)
                {model} = algorithm.fit({train})

            # Used in ensembles, e.g. VotingClassifierOperation
            setattr({model}, 'ensemble_weights', {weights})

            # Lazy execution in case of sampling the data in UI
            def call_transform(df):
                return {model}.transform(df)
            {output} = dataframe_util.LazySparkTransformationDataframe(
                {model}, {train}, call_transform)
            display_text = {display_text}
            if display_text:
                rows = [[m, getattr({model}, m)] for m in metrics
                    if hasattr({model}, m)]
                headers = {headers}
                content = SimpleTableReport(
                    'table table-striped table-bordered table-sm',
                    headers, rows)

                result = '<h4>{title}</h4>'

                emit(status='COMPLETED', message=result + content.generate(),
                    type='HTML', title='{title}')

            """.format(
                model=self.model,
                algorithm=self.named_inputs.get('algorithm'),
                train=self.named_inputs['train input data'],
                label=self.label, feat=repr(self.features),
                prediction=self.prediction,
                output=self.output,
                msg0=_('Features are not assembled as a vector. They will be '
                       'implicitly assembled and rows with null values will be '
                       'discarded. If this is undesirable, explicitly add a '
                       'feature assembler in the workflow.'),
                msg1=_('Label attribute is categorical, it will be '
                       'implicitly indexed as string.'),
                display_text=display_text,
                title=_('Generated classification model parameters'),
                headers=[_('Parameter'), _('Value'), ],
                task_id=self.parameters['task_id'],
                operation_id=self.parameters['operation_id'],
                weights=repr(
                    [float(w) for w in self.ensemble_weights.split(',')]
                ))

            return dedent(code)


class ClassifierOperation(Operation):
    """
    Base class for classification algorithms
    """
    GRID_PARAM = 'paramgrid'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.name = "BaseClassifier()"
        self.output = self.named_outputs.get('algorithm',
                                             'algo_task_{}'.format(self.order))
        self.metrics = []
        self.summary_metrics = []

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return self.output

    def generate_code(self):
        if self.has_code:
            param_grid = {

            }
            declare = dedent("""
            param_grid = {param_grid}
            # Output result is the classifier and its parameters. Parameters are
            # need in classification model or cross validator.
            {out} = ({name},
                param_grid,
                {metrics})
            """).format(out=self.output, name=self.name,
                        param_grid=json.dumps(param_grid, indent=4),
                        metrics=json.dumps(self.metrics))

            code = [declare]
            return "\n".join(code)
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class SvmClassifierOperation(ClassifierOperation):
    MAX_ITER_PARAM = 'max_iter'
    STANDARDIZATION_PARAM = 'standardization'
    THRESHOLD_PARAM = 'threshold'
    WEIGHT_ATTR_PARAM = 'weight_attr'
    TOL_PARAM = 'tol'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.metrics = ['coefficients', 'intercept', 'numClasses',
                        'numFeatures']

        param_grid = parameters.get('paramgrid', {})
        ctor_params = {}
        params_name = [
            ['maxIter', self.MAX_ITER_PARAM, int],
            ['standardization', self.STANDARDIZATION_PARAM,
             lambda x: x in ('1', 1, 'true', True)],
            ['threshold', self.THRESHOLD_PARAM, float],
            ['tol', self.TOL_PARAM, float],
            ['weightAttr', self.WEIGHT_ATTR_PARAM, str],

        ]
        for spark_name, lemonade_name, f in params_name:
            if lemonade_name in param_grid and param_grid.get(lemonade_name):
                ctor_params[spark_name] = f(param_grid.get(lemonade_name))

        self.name = 'classification.LinearSVC(**{kwargs})'.format(
            kwargs=ctor_params)


class LogisticRegressionClassifierOperation(ClassifierOperation):
    WEIGHT_COL_PARAM = 'weight_col'
    FAMILY_PARAM = 'family'
    ALLOWED_FAMILY_VALUE = ['auto', 'binomial', 'multinomial']

    AGGREGATION_DEPTH_PARAM = 'aggregation_depth'
    ELASTIC_NET_PARAM_PARAM = 'elastic_net_param'
    FIT_INTERCEPT_PARAM = 'fit_intercept'
    MAX_ITER_PARAM = 'max_iter'
    REG_PARAM_PARAM = 'reg_param'
    TOL_PARAM = 'tol'
    THRESHOLD_PARAM = 'threshold'
    THRESHOLDS_PARAM = 'thresholds'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'classification.LogisticRegression'
        self.metrics = ['coefficientMatrix', 'coefficients', 'intercept',
                        'numClasses', 'numFeatures']

        param_grid = parameters.get('paramgrid', {})
        ctor_params = {}
        params_name = [
            ['weightCol', self.WEIGHT_COL_PARAM, str],
            ['family', self.FAMILY_PARAM, str],
            ['aggregationDepth', self.AGGREGATION_DEPTH_PARAM, int],
            ['elasticNetParam', self.ELASTIC_NET_PARAM_PARAM, float],
            ['fitIntercept', self.FIT_INTERCEPT_PARAM, bool],
            ['maxIter', self.MAX_ITER_PARAM, int],
            ['regParam', self.REG_PARAM_PARAM, float],
            ['tol', self.TOL_PARAM, float],
            ['threshold', self.THRESHOLD_PARAM, float],
            ['thresholds', self.THRESHOLDS_PARAM,
             lambda x: [float(y) for y in x.split(',')]],
        ]
        for spark_name, lemonade_name, f in params_name:
            if lemonade_name in param_grid and param_grid.get(lemonade_name):
                ctor_params[spark_name] = f(param_grid.get(lemonade_name))

        self.name = 'classification.LogisticRegression(**{kwargs})'.format(
            kwargs=ctor_params)


class DecisionTreeClassifierOperation(ClassifierOperation):
    CACHE_NODE_IDS_PARAM = 'cache_node_ids'
    MAX_BINS_PARAM = 'max_bins'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    MIN_INSTANCES_PER_NODE_PARAM = 'min_instances_per_node'
    IMPURITY_PARAM = 'impurity'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.metrics = ['depth', 'featureImportances', 'numClasses',
                        'numFeatures', 'numNodes', ]

        param_grid = parameters.get('paramgrid', {})
        ctor_params = {}
        params_name = [
            ['maxBins', self.MAX_BINS_PARAM, int],
            ['cacheNodeIds', self.CACHE_NODE_IDS_PARAM,
             lambda x: x in ('1', 1, 'true', True)],
            ['maxDepth', self.MAX_DEPTH_PARAM, int],
            ['minInfoGain', self.MIN_INFO_GAIN_PARAM, float],
            ['minInstancesPerNode', self.MIN_INSTANCES_PER_NODE_PARAM, int],
            ['impurity', self.IMPURITY_PARAM, str],

        ]
        for spark_name, lemonade_name, f in params_name:
            if lemonade_name in param_grid and param_grid.get(lemonade_name):
                ctor_params[spark_name] = f(param_grid.get(lemonade_name))

        self.name = 'classification.DecisionTreeClassifier(**{kwargs})'.format(
            kwargs=ctor_params)


class GBTClassifierOperation(ClassifierOperation):
    CACHE_NODE_IDS_PARAM = 'cache_node_ids'
    LOSS_TYPE_PARAM = 'loss_type'
    MAX_BINS_PARAM = 'max_bins'
    MAX_DEPTH_PARAM = 'max_depth'
    MAX_ITER_PARAM = 'max_iter'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    MIN_INSTANCES_PER_NODE_PARAM = 'min_instances_per_node'
    STEP_SIZE_PARAM = 'step_size'
    SUBSAMPLING_RATE_PARAM = 'subsampling_rate'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.metrics = ['featureImportances', 'numFeatures', 'totalNumNodes',
                        'treeWeights', 'trees']

        param_grid = parameters.get('paramgrid', {})
        ctor_params = {}
        params_name = [
            ['cacheNodeIds', self.CACHE_NODE_IDS_PARAM,
             lambda x: x in ('1', 1, 'true', True)],
            ['lossType', self.LOSS_TYPE_PARAM, str],
            ['maxBins', self.MAX_BINS_PARAM, int],
            ['maxDepth', self.MAX_DEPTH_PARAM, int],
            ['maxIter', self.MAX_ITER_PARAM, int],
            ['minInfoGain', self.MIN_INFO_GAIN_PARAM, float],
            ['minInstancesPerNode', self.MIN_INSTANCES_PER_NODE_PARAM, int],
            ['stepSize', self.STEP_SIZE_PARAM, float],
            ['subsamplingRate', self.SUBSAMPLING_RATE_PARAM, float],

        ]
        for spark_name, lemonade_name, f in params_name:
            if lemonade_name in param_grid and param_grid.get(lemonade_name):
                ctor_params[spark_name] = f(param_grid.get(lemonade_name))
        self.name = 'classification.GBTClassifier(**{kwargs})'.format(
            kwargs=ctor_params)


class NaiveBayesClassifierOperation(ClassifierOperation):
    SMOOTHING_PARAM = 'smoothing'
    MODEL_TYPE_PARAM = 'model_type'
    THRESHOLDS_PARAM = 'thresholds'
    WEIGHT_ATTR_PARAM = 'weight_attr'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.metrics = ['numClasses', 'numFeatures', 'pi', 'theta']

        param_grid = parameters.get('paramgrid', {})
        ctor_params = {}
        params_name = [
            ['smoothing', self.SMOOTHING_PARAM, float],
            ['modelType', self.MODEL_TYPE_PARAM, str],
            ['thresholds', self.THRESHOLDS_PARAM,
             lambda x: [float(y) for y in x.split(',')]],
            ['weightAttr', self.WEIGHT_ATTR_PARAM, str],
        ]
        for spark_name, lemonade_name, f in params_name:
            if lemonade_name in param_grid and param_grid.get(lemonade_name):
                ctor_params[spark_name] = f(param_grid.get(lemonade_name))

        self.name = 'classification.NaiveBayes(**{kwargs})'.format(
            kwargs=ctor_params)


class RandomForestClassifierOperation(ClassifierOperation):
    IMPURITY_PARAM = 'impurity'
    CACHE_NODE_IDS_PARAM = 'cache_node_ids'
    FEATURE_SUBSET_STRATEGY_PARAM = 'feature_subset_strategy'
    MAX_BINS_PARAM = 'max_bins'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    MIN_INSTANCES_PER_NODE_PARAM = 'min_instances_per_node'
    NUM_TREES_PARAM = 'num_trees'
    SUBSAMPLING_RATE_PARAM = 'subsampling_rate'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.metrics = ['featureImportances', 'getNumTrees',
                        'numClasses', 'numFeatures', 'trees']

        param_grid = parameters.get('paramgrid', {})
        ctor_params = {}
        params_name = [
            ['impurity', self.IMPURITY_PARAM, str],
            ['cacheNodeIds', self.CACHE_NODE_IDS_PARAM,
             lambda x: x in ('1', 1, 'true', True)],
            ['featureSubsetStrategy', self.FEATURE_SUBSET_STRATEGY_PARAM, str],
            ['maxBins', self.MAX_BINS_PARAM, int],
            ['maxDepth', self.MAX_DEPTH_PARAM, int],
            ['minInfoGain', self.MIN_INFO_GAIN_PARAM, float],
            ['minInstancesPerNode', self.MIN_INSTANCES_PER_NODE_PARAM, int],
            ['numTrees', self.NUM_TREES_PARAM, int],
            ['subsamplingRate', self.SUBSAMPLING_RATE_PARAM, float],
        ]
        for spark_name, lemonade_name, f in params_name:
            if lemonade_name in param_grid and param_grid.get(lemonade_name):
                ctor_params[spark_name] = f(param_grid.get(lemonade_name))

        self.name = 'classification.RandomForestClassifier(**{kwargs})'.format(
            kwargs=ctor_params)


class PerceptronClassifier(ClassifierOperation):
    BLOCK_SIZE_PARAM = 'block_size'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'
    SOLVER_PARAM = 'solver'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.metrics = ['layers', 'numFeatures', 'weights']
        param_grid = parameters.get('paramgrid', {})
        ctor_params = {}
        params_name = [
            ['blockSize', self.BLOCK_SIZE_PARAM, int],
            ['maxIter', self.MAX_ITER_PARAM, int],
            ['seed', self.SEED_PARAM, int],
            ['solver', self.SOLVER_PARAM, str]
        ]
        for spark_name, lemonade_name, f in params_name:
            if lemonade_name in param_grid and param_grid.get(lemonade_name):
                ctor_params[spark_name] = f(param_grid.get(lemonade_name))

        self.name = 'classification.MultilayerPerceptronClassifier(**{k})' \
            .format(k=ctor_params)


class OneVsRestClassifier(ClassifierOperation):
    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.metrics = ['classifier', 'featuresCol', 'labelCol',
                        'predictionCol']

        self.has_code = self.has_code and 'algorithm' in self.named_inputs

        ctor_params = {
        }
        self.name = 'classification.OneVsRest(classifier={c}[0], **{k})'.format(
            c=self.named_inputs.get('algorithm'), k=ctor_params)


class VotingClassifierOperation(Operation):
    PREDICTION_ATTRIBUTE_PARAM = 'prediction_attribute'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.models = named_inputs.get('models')
        self.prediction = parameters.get(
            self.PREDICTION_ATTRIBUTE_PARAM,
            'voting_prediction') or 'voting_prediction'
        self.has_code = self.has_code = all(
            [any(['output data' in self.named_outputs,
                  self.contains_results()]),
             len(self.models) > 0], )

    def generate_code(self):
        input_data = self.named_inputs.get('input data')
        code = dedent("""
            evaluators = [m.copy() for m in [{models}]]
            weights_count = len(evaluators)

            prediction_col = '{prediction}'
            conflict_names = ['rawPredictionCol', 'predictionCol',
                'probabilityCol']
            predictions = []
            weights = []
            rows = []
            for i, evaluator in enumerate(evaluators):
                weights.append(evaluator.ensemble_weights[0]
                    if hasattr(evaluator, 'ensemble_weights') else 1)
                # noinspection PyProtectedMember
                params = evaluator._paramMap
                rows.append([evaluator.__class__.__name__ , i + 1, weights[i]])
                for conflict in conflict_names:
                    k = [p for p in params.keys() if p.name == conflict]
                    if k:
                        params[k[0]] = '{{}}{{}}'.format(conflict, i)
                    if conflict == 'predictionCol':
                        predictions.append(params[k[0]])

            headers = ['{name}', '{order}', '{weight}']

            content = SimpleTableReport(
                    'table table-striped table-bordered table-sm',
                    headers, rows,
                    title='{title}')

            emit_event(
                'update task', status='COMPLETED',
                identifier='{task_id}',
                message=content.generate(),
                type='HTML', title='{title}',
                task={{'id': '{task_id}'}},
                operation={{'id': {operation_id}}},
                operation_id={operation_id})

            pipeline = Pipeline(stages=evaluators)
            model = pipeline.fit({input})

            def summarize(arr):
                acc= collections.defaultdict(int)
                for j, v in enumerate(arr):
                    acc[v] += weights[j]
                return max(acc.items(), key=lambda x: x[1])[0]

            {out} = model.transform({input}).withColumn('all_predictions',
                functions.array(*predictions)).withColumn(prediction_col,
                    functions.udf(summarize, types.DoubleType())(
                        'all_predictions'))

        """.format(models=', '.join(self.models), input=input_data,
                   out=self.output, prediction=self.prediction,
                   title=_('Models used in ensemble classification'),
                   task_id=self.parameters['task_id'],
                   operation_id=self.parameters['operation_id'],
                   name=_('name'), order=_('order'), weight=_('weight'),
                   ))
        return code


class ClassificationReport(ReportOperation):
    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ReportOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
            [len(self.named_inputs) > 1, self.contains_results()])
        self.multiple_inputs = True

    def get_data_out_names(self, sep=','):
        return ''

    def generate_code(self):
        code = dedent("{output} = 'ok'".format(output='FIXME'))
        return code


"""
Clustering part
"""


class ClusteringModelOperation(Operation):
    FEATURES_ATTRIBUTE_PARAM = 'features'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 2,
                             self.contains_results()])

        if self.FEATURES_ATTRIBUTE_PARAM not in parameters:
            msg = _("Parameter '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.FEATURES_ATTRIBUTE_PARAM, self.__class__))

        self.features = parameters.get(self.FEATURES_ATTRIBUTE_PARAM)

        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.model = self.named_outputs.get('model',
                                            'model_task_{}'.format(self.order))
        self.prediction = parameters.get(self.PREDICTION_ATTRIBUTE_PARAM,
                                         'prediction')

        self.centroids = self.named_outputs.get(
            'cluster centroids', 'centroids_task_{}'.format(self.order))

    @property
    def get_inputs_names(self):
        return ', '.join([
            self.named_inputs.get('train input data',
                                  'train_task_{}'.format(self.order)),
            self.named_inputs.get('algorithm',
                                  'algo_task_{}'.format(self.order))])

    def get_data_out_names(self, sep=','):
        return sep.join([self.output, self.centroids])

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model, self.centroids])

    def generate_code(self):

        if self.has_code:
            code = """
            emit = functools.partial(
                emit_event, name='update task',
                status='RUNNING', type='TEXT',
                identifier='{task_id}',
                operation={{'id': {operation_id}}}, operation_id={operation_id},
                task={{'id': '{task_id}'}},
                title='{title}')
            alg = {algorithm}

            # Clone the algorithm because it can be used more than once
            # and this may cause concurrency problems
            params = dict([(p.name, v) for p, v in
                alg.extractParamMap().items()])
            algorithm_cls = globals()[alg.__class__.__name__]
            algorithm = algorithm_cls()
            algorithm.setParams(**params)
            features = {features}
            requires_pipeline = False

            stages = [] # record pipeline stages
            if len(features) > 1 and not isinstance(
                {input}.schema[str(features[0])].dataType, VectorUDT):
                emit(message='{msg2}')
                for f in features:
                    if not dataframe_util.is_numeric({input}.schema, f):
                        raise ValueError('{msg1}')

                # Remove rows with null (VectorAssembler doesn't support it)
                cond = ' AND '.join(['{{}} IS NOT NULL '.format(c)
                    for c in features])
                stages.append(SQLTransformer(
                    statement='SELECT * FROM __THIS__ WHERE {{}}'.format(cond)))
                final_features = 'features_tmp'
                stages.append(feature.VectorAssembler(
                    inputCols=features, outputCol=final_features))
                requires_pipeline = True

            else:
                # If more than 1 vector is passed, use only the first
                final_features = features[0]

            algorithm.setFeaturesCol(final_features)

            if hasattr(algorithm, 'setPredictionCol'):
                algorithm.setPredictionCol('{prediction}')
            stages.append(algorithm)
            pipeline = Pipeline(stages=stages)

            pipeline_model = pipeline.fit({input})
            {model} = pipeline_model

            # There is no way to pass which attribute was used in clustering, so
            # information will be stored in a new attribute called features.
            setattr({model}, 'features', {features})

            # Lazy execution in case of sampling the data in UI
            def call_transform(df):
                if requires_pipeline:
                    return pipeline_model.transform(df).drop(final_features)
                else:
                    return pipeline_model.transform(df)
            {output} = dataframe_util.LazySparkTransformationDataframe(
                {model}, {input}, call_transform)

            summary = getattr({model}, 'summary', None)

            # Lazy execution in case of sampling the data in UI
            def call_clusters(df):
                if hasattr({model}, 'clusterCenters'):
                    return spark_session.createDataFrame(
                        [center.tolist()
                            for center in {model}.clusterCenters()])
                else:
                    return spark_session.createDataFrame([],
                        types.StructType([]))

            {centroids} = dataframe_util.LazySparkTransformationDataframe(
                {model}, {input}, call_clusters)

            if summary:
                summary_rows = []
                for p in dir(summary):
                    if not p.startswith('_') and p != "cluster":
                        try:
                            summary_rows.append(
                                [p, getattr(summary, p)])
                        except Exception as e:
                            summary_rows.append([p, e.message])
                summary_content = SimpleTableReport(
                    'table table-striped table-bordered', [],
                    summary_rows,
                    title='Summary')
                emit_event('update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=summary_content.generate(),
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}' }},
                    operation={{'id': {operation_id} }},
                    operation_id={operation_id})
            """.format(model=self.model,
                       algorithm=self.named_inputs['algorithm'],
                       input=self.named_inputs['train input data'],
                       output=self.output,
                       features=repr(self.features),
                       prediction=self.prediction,
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title=_("Clustering result"),
                       centroids=self.centroids,
                       msg1=_('Regression only support numerical features.'),
                       msg2=_('Features are not assembled as a vector. '
                              'They will be implicitly assembled and rows with '
                              'null values will be discarded. If this is '
                              'undesirable, explicitly add a feature assembler '
                              'in the workflow.'), )

            return dedent(code)


class ClusteringOperation(Operation):
    """
    Base class for clustering algorithms
    """

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.name = "BaseClustering"
        self.set_values = []
        self.output = self.named_outputs.get('algorithm',
                                             'algo_task_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def generate_code(self):
        if self.has_code:
            declare = "{0} = {1}()".format(self.output, self.name)
            code = [declare]
            code.extend(['{0}.set{1}({2})'.format(self.output, name, v)
                         for name, v in self.set_values])
            return "\n".join(code)


class LdaClusteringOperation(ClusteringOperation):
    NUMBER_OF_TOPICS_PARAM = 'number_of_topics'
    OPTIMIZER_PARAM = 'optimizer'
    MAX_ITERATIONS_PARAM = 'max_iterations'
    DOC_CONCENTRATION_PARAM = 'doc_concentration'
    TOPIC_CONCENTRATION_PARAM = 'topic_concentration'

    ONLINE_OPTIMIZER = 'online'
    EM_OPTIMIZER = 'em'

    def __init__(self, parameters, named_inputs, named_outputs):
        ClusteringOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.number_of_clusters = int(parameters.get(
            self.NUMBER_OF_TOPICS_PARAM, 10))
        self.optimizer = parameters.get(self.OPTIMIZER_PARAM,
                                        self.ONLINE_OPTIMIZER)
        if self.optimizer not in [self.ONLINE_OPTIMIZER, self.EM_OPTIMIZER]:
            raise ValueError(
                _('Invalid optimizer value {} for class {}').format(
                    self.optimizer, self.__class__))

        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)

        self.doc_concentration = parameters.get(self.DOC_CONCENTRATION_PARAM)
        # if not self.doc_concentration:
        #     self.doc_concentration = self.number_of_clusters * [-1.0]

        self.topic_concentration = parameters.get(
            self.TOPIC_CONCENTRATION_PARAM)

        self.set_values = [
            ['K', self.number_of_clusters],
            ['MaxIter', self.max_iterations],
            ['Optimizer', "'{}'".format(self.optimizer)],
        ]
        if self.doc_concentration:
            try:
                doc_concentration = [float(v) for v in
                                     str(self.doc_concentration).split(',') if
                                     v.strip()]
                self.set_values.append(['DocConcentration', doc_concentration])
            except Exception:
                raise ValueError(
                    _('Invalid document concentration: {}. It must be a single '
                      'decimal value or a list of decimal numbers separated by '
                      'comma.').format(
                        self.doc_concentration))

        if self.topic_concentration:
            self.set_values.append(
                ['TopicConcentration', self.topic_concentration])
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.name = "clustering.LDA"


class KMeansClusteringOperation(ClusteringOperation):
    K_PARAM = 'number_of_clusters'
    MAX_ITERATIONS_PARAM = 'max_iterations'
    TYPE_PARAMETER = 'type'
    INIT_MODE_PARAMETER = 'init_mode'
    TOLERANCE_PARAMETER = 'tolerance'

    TYPE_TRADITIONAL = 'kmeans'
    TYPE_BISECTING = 'bisecting'

    INIT_MODE_KMEANS_PARALLEL = 'k-means||'
    INIT_MODE_RANDOM = 'random'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClusteringOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.number_of_clusters = parameters.get(self.K_PARAM,
                                                 10)

        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)
        self.type = parameters.get(self.TYPE_PARAMETER)
        self.tolerance = float(parameters.get(self.TOLERANCE_PARAMETER, 0.001))

        if self.type == self.TYPE_BISECTING:
            self.name = "BisectingKMeans"
            self.set_values = [
                ['MaxIter', self.max_iterations],
                ['K', self.number_of_clusters],
            ]
        elif self.type == self.TYPE_TRADITIONAL:
            if parameters.get(
                    self.INIT_MODE_PARAMETER) == self.INIT_MODE_RANDOM:
                self.init_mode = self.INIT_MODE_RANDOM
            else:
                self.init_mode = self.INIT_MODE_KMEANS_PARALLEL
            self.set_values.append(['InitMode', '"{}"'.format(self.init_mode)])
            self.name = "clustering.KMeans"
            self.set_values = [
                ['MaxIter', self.max_iterations],
                ['K', self.number_of_clusters],
                ['Tol', self.tolerance],
                ['InitMode', '"{}"'.format(self.init_mode)]
            ]
        else:
            raise ValueError(
                _('Invalid type {} for class {}').format(
                    self.type, self.__class__))

        self.has_code = any([len(named_outputs) > 0, self.contains_results()])


class GaussianMixtureClusteringOperation(ClusteringOperation):
    K_PARAM = 'number_of_clusters'
    MAX_ITERATIONS_PARAM = 'max_iterations'
    TOLERANCE_PARAMETER = 'tolerance'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClusteringOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.number_of_clusters = parameters.get(self.K_PARAM, 10)
        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)
        self.tolerance = float(parameters.get(self.TOLERANCE_PARAMETER, 0.001))

        self.set_values = [
            ['MaxIter', self.max_iterations],
            ['K', self.number_of_clusters],
            ['Tol', self.tolerance],
        ]
        self.name = "clustering.GaussianMixture"
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])


class TopicReportOperation(ReportOperation):
    """
    Produces a report for topic identification in text
    """
    TERMS_PER_TOPIC_PARAM = 'terms_per_topic'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ReportOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.terms_per_topic = parameters.get(self.TERMS_PER_TOPIC_PARAM, 10)

        self.has_code = any(
            [len(self.named_inputs) == 3, self.contains_results()])
        self.output = self.named_outputs.get('topics',
                                             'topics_{}'.format(self.order))

        self.vocabulary_input = self.named_inputs.get(
            'vocabulary')

    def get_output_names(self, sep=", "):
        return self.output

    def get_data_out_names(self, sep=','):
        return self.output

    def generate_code(self):
        code = dedent("""
            # TODO: evaluate if using broadcast() is more efficient
            terms_idx_to_str = functions.udf(lambda term_indexes:
                [{vocabulary}['text_vector'][inx]  for inx in term_indexes])
            topic_df = {model}.describeTopics(
                maxTermsPerTopic={tpt}).withColumn(
                    'terms', terms_idx_to_str(functions.col('termIndices')))

            {output} = topic_df

            # See hack in ClusteringModelOperation
            features = {model}.features

            '''
            for row in topic_df.collect():
                topic_number = row[0]
                topic_terms  = row[1]
                print "Topic: ", topic_number
                print '========================='
                print '\\t',
                for inx in topic_terms[:{tpt}]:
                    print {vocabulary}[features][inx],
                print
            '''
        """.format(model=self.named_inputs['model'],
                   tpt=self.terms_per_topic,
                   vocabulary=self.vocabulary_input,
                   output=self.output,
                   input=self.named_inputs['input data']))
        return code


"""
  Collaborative Filtering part
"""


class RecommendationModel(Operation):
    # RANK_PARAM = 'rank'
    # MAX_ITER_PARAM = 'max_iter'
    # USER_COL_PARAM = 'user_col'
    # ITEM_COL_PARAM = 'item_col'
    # RATING_COL_PARAM = 'rating_col'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 2,
                             self.contains_results()])

        # if not all([self.RANK_PARAM in parameters,
        #             self.RATING_COL_PARAM in parameters]):
        #     msg = _("Parameters '{}' and '{}' must be informed for task {}")
        #     raise ValueError(msg.format(
        #         self.RANK_PARAM, self.RATING_COL_PARAM,
        #         self.__class__.__name__))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))
        self.output = self.named_outputs.get(
            'output data', 'data_{}'.format(self.order))
        # self.ratingCol = parameters.get(self.RATING_COL_PARAM)

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        code = """
        {0} = {1}.fit({2})
        {output_data} = {0}.transform({2})
        """.format(self.model, self.named_inputs['algorithm'],
                   self.named_inputs['input data'],
                   output_data=self.output)

        return dedent(code)


class CollaborativeOperation(Operation):
    """
    Base class for Collaborative Filtering algorithm
    """

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.name = "als"
        self.set_values = []
        # Define outputs and model
        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.model = self.named_outputs.get('model', 'model_task_{}'.format(
            self.order))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.model])

    def generate_code(self):
        declare = "{0} = {1}()".format(self.output, self.name)
        code = [declare]
        code.extend(['{0}.set{1}({2})'.format(self.output, name, v)
                     for name, v in self.set_values])
        return "\n".join(code)


class AlternatingLeastSquaresOperation(Operation):
    """
        Alternating Least Squares (ALS) matrix factorization.

        The spark algorithm used is based on
        `"Collaborative Filtering for Implicit Feedback Datasets",
        <http://dx.doi.org/10.1109/ICDM.2008.22>`_
    """
    RANK_PARAM = 'rank'
    MAX_ITER_PARAM = 'max_iter'
    USER_COL_PARAM = 'user_col'
    ITEM_COL_PARAM = 'item_col'
    RATING_COL_PARAM = 'rating_col'
    REG_PARAM = 'reg_param'

    IMPLICIT_PREFS_PARAM = 'implicitPrefs'
    ALPHA_PARAM = 'alpha'
    SEED_PARAM = 'seed'
    NUM_USER_BLOCKS_PARAM = 'numUserBlocks'
    NUM_ITEM_BLOCKS_PARAM = 'numItemBlocks'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.rank = parameters.get(self.RANK_PARAM, 10)
        self.maxIter = parameters.get(self.MAX_ITER_PARAM, 10)
        self.userCol = parameters.get(self.USER_COL_PARAM, 'user_id')[0]
        self.itemCol = parameters.get(self.ITEM_COL_PARAM, 'movie_id')[0]
        self.ratingCol = parameters.get(self.RATING_COL_PARAM, 'rating')[0]

        self.regParam = parameters.get(self.REG_PARAM, 0.1)
        self.implicitPrefs = parameters.get(self.IMPLICIT_PREFS_PARAM, False)

        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.name = "collaborativefiltering.ALS"

        # Define input and output
        # output = self.named_outputs['output data']
        # self.input = self.named_inputs['train input data']

    def generate_code(self):
        code = dedent("""
                # Build the recommendation model using ALS on the training data
                # Strategy for dealing with unknown or new users/items at
                # prediction time is drop. See SPARK-14489 and SPARK-19345.
                {algorithm} = ALS(maxIter={maxIter}, regParam={regParam},
                        userCol='{userCol}', itemCol='{itemCol}',
                        ratingCol='{ratingCol}',
                        coldStartStrategy='drop')
                """.format(
            algorithm=self.named_outputs['algorithm'],
            maxIter=self.maxIter,
            regParam=float(self.regParam),
            userCol='{user}'.format(user=self.userCol),
            itemCol='{item}'.format(item=self.itemCol),
            ratingCol='{rating}'.format(rating=self.ratingCol))
        )

        return code


'''
    Logistic Regression Classification
'''


class LogisticRegressionModel(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    WEIGHT_COL_PARAM = ''
    MAX_ITER_PARAM = 'max_iter'
    FAMILY_PARAM = 'family'
    PREDICTION_COL_PARAM = 'prediction'

    REG_PARAM = 'reg_param'
    ELASTIC_NET_PARAM = 'elastic_net'

    # Have summaries model with measure results
    TYPE_BINOMIAL = 'binomial'
    # Multinomial family doesn't have summaries model
    TYPE_MULTINOMIAL = 'multinomial'

    TYPE_AUTO = 'auto'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.inputs = None
        self.has_code = any([len(named_outputs) > 0 and len(self.inputs) == 2,
                             self.contains_results()])

        if not all([self.FEATURES_PARAM in parameters['workflow_json'],
                    self.LABEL_PARAM in parameters['workflow_json']]):
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.FEATURES_PARAM, self.LABEL_PARAM,
                self.__class__.__name__))

        self.model = self.named_outputs.get('model')
        self.output = self.named_outputs.get('output data')

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:

            code = """
            {0} = {1}.fit({2})
            {output_data} = {0}.transform({2})
            """.format(self.model, self.named_inputs['algorithm'],
                       self.named_inputs['input data'], output_data=self.output)

            return dedent(code)
        else:
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format('[]inputs',
                                        '[]outputs',
                                        self.__class__))


'''
    Regression Algorithms
'''


class RegressionModelOperation(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_COL_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 2,
                             self.contains_results()])

        if self.has_code:
            self.algorithm = self.named_inputs['algorithm']
            self.input = self.named_inputs['train input data']

            if not all([self.FEATURES_PARAM in parameters,
                        self.LABEL_PARAM in parameters]):
                msg = _("Parameters '{}' and '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM, self.LABEL_PARAM,
                    self.__class__.__name__))

            self.features = parameters[self.FEATURES_PARAM]
            self.label = parameters[self.LABEL_PARAM]
            self.prediction = parameters.get(self.PREDICTION_COL_PARAM,
                                             'prediction') or 'prediction'
            self.model = self.named_outputs.get(
                'model', 'model_{}'.format(self.order))
            self.output = self.named_outputs.get(
                'output data', 'out_task_{}'.format(self.order))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:
            code = """
            emit = functools.partial(
                emit_event, name='update task',
                status='RUNNING', type='TEXT',
                identifier='{task_id}',
                operation={{'id': {operation_id}}}, operation_id={operation_id},
                task={{'id': '{task_id}'}},
                title='{title}')
            alg = {algorithm}

            # Clone the algorithm because it can be used more than once
            # and this may cause concurrency problems
            params = dict([(p.name, v) for p, v in
                alg.extractParamMap().items()])

            algorithm_cls = globals()[alg.__class__.__name__]
            algorithm = algorithm_cls()
            algorithm.setParams(**params)
            algorithm.setPredictionCol('{prediction}')
            algorithm.setLabelCol('{label}')

            features = {features}
            requires_pipeline = False

            stages = [] # record pipeline stages
            if len(features) > 1 and not isinstance(
                {input}.schema[str(features[0])].dataType, VectorUDT):
                emit(message='{msg2}')
                for f in features:
                    if not dataframe_util.is_numeric({input}.schema, f):
                        raise ValueError('{msg1}')

                # Remove rows with null (VectorAssembler doesn't support it)
                cond = ' AND '.join(['{{}} IS NOT NULL '.format(c)
                    for c in features])
                stages.append(SQLTransformer(
                    statement='SELECT * FROM __THIS__ WHERE {{}}'.format(cond)))
                final_features = 'features_tmp'
                stages.append(feature.VectorAssembler(
                    inputCols=features, outputCol=final_features))
                requires_pipeline = True

            else:
                # If more than 1 vector is passed, use only the first
                final_features = features[0]

            algorithm.setFeaturesCol(final_features)
            stages.append(algorithm)
            pipeline = Pipeline(stages=stages)

            try:

                pipeline_model = pipeline.fit({input})
                def call_transform(df):
                    if requires_pipeline:
                        return pipeline_model.transform(df).drop(final_features)
                    else:
                        return pipeline_model.transform(df)
                {output_data} = dataframe_util.LazySparkTransformationDataframe(
                    pipeline_model, {input}, call_transform)

                {model} = pipeline_model

                display_text = {display_text}
                if display_text:
                    headers = []
                    row = []
                    metrics = ['coefficients', 'intercept', 'scale', ]

                    for metric in metrics:
                        value = getattr({model}, metric, None)
                        if value:
                            headers.append(metric)
                            row.append(str(value))

                    if row:
                        content = SimpleTableReport(
                            'table table-striped table-bordered',
                            headers, [row])
                        emit_event('update task', status='COMPLETED',
                            identifier='{task_id}',
                            message=content.generate(),
                            type='HTML', title='{title}',
                            task={{'id': '{task_id}' }},
                            operation={{'id': {operation_id} }},
                            operation_id={operation_id})

                    summary = getattr({model}, 'summary', None)
                    if summary:
                        summary_rows = []
                        for p in dir(summary):
                            if not p.startswith('_'):
                                try:
                                    summary_rows.append(
                                        [p, getattr(summary, p)])
                                except Exception as e:
                                    summary_rows.append([p, e.message])
                        summary_content = SimpleTableReport(
                            'table table-striped table-bordered', [],
                            summary_rows,
                            title='Summary')
                        emit_event('update task', status='COMPLETED',
                            identifier='{task_id}',
                            message=summary_content.generate(),
                            type='HTML', title='{title}',
                            task={{'id': '{task_id}' }},
                            operation={{'id': {operation_id} }},
                            operation_id={operation_id})

            except IllegalArgumentException as iae:
                if 'org.apache.spark.ml.linalg.Vector' in iae:
                    raise ValueError('{msg0}')
                else:
                    raise
            """.format(model=self.model, algorithm=self.algorithm,
                       input=self.named_inputs['train input data'],
                       output_data=self.output, prediction=self.prediction,
                       label=self.label[0], features=repr(self.features),
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title="Regression result",
                       msg0=_('Assemble features in a vector before using a '
                              'regression model'),
                       msg1=_('Regression only support numerical features.'),
                       msg2=_('Features are not assembled as a vector. '
                              'They will be implicitly assembled and rows with '
                              'null values will be discarded. If this is '
                              'undesirable, explicitly add a feature assembler '
                              'in the workflow.'),
                       display_text=self.parameters['task']['forms'].get(
                           'display_text', {}).get('value') in (1, '1'))

            return dedent(code)


# noinspection PyAbstractClass
class RegressionOperation(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_ATTR_PARAM = 'prediction'

    __slots__ = ('label', 'features', 'prediction')

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.label = self.features = self.prediction = None
        self.output = named_outputs.get(
            'algorithm', 'regression_algorithm_{}'.format(self.order))

    def read_common_params(self, parameters):
        if not all([self.LABEL_PARAM in parameters,
                    self.FEATURES_PARAM in parameters]):
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.FEATURES_PARAM, self.LABEL_PARAM,
                self.__class__))
        else:
            self.label = parameters.get(self.LABEL_PARAM)[0]
            self.features = parameters.get(self.FEATURES_PARAM)[0]
            self.prediction = parameters.get(self.PREDICTION_ATTR_PARAM)[0]
            self.output = self.named_outputs.get(
                'algorithm', 'regression_algorithm_{}'.format(self.order))

    def get_output_names(self, sep=', '):
        return self.output

    def get_data_out_names(self, sep=','):
        return ''


class LinearRegressionOperation(RegressionOperation):
    MAX_ITER_PARAM = 'max_iter'
    WEIGHT_COL_PARAM = 'weight'
    REG_PARAM = 'reg_param'
    ELASTIC_NET_PARAM = 'elastic_net'

    SOLVER_PARAM = 'solver'

    TYPE_SOLVER_AUTO = 'auto'
    TYPE_SOLVER_NORMAL = 'normal'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.LinearRegression'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            # self.read_common_params(parameters)

            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10) or 10
            self.reg_param = parameters.get(self.REG_PARAM, 0.1) or 0.0
            self.weight_col = parameters.get(self.WEIGHT_COL_PARAM, None)

            self.solver = self.parameters.get(self.SOLVER_PARAM,
                                              self.TYPE_SOLVER_AUTO)
            self.elastic = self.parameters.get(self.ELASTIC_NET_PARAM,
                                               0.0) or 0.0

    def generate_code(self):
        if self.has_code:
            code = dedent("""
            {output} = LinearRegression(
                maxIter={max_iter}, regParam={reg_param},
                solver='{solver}',
                elasticNetParam={elastic})""").format(
                output=self.output,
                max_iter=self.max_iter,
                reg_param=self.reg_param,
                solver=self.solver,
                elastic=self.elastic,
            )
            return code
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class GeneralizedLinearRegressionOperation(RegressionOperation):
    FAMILY_PARAM = 'family'
    LINK_PARAM = 'link'
    MAX_ITER_PARAM = 'max_iter'
    REG_PARAM = 'reg_param'
    WEIGHT_COL_PARAM = 'weight'
    SOLVER_PARAM = 'solver'
    LINK_PREDICTION_COL_PARAM = 'link_prediction'

    TYPE_SOLVER_IRLS = 'irls'
    TYPE_SOLVER_NORMAL = 'normal'

    TYPE_FAMILY_GAUSSIAN = 'gaussian'
    TYPE_FAMILY_BINOMIAL = 'binomial'
    TYPE_FAMILY_POISSON = 'poisson'
    TYPE_FAMILY_GAMMA = 'gamma'

    TYPE_LINK_IDENTITY = 'identity'  # gaussian-poisson-gamma
    TYPE_LINK_LOG = 'log'  # gaussian-poisson-gama
    TYPE_LINK_INVERSE = 'inverse'  # gaussian-gamma
    TYPE_LINK_LOGIT = 'logit'  # binomial-
    TYPE_LINK_PROBIT = 'probit'  # binomial
    TYPE_LINK_CLOGLOG = 'cloglog'  # binomial
    TYPE_LINK_SQRT = 'sqrt'  # poisson

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.GeneralizedLinearRegression'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10)
            self.reg_param = parameters.get(self.REG_PARAM, 0.1)
            self.weight_col = parameters.get(self.WEIGHT_COL_PARAM, None)

            self.family = self.parameters.get(self.FAMILY_PARAM,
                                              self.TYPE_FAMILY_BINOMIAL)
            self.link = self.parameters.get(self.LINK_PARAM)

            if self.link is not None:
                self.link = "'{}'".format(self.link)

            # @FIXME: Need to understand the purpose of this parameter
            self.link_prediction_col = self.parameters.get(
                self.LINK_PREDICTION_COL_PARAM, [None])[0]

            if self.link_prediction_col is not None:
                self.link_prediction_col = "'{}'".format(
                    self.link_prediction_col)

    def generate_code(self):
        if self.has_code:
            declare = dedent("""
            {output} = GeneralizedLinearRegression(
                maxIter={max_iter},
                regParam={reg_param},
                family='{type_family}',
                link={link})
            """).format(output=self.output,
                        features=self.features,
                        label=self.label,
                        max_iter=self.max_iter,
                        reg_param=self.reg_param,
                        type_family=self.family,
                        link=self.link,
                        link_col=self.link_prediction_col
                        )
            # add , weightCol={weight} if exist
            code = [declare]
            return "\n".join(code)
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class DecisionTreeRegressionOperation(RegressionOperation):
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_INSTANCE_PER_NODE_PARAM = 'min_instance'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    SEED_PARAM = 'seed'

    VARIANCE_COL_PARAM = 'variance_col'
    IMPURITY_PARAM = 'impurity'

    TYPE_IMPURITY_VARIANCE = 'variance'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.DecisionTreeRegressor'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.read_common_params(parameters)

            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 5)
            self.min_instance = parameters.get(self.MIN_INSTANCE_PER_NODE_PARAM,
                                               1)
            self.min_info_gain = parameters.get(self.MIN_INFO_GAIN_PARAM, 0.0)

            self.variance_col = self.parameters.get(self.VARIANCE_COL_PARAM,
                                                    None)
            self.seed = self.parameters.get(self.SEED_PARAM, None)

            self.impurity = self.parameters.get(self.IMPURITY_PARAM, 'variance')

    def generate_code(self):
        if self.has_code:
            declare = dedent("""
            {output} = DecisionTreeRegressor(featuresCol='{features}',
                                             labelCol='{label}',
                                             maxDepth={max_iter},
                                             minInstancesPerNode={min_instance},
                                             minInfoGain={min_info},
                                             impurity='{impurity}',
                                             seed={seed},
                                             varianceCol={variance_col}
                                             )
            """).format(output=self.output,
                        features=self.features,
                        label=self.label,
                        max_depth=self.max_depth,
                        min_instance=self.min_instance,
                        min_info=self.min_info_gain,
                        impurity=self.impurity,
                        seed=self.seed,
                        variance_col=self.variance_col
                        )
            # add , weightCol={weight} if exist
            code = [declare]
            return "\n".join(code)
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class GBTRegressorOperation(RegressionOperation):
    MAX_ITER_PARAM = 'max_iter'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_INSTANCE_PER_NODE_PARAM = 'min_instance'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    SEED_PARAM = 'seed'

    IMPURITY_PARAM = 'impurity'

    TYPE_IMPURITY_VARIANCE = 'variance'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.GBTRegressor'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10) or 10
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 5) or 5
            self.min_instance = parameters.get(
                self.MIN_INSTANCE_PER_NODE_PARAM, 1) or 1
            self.min_info_gain = parameters.get(
                self.MIN_INFO_GAIN_PARAM, 0.0) or 0.0

            self.seed = self.parameters.get(self.SEED_PARAM)

            self.impurity = self.parameters.get(self.IMPURITY_PARAM, 'variance')

    def generate_code(self):
        if self.has_code:
            declare = dedent("""
            {output} = GBTRegressor(maxDepth={max_depth},
                                   minInstancesPerNode={min_instance},
                                   minInfoGain={min_info},
                                   seed={seed},
                                   maxIter={max_iter})
            # Only variance is valid, there is a bug, does not work in
            # constructor
            {output}.setImpurity('{impurity}')
            """).format(output=self.output,
                        features=self.features,
                        label=self.label,
                        max_depth=self.max_depth,
                        min_instance=self.min_instance,
                        min_info=self.min_info_gain,
                        impurity=self.impurity,
                        seed=self.seed if self.seed else None,
                        max_iter=self.max_iter)
            # add , weightCol={weight} if exist
            code = [declare]
            return "\n".join(code)
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class AFTSurvivalRegressionOperation(RegressionOperation):
    MAX_ITER_PARAM = 'max_iter'
    AGR_DETPTH_PARAM = 'aggregation_depth'
    CENSOR_COL_PARAM = 'censor'
    QUANTILES_PROBABILITIES_PARAM = 'quantile_probabilities'
    QUANTILES_COL_PARAM = 'quantiles_col'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.AFTSurvivalRegression'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 100)
            self.agg_depth = parameters.get(self.AGR_DETPTH_PARAM, 2)

            self.censor = self.parameters.get(
                self.CENSOR_COL_PARAM, ['censor'])[0]
            self.quantile_prob = self.parameters.get(
                self.QUANTILES_PROBABILITIES_PARAM,
                '0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99')
            if self.quantile_prob[0] != '[':
                self.quantile_prob = '[{}]'.format(self.quantile_prob)

            self.quantile_col = self.parameters.get(self.QUANTILES_COL_PARAM,
                                                    'None')

    def generate_code(self):
        if self.has_code:
            declare = dedent("""
            {output} = AFTSurvivalRegression(
                 maxIter={max_iter},
                 censorCol='{censor}',
                 quantileProbabilities={quantile_prob},
                 quantilesCol='{quantile_col}',
                 aggregationDepth={agg_depth}
                 )
            """.format(output=self.output,
                       max_iter=self.max_iter,
                       censor=self.censor,
                       quantile_prob=self.quantile_prob,
                       quantile_col=self.quantile_col,
                       agg_depth=self.agg_depth, ))
            # add , weightCol={weight} if exist
            code = [declare]
            return "\n".join(code)
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class RandomForestRegressorOperation(RegressionOperation):
    MAX_DEPTH_PARAM = 'max_depth'
    MAX_BINS_PARAM = 'max_bins'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    NUM_TREES_PARAM = 'num_trees'
    FEATURE_SUBSET_STRATEGY_PARAM = 'feature_subset_strategy'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.RandomForestRegressor'
        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 5) or 5
            self.max_bins = parameters.get(self.MAX_BINS_PARAM, 32) or 32
            self.min_info_gain = parameters.get(self.MIN_INFO_GAIN_PARAM,
                                                0.0) or 0.0
            self.num_trees = parameters.get(self.NUM_TREES_PARAM, 20) or 20
            self.feature_subset_strategy = parameters.get(
                self.FEATURE_SUBSET_STRATEGY_PARAM, 'auto')

    def generate_code(self):
        code = dedent("""
            {output} = RandomForestRegressor(
                maxDepth={max_depth},
                maxBins={max_bins},
                minInfoGain={min_info_gain},
                impurity="variance",
                numTrees={num_trees},
                featureSubsetStrategy='{feature_subset_strategy}')
            """).format(output=self.output,
                        max_depth=self.max_depth,
                        max_bins=self.max_bins,
                        min_info_gain=self.min_info_gain,
                        num_trees=self.num_trees,
                        feature_subset_strategy=self.feature_subset_strategy)
        return code


class IsotonicRegressionOperation(RegressionOperation):
    """
        Only univariate (single feature) algorithm supported
    """
    WEIGHT_COL_PARAM = 'weight'
    ISOTONIC_PARAM = 'isotonic'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.IsotonicRegression'
        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.weight_col = parameters.get(self.WEIGHT_COL_PARAM, None)
            self.isotonic = parameters.get(
                self.ISOTONIC_PARAM, True) in (1, '1', 'true', True)

    def generate_code(self):
        code = dedent("""
        {output} = IsotonicRegression(isotonic={isotonic})
        """).format(output=self.output,
                    isotonic=self.isotonic, )
        return code


class SaveModelOperation(Operation):
    NAME_PARAM = 'name'
    PATH_PARAM = 'path'
    STORAGE_PARAM = 'storage'
    SAVE_CRITERIA_PARAM = 'save_criteria'
    WRITE_MODE_PARAM = 'write_mode'

    CRITERIA_BEST = 'BEST'
    CRITERIA_ALL = 'ALL'
    CRITERIA_OPTIONS = [CRITERIA_BEST, CRITERIA_ALL]

    WRITE_MODE_ERROR = 'ERROR'
    WRITE_MODE_OVERWRITE = 'OVERWRITE'
    WRITE_MODE_OPTIONS = [WRITE_MODE_ERROR,
                          WRITE_MODE_OVERWRITE]

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.parameters = parameters

        self.name = parameters.get(self.NAME_PARAM)
        self.storage_id = parameters.get(self.STORAGE_PARAM)

        if self.name is None or self.storage_id is None:
            msg = _('Missing parameters. Check if values for parameters {} '
                    'were informed')
            raise ValueError(
                msg.format(', '.join([self.NAME_PARAM, self.STORAGE_PARAM])))

        self.path = parameters.get(self.PATH_PARAM, '/limonero/models').rstrip(
            '/')

        self.write_mode = parameters.get(self.WRITE_MODE_PARAM,
                                         self.WRITE_MODE_ERROR)
        if self.write_mode not in self.WRITE_MODE_OPTIONS:
            raise ValueError(
                _('Invalid value for parameter {param}: {value}').format(
                    param=self.WRITE_MODE_PARAM, value=self.write_mode))

        self.criteria = parameters.get(self.SAVE_CRITERIA_PARAM,
                                       self.CRITERIA_ALL)

        if self.criteria not in self.CRITERIA_OPTIONS:
            raise ValueError(
                _('Invalid value for parameter {param}: {value}').format(
                    param=self.SAVE_CRITERIA_PARAM, value=self.criteria))

        self.has_code = any([len(named_inputs) > 0, self.contains_results()])

    def generate_code(self):
        limonero_config = self.parameters.get('configuration') \
            .get('juicer').get('services').get('limonero')

        url = limonero_config['url']
        token = str(limonero_config['auth_token'])
        storage = limonero_service.get_storage_info(url, token, self.storage_id)

        models = self.named_inputs['models']
        if not isinstance(models, list):
            models = [models]

        if self.write_mode == self.WRITE_MODE_OVERWRITE:
            write_mode = 'overwrite().'
        else:
            write_mode = ''

        user = self.parameters.get('user', {})
        code = dedent("""
            from juicer.spark.ml_operation import ModelsEvaluationResultList
            from juicer.service.limonero_service import register_model

            all_models = [{models}]
            with_evaluation = [m for m in all_models if isinstance(
                m, ModelsEvaluationResultList)]
            criteria = '{criteria}'

            if criteria != 'ALL' and len(with_evaluation) != len(all_models):
                raise ValueError('{msg0}')
            if criteria == 'ALL':
                models_to_save = list(itertools.chain.from_iterable(
                    map(lambda m: m.models if isinstance(m,
                        ModelsEvaluationResultList) else [m], all_models)))
            elif criteria == 'BEST':
                metrics_used = set([m.metric_name for m in all_models])
                if len(metrics_used) > 1:
                    msg = '{msg1}'
                    raise ValueError(msg.format(', '.join(metrics_used)))

                models_to_save = [m.best for m in all_models]

            def _save_model(model_to_save, model_path, model_name):
                final_model_path = '{final_url}/{{}}'.format(model_path)
                model_to_save.write().{overwrite}save(final_model_path)
                # Save model information in Limonero
                model_type = '{{}}.{{}}'.format(model_to_save.__module__,
                    model_to_save.__class__.__name__)

                model_payload = {{
                    "user_id": {user_id},
                    "user_name": '{user_name}',
                    "user_login": '{user_login}',
                    "name": model_name,
                    "class_name": model_type,
                    "storage_id": {storage_id},
                    "path":  model_path,
                    "type": "UNSPECIFIED"
                }}
                # Save model information in Limonero
                register_model('{url}', model_payload, '{token}')

            for i, model in enumerate(models_to_save):
                if isinstance(model, dict): # For instance, it's a Indexer
                    for k, v in model.items():
                        name = '{name} - {{}}'.format(k)
                        path = '{path}/{name}.{{0}}.{{1:04d}}'.format(k, i)
                        _save_model(v, path, name)
                else:
                    name = '{name}'
                    path = '{path}/{name}.{{0:04d}}'.format(i)
                    _save_model(model, path, name)
        """.format(models=', '.join(models), overwrite=write_mode,
                   path=self.path,
                   final_url=storage['url'],
                   url=url,
                   token=token,
                   storage_id=self.storage_id,
                   name=self.name.replace(' ', '_'),
                   criteria=self.criteria,
                   msg0=_('You cannot mix models with and without '
                          'evaluation (e.g. indexers) when saving models '
                          'and criteria is different from ALL'),
                   msg1=_('You cannot mix models built using with '
                          'different metrics ({}).'),
                   user_id=user.get('id'),
                   user_name=user.get('name'),
                   user_login=user.get('login')))
        return code


class LoadModelOperation(Operation):
    MODEL_PARAM = 'model'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.parameters = parameters

        self.model = parameters.get(self.MODEL_PARAM)
        if not self.model:
            msg = 'Missing parameter model'
            raise ValueError(msg)

        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.output_model = named_outputs.get(
            'model', 'model_{}'.format(self.order))

    def generate_code(self):
        limonero_config = self.parameters.get('configuration') \
            .get('juicer').get('services').get('limonero')

        url = limonero_config['url']
        token = str(limonero_config['auth_token'])

        model_data = query_limonero(url, '/models', token, self.model)
        parts = model_data['class_name'].split('.')
        url = model_data['storage']['url']
        if url[-1] != '/':
            url += '/'

        path = '{}{}'.format(url, model_data['path'])

        code = dedent("""
            from {pkg} import {cls}
            {output} = {cls}.load('{path}')
        """.format(
            output=self.output_model,
            path=path,
            cls=parts[-1],
            pkg='.'.join(parts[:-1])
        ))
        return code


class PCAOperation(Operation):
    K_PARAM = 'k'
    ATTRIBUTE_PARAM = 'attribute'
    OUTPUT_ATTRIBUTE_PARAM = 'output_attribute'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.parameters = parameters

        self.k = parameters.get(self.K_PARAM)
        if not self.k:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.K_PARAM, self.__class__))
        self.attribute = parameters.get(self.ATTRIBUTE_PARAM)
        if not self.attribute or len(self.attribute) == 0:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTE_PARAM, self.__class__))

        self.output_attribute = parameters.get(self.OUTPUT_ATTRIBUTE_PARAM,
                                               'pca_features')
        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) > 0,
                             self.contains_results()])
        self.output = named_outputs.get('output data',
                                        'out_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = dedent("""
            pca = PCA(k={k}, inputCol='{inputAttr}', outputCol='{outputAttr}')
            model = pca.fit({input})
            {out} = model.transform({input})
        """.format(
            k=self.k,
            inputAttr=self.attribute[0],
            outputAttr=self.output_attribute,
            input=input_data,
            out=self.output

        ))
        return code


class LSHOperation(Operation):
    NUM_HASH_TABLES_PARAM = 'num_hash_tables'
    ATTRIBUTE_PARAM = 'attribute'
    OUTPUT_ATTRIBUTE_PARAM = 'output_attribute'
    TYPE_ATTRIBUTE = 'type'
    BUCKET_LENGTH_PARAM = 'bucket_length'
    SEED_PARAM = 'seed'

    TYPES = ['min-hash-lsh', 'bucketed-random']

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.parameters = parameters

        self.num_hash_tables = parameters.get(self.NUM_HASH_TABLES_PARAM)
        if not self.num_hash_tables:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.NUM_HASH_TABLES_PARAM, self.__class__))

        self.attribute = parameters.get(self.ATTRIBUTE_PARAM)
        if not self.attribute or len(self.attribute) == 0:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTE_PARAM, self.__class__))

        self.type = parameters.get(self.TYPE_ATTRIBUTE, 'min-hash-lsh')
        if self.type not in self.TYPES:
            raise ValueError(
                _("Invalid type '{}' for class {}").format(
                    self.type, self.__class__))

        self.bucket_length = parameters.get(self.BUCKET_LENGTH_PARAM) or 100
        self.seed = parameters.get(self.SEED_PARAM)

        self.output_attribute = parameters.get(self.OUTPUT_ATTRIBUTE_PARAM,
                                               'hashes')
        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) > 0,
                             self.contains_results()])
        self.output = named_outputs.get('output data',
                                        'out_{}'.format(self.order))

        self.output_model = named_outputs.get('model',
                                              'model_{}'.format(self.order))

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.output_model])

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = dedent("""
            hash_type = '{type}'
            if hash_type == 'bucketed-random':
                lsh = BucketedRandomProjectionLSH(
                    inputCol='{inputAttr}',
                    outputCol='{outputAttr}',
                    bucketLength={bucket_length},
                    numHashTables={num_hash_tables})
            elif hash_type == 'min-hash-lsh':
                lsh = MinHashLSH(
                    inputCol='{inputAttr}',
                    outputCol='{outputAttr}',
                    numHashTables={num_hash_tables})
            {model} = lsh.fit({input})
            {out} = {model}.transform({input})
        """.format(
            num_hash_tables=self.num_hash_tables,
            bucket_length=self.bucket_length,
            inputAttr=self.attribute[0],
            outputAttr=self.output_attribute,
            input=input_data,
            out=self.output,
            model=self.output_model,
            type=self.type,

        ))
        return code


class OutlierDetectionOperation(Operation):
    FEATURES_PARAM = 'features'
    ALIAS_PARAM = 'alias'
    MIN_POINTS_PARAM = 'min_points'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 1,
                             self.contains_results()])

        if self.FEATURES_PARAM not in parameters:
            msg = _("Parameter '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.FEATURES_PARAM, self.__class__))

        self.features = parameters.get(self.FEATURES_PARAM)
        self.min_points = parameters.get(self.MIN_POINTS_PARAM, 5)
        self.alias = parameters.get(self.ALIAS_PARAM, 'outlier')

        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs.get('input data')

        code = dedent("""
        emit = get_emitter(emit_event, {operation_id},
                           '{task_id}')

        from juicer.spark.ext import LocalOutlierFactor
        algorithm = LocalOutlierFactor(minPts={min_pts})

        features = [{features}]
        stages = []

        if isinstance({input}.schema[str(features[0])].dataType, VectorUDT):
            if len(features) > 1 :
                emit(message=_(
                'Features are assembled as a vector, but there are other '
                'attributes in the list of features. They will be ignored'),)

            algorithm.setFeaturesCol(features[0])
            algorithm.setOutputCol('{prediction}')
            stages.append(algorithm)
        else:
            emit(message=_(
                'Features are not assembled as a vector. They will be '
                'implicitly assembled and rows with null values will be '
                'discarded. If this is undesirable, explicitly add a feature '
                'assembler in the workflow.'),)
            to_assemble = []
            for f in features:
                if not dataframe_util.is_numeric({input}.schema, f):
                    name = f + '_tmp'
                    to_assemble.append(name)
                    stages.append(feature.StringIndexer(
                        inputCol=f, outputCol=name, handleInvalid='keep'))
                else:
                    to_assemble.append(f)

            # Remove rows with null (VectorAssembler doesn't support it)
            cond = ' AND '.join(['{{}} IS NOT NULL '.format(c)
                for c in to_assemble])
            stages.append(SQLTransformer(
                statement='SELECT * FROM __THIS__ WHERE {{}}'.format(cond)))

            final_features = 'features_tmp'
            stages.append(feature.VectorAssembler(
                inputCols=to_assemble, outputCol=final_features))

            algorithm.setFeaturesCol(final_features)
            algorithm.setOutputCol('{prediction}')
            stages.append(algorithm)


        pipeline = Pipeline(stages=stages)
        model = pipeline.fit({input})

        # Join between original dataset and results
        indexed_df = dataframe_util.with_column_index({input}, 'tmp_inx')
        lof_result = model.transform({input})
        {output} = indexed_df.join(
            lof_result, lof_result['index'] == indexed_df['tmp_inx'], 'left')
        {output} = {output}.drop(
            'tmp_inx', 'index', 'vector').withColumnRenamed(
            'lof', '{prediction}')

        """.format(
            input=input_data,
            features=', '.join(["'{}'".format(f) for f in self.features]),
            prediction=self.alias,
            min_pts=self.min_points,
            output=self.output,
            task_id=self.parameters['task_id'],
            operation_id=self.parameters['operation_id'],
        ))
        return code
