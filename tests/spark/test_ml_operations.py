# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import ast
import json
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.runner import configuration
from juicer.spark.ml_operation import FeatureIndexerOperation, \
    FeatureAssemblerOperation, \
    OneHotEncoderOperation, \
    IndexToStringOperation, \
    ApplyModelOperation, EvaluateModelOperation, \
    CrossValidationOperation, ClassificationModelOperation, \
    ClassifierOperation, SvmClassifierOperation, \
    DecisionTreeClassifierOperation, \
    GBTClassifierOperation, \
    NaiveBayesClassifierOperation, \
    RandomForestClassifierOperation, \
    LogisticRegressionClassifierOperation, \
    PerceptronClassifier, \
    ClusteringOperation, ClusteringModelOperation, \
    LdaClusteringOperation, KMeansClusteringOperation, \
    GaussianMixtureClusteringOperation, TopicReportOperation, \
    CollaborativeOperation, AlternatingLeastSquaresOperation

from tests import compare_ast, format_code_comparison

'''
 FeatureIndexer tests
'''


def test_feature_indexer_operation_success():
    params = {
        FeatureIndexerOperation.TYPE_PARAM: 'string',
        FeatureIndexerOperation.ATTRIBUTES_PARAM: ['col'],
        FeatureIndexerOperation.ALIAS_PARAM: 'c',
        FeatureIndexerOperation.MAX_CATEGORIES_PARAM: '20',
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = FeatureIndexerOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        col_alias = dict(tuple({alias}))
        indexers = [feature.StringIndexer(inputCol=col, outputCol=alias,
                            handleInvalid='keep')
                    for col, alias in col_alias.items()]

        # Use Pipeline to process all attributes once
        pipeline = Pipeline(stages=indexers)
        models_task_1 = dict([(c, indexers[i].fit({in1}))
                  for i, c in enumerate(col_alias.values())])

        # labels = [model.labels for model in models.itervalues()]
        # Spark ML 2.0.1 do not deal with null in indexer.
        # See SPARK-11569

        # input_1_without_null = input_1.na.fill('NA', subset=col_alias.keys())
        {in1}_without_null = {in1}.na.fill('NA', subset=col_alias.keys())
        {out} = pipeline.fit({in1}_without_null).transform({in1}_without_null)

        for k, v in col_alias.items():
            field_metadata = output_1.schema[str(v)].metadata
            if not 'lemon_attr' in field_metadata:
                field_metadata['lemon_attr'] = {{}}
            field_metadata['lemon_attr']['indexed_from'] = k
            {out} = {out}.withColumn(
                    v, functions.col(v).alias('', metadata=field_metadata))

        if 'indexer' not in cached_state:
            cached_state['indexers'] = {{}}
        for name, model in models_task_1.items():
            cached_state['indexers'][name] = model

        """.format(attr=params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
                   in1=in1,
                   out=out,
                   alias=json.dumps(
                       zip(params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
                           params[FeatureIndexerOperation.ALIAS_PARAM]))))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_feature_indexer_string_type_param_operation_failure():
    params = {
        FeatureIndexerOperation.ATTRIBUTES_PARAM: ['col'],
        FeatureIndexerOperation.TYPE_PARAM: 'XxX',
        FeatureIndexerOperation.ALIAS_PARAM: 'c',
        FeatureIndexerOperation.MAX_CATEGORIES_PARAM: '20',
    }

    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}

        indexer = FeatureIndexerOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
        indexer.generate_code()


def test_feature_indexer_string_missing_attribute_param_operation_failure():
    params = {
        FeatureIndexerOperation.TYPE_PARAM: 'string',
        FeatureIndexerOperation.ALIAS_PARAM: 'c',
        FeatureIndexerOperation.MAX_CATEGORIES_PARAM: '20',
    }

    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        FeatureIndexerOperation(params, named_inputs=n_in,
                                named_outputs=n_out)


def test_feature_indexer_vector_missing_attribute_param_operation_failure():
    params = {
        FeatureIndexerOperation.TYPE_PARAM: 'string',
        FeatureIndexerOperation.ALIAS_PARAM: 'c',
        FeatureIndexerOperation.MAX_CATEGORIES_PARAM: '20',
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        FeatureIndexerOperation(params, named_inputs=n_in,
                                named_outputs=n_out)


def test_feature_indexer_missing_attribute_failure():
    params = {
        FeatureIndexerOperation.TYPE_PARAM: 'vector',
        FeatureIndexerOperation.ALIAS_PARAM: 'c',
        FeatureIndexerOperation.MAX_CATEGORIES_PARAM: 20,
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1', 'indexer models': 'models'}

    with pytest.raises(ValueError):
        instance = FeatureIndexerOperation(params, named_inputs=n_in,
                                           named_outputs=n_out)


def test_feature_indexer_output_names_success():
    params = {

        FeatureIndexerOperation.TYPE_PARAM: 'vector',
        FeatureIndexerOperation.ATTRIBUTES_PARAM: ['col'],
        FeatureIndexerOperation.ALIAS_PARAM: 'c',
        FeatureIndexerOperation.MAX_CATEGORIES_PARAM: 20,
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1', 'indexer models': 'models'}

    instance = FeatureIndexerOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)
    assert instance.get_data_out_names() == n_out['output data']
    assert instance.get_output_names() == ','.join([n_out['output data'],
                                                    n_out['indexer models']])


def test_feature_indexer_vector_operation_success():
    params = {

        FeatureIndexerOperation.TYPE_PARAM: 'vector',
        FeatureIndexerOperation.ATTRIBUTES_PARAM: ['col'],
        FeatureIndexerOperation.ALIAS_PARAM: 'c',
        FeatureIndexerOperation.MAX_CATEGORIES_PARAM: 20,
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = FeatureIndexerOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
            col_alias = dict({3})
            indexers = [feature.VectorIndexer(maxCategories={4},
                            inputCol=col, outputCol=alias)
                            for col, alias in col_alias.items()]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=indexers)
            models_task_1 = dict([(col, indexers[i].fit({1})) for i, col in
                        enumerate(col_alias.values())])
            labels = None

            # Spark ML 2.0.1 do not deal with null in indexer.
            # See SPARK-11569
            {1}_without_null = {1}.na.fill('NA', subset=col_alias.keys())

            {2} = pipeline.fit({1}_without_null).transform({1}_without_null)
            for k, v in col_alias.items():
                field_metadata = output_1.schema[str(v)].metadata
                if not 'lemon_attr' in field_metadata:
                    field_metadata['lemon_attr'] = {{}}
                field_metadata['lemon_attr']['indexed_from'] = k
                {2} = {2}.withColumn(
                        v, functions.col(v).alias('', metadata=field_metadata))
            if 'indexer' not in cached_state:
                cached_state['indexers'] = {{}}
            for name, model in models_task_1.items():
                cached_state['indexers'][name] = model

            """.format(params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
                       in1,
                       out,
                       json.dumps(
                           zip(params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
                               params[FeatureIndexerOperation.ALIAS_PARAM])),
                       params[FeatureIndexerOperation.MAX_CATEGORIES_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_feature_indexer_vector_operation_failure():
    params = {

        FeatureIndexerOperation.TYPE_PARAM: 'vector',
        FeatureIndexerOperation.ATTRIBUTES_PARAM: ['col'],
        FeatureIndexerOperation.ALIAS_PARAM: 'c',
        FeatureIndexerOperation.MAX_CATEGORIES_PARAM: -1,
    }

    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        FeatureIndexerOperation(params, named_inputs=n_in,
                                named_outputs=n_out)


def test_feature_indexer_operation_failure():
    params = {}
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        FeatureIndexerOperation(params, named_inputs=n_in,
                                named_outputs=n_out)


'''
IndexToStringOperation tests
'''


def test_index_to_string_missing_required_failure():
    params = {}
    with pytest.raises(ValueError) as e:
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        IndexToStringOperation(params, named_inputs=n_in,
                               named_outputs=n_out)
    assert IndexToStringOperation.ATTRIBUTES_PARAM in str(e)

    params[IndexToStringOperation.ATTRIBUTES_PARAM] = 'name'
    with pytest.raises(ValueError) as e:
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        IndexToStringOperation(params, named_inputs=n_in,
                               named_outputs=n_out)
    assert IndexToStringOperation.ORIGINAL_NAMES_PARAM in str(e)


def test_index_to_string_success():
    params = {
        IndexToStringOperation.ATTRIBUTES_PARAM: ['name'],
        IndexToStringOperation.ORIGINAL_NAMES_PARAM: ['original'],
        IndexToStringOperation.ALIAS_PARAM: 'new_name',
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    instance = IndexToStringOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
            original_names = {original}
            col_alias = dict([["{features}", "{alias}"]])
            converter = [feature.IndexToString(
                inputCol=col, outputCol=alias, labels=[])
                     for i, (col, alias) in enumerate(col_alias.items())]
            pipeline = Pipeline(stages=converter)

            {output_1} = pipeline.fit({input_1}).transform({input_1})
            """.format(
        features=params[IndexToStringOperation.ATTRIBUTES_PARAM][0],
        output_1=n_out['output data'],
        input_1=n_in['input data'],
        original=json.dumps(
            params[IndexToStringOperation.ORIGINAL_NAMES_PARAM]),
        alias=params[IndexToStringOperation.ALIAS_PARAM],
    ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


'''
OneHotEncoderOperation tests
'''


def test_one_hot_encoder_missing_required_failure():
    params = {}
    with pytest.raises(ValueError) as e:
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        OneHotEncoderOperation(params, named_inputs=n_in,
                               named_outputs=n_out)
    assert OneHotEncoderOperation.ATTRIBUTES_PARAM in str(e)


def test_one_hot_encoder_success():
    params = {
        OneHotEncoderOperation.ATTRIBUTES_PARAM: ['name'],
        OneHotEncoderOperation.ALIAS_PARAM: 'new_name',
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    instance = OneHotEncoderOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
            col_alias = dict([["{features}", "{alias}"]])
            encoders = [feature.OneHotEncoder(
                inputCol=col, outputCol=alias, dropLast=True)
                     for (col, alias) in col_alias.items()]
            pipeline = Pipeline(stages=encoders)

            {output_1} = pipeline.fit({input_1}).transform({input_1})
            """.format(
        features=params[OneHotEncoderOperation.ATTRIBUTES_PARAM][0],
        output_1=n_out['output data'],
        input_1=n_in['input data'],
        alias=params[OneHotEncoderOperation.ALIAS_PARAM],
    ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_one_hot_encoder_output_names_success():
    params = {

        OneHotEncoderOperation.ATTRIBUTES_PARAM: ['col'],
        OneHotEncoderOperation.ALIAS_PARAM: 'c',
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = OneHotEncoderOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)
    assert instance.get_data_out_names() == n_out['output data']
    assert instance.get_output_names() == n_out['output data']

    instance = OneHotEncoderOperation(params, named_inputs={},
                                      named_outputs={})
    assert instance.get_data_out_names() == 'out_task_1'
    assert instance.get_output_names() == 'out_task_1'


'''
 FeatureAssembler tests
'''


def test_feature_assembler_operation_success():
    params = {
        FeatureAssemblerOperation.ATTRIBUTES_PARAM: ['col'],
        FeatureAssemblerOperation.ALIAS_PARAM: 'c'
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = FeatureAssemblerOperation(params, named_inputs=n_in,
                                         named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
            assembler = feature.VectorAssembler(inputCols={features},
                                                outputCol="{alias}")
            {input_1}_without_null = {input_1}.na.drop(subset={features})
            {output_1} = assembler.transform({input_1}_without_null)


            """.format(features=json.dumps(
        params[FeatureIndexerOperation.ATTRIBUTES_PARAM]),
        alias=params[FeatureAssemblerOperation.ALIAS_PARAM],
        output_1=out,
        input_1=in1))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_feature_assembler_operation_failure():
    params = {
        # FeatureAssembler.ATTRIBUTES_PARAM: ['col'],
        FeatureAssemblerOperation.ALIAS_PARAM: 'c'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        FeatureAssemblerOperation(params, named_inputs=n_in,
                                  named_outputs=n_out)


'''
 ApplyModel tests
'''


def test_apply_model_operation_success():
    params = {}

    n_in = {'input data': 'input_1', 'model': 'model1'}
    n_out = {'output data': 'output_1', 'model': 'model1'}

    in1 = n_in['input data']
    model = n_in['model']
    out = n_out['output data']

    instance = ApplyModelOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
    # params = {{'predictionCol': 'prediction'}}
    params = {{}}
    {output_1} = {model}.transform({input_1}, params)
    """.format(
        output_1=out, input_1=in1, model=model))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


'''
    EvaluateModel tests
'''


@pytest.mark.skip
def test_evaluate_model_operation_success():
    params = {

        EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'c',
        EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: 'c',
        EvaluateModelOperation.METRIC_PARAM: 'f1',
        'task_id': '2323-afffa-343bdaff',
        'workflow_id': 203,
        'workflow_name': 'test',
        'job_id': 34,
        'user': {
            'id': 12,
            'name': 'admin',
            'login': 'admin'
        },
        'operation_id': 2793,
        'task': {
            'forms': {}
        }
    }
    configuration.set_config(
        {
            'juicer': {
                'services': {
                    'limonero': {
                        'url': 'http://localhost',
                        'auth_token': 'FAKE',
                    },
                    'caipirinha': {
                        'url': 'http://localhost',
                        'auth_token': 'FAKE',
                        'storage_id': 343
                    }
                }
            }
        }
    )
    n_in = {'input data': 'input_1', 'model': 'df_model'}
    n_out = {'metric': 'metric_value', 'evaluator': 'df_evaluator'}
    instance = EvaluateModelOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        metric_value = 0.0
        display_text = True
        display_image = True

        metric = '{metric}'
        if metric in ['areaUnderROC', 'areaUnderPR']:
            evaluator = evaluation.BinaryClassificationEvaluator(
                predictionCol='{predic_atr}',
                labelCol='{label_atr}',
                metricName=metric)
            metric_value = evaluator.evaluate({input_1})
            if display_text:
                result = '<h4>{{}}: {{}}</h4>'.format('{metric}',
                    metric_value)

                emit_event(
                   'update task', status='COMPLETED',
                   identifier='{task_id}',
                   message=result,
                   type='HTML', title='Evaluation result',
                   task={{'id': '{task_id}'}},
                   operation={{'id': {operation_id}}},
                   operation_id={operation_id})

        elif metric in ['f1', 'weightedPrecision', 'weightedRecall',
                'accuracy']:
            label_prediction = input_1.select(
                '{label_atr}', '{predic_atr}')
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
                classes = ['c: {{}}'.format(x[0]) for x in
                    label_prediction.select(
                        'c').distinct().sort(
                            'c', ascending=True).collect()]

                content = ConfusionMatrixImageReport(
                    cm=evaluator.confusionMatrix().toArray(),
                    classes=classes,)

                emit_event(
                   'update task', status='COMPLETED',
                   identifier='{task_id}',
                   message=content.generate(),
                   type='IMAGE', title='Evaluation result',
                   task={{'id': '{task_id}'}},
                   operation={{'id': {operation_id}}},
                   operation_id={operation_id})


            if display_text:
                headers = ['Metric', 'Value']
                rows = [
                    ['F1', evaluator.weightedFMeasure()],
                    ['Weighted Precision', evaluator.weightedPrecision],
                    ['Weighted Recall', evaluator.weightedRecall],
                    ['Accuracy', evaluator.accuracy],
                ]

                content = SimpleTableReport(
                        'table table-striped table-bordered table-sm',
                        headers, rows, title='Evaluation result')

                emit_event(
                   'update task', status='COMPLETED',
                   identifier='{task_id}',
                   message=content.generate(),
                   type='HTML', title='Evaluation result',
                   task={{'id': '{task_id}'}},
                   operation={{'id': {operation_id}}},
                   operation_id={operation_id})

        from juicer.spark.ml_operation import ModelsEvaluationResultList
        model_task_1 = ModelsEvaluationResultList(
            [df_model], df_model, '{metric}', metric_value)
        f1 = metric_value
        model_task_1 = None
            """.format(output=n_out['metric'],
                       evaluator_out=n_out['evaluator'],
                       input_2=n_in['model'],
                       input_1=n_in['input data'],
                       task_id=params['task_id'],
                       operation_id=params['operation_id'],
                       predic_atr=params[
                           EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM],
                       label_atr=params[
                           EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM],
                       metric=params[EvaluateModelOperation.METRIC_PARAM],
                       evaluator=EvaluateModelOperation.METRIC_TO_EVALUATOR[
                           params[EvaluateModelOperation.METRIC_PARAM]][0],
                       predic_col=EvaluateModelOperation.METRIC_TO_EVALUATOR[
                           params[EvaluateModelOperation.METRIC_PARAM]][1]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code,
                                                expected_code)


def test_evaluate_model_operation_wrong_metric_param_failure():
    params = {

        EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'c',
        EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: 'c',
        EvaluateModelOperation.METRIC_PARAM: 'mist',
    }
    n_in = {'input data': 'input_1', 'model': 'df_model'}
    n_out = {'metric': 'df_metric', 'evaluator': 'df_evaluator'}
    with pytest.raises(ValueError):
        EvaluateModelOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_evaluate_model_operation_missing_metric_param_failure():
    params = {

        EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'c',
        EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: 'c',
        EvaluateModelOperation.METRIC_PARAM: '',
    }
    n_in = {'input data': 'input_1', 'model': 'df_model'}
    n_out = {'metric': 'df_metric', 'evaluator': 'df_evaluator'}
    with pytest.raises(ValueError):
        EvaluateModelOperation(params, named_inputs=n_in,
                               named_outputs=n_out)


def test_evaluate_model_operation_missing_prediction_attribute_failure():
    params = {

        EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: '',
        EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: 'c',
        EvaluateModelOperation.METRIC_PARAM: 'f1',
    }
    n_in = {'input data': 'input_1', 'model': 'df_model'}
    n_out = {'metric': 'df_metric', 'evaluator': 'df_evaluator'}
    with pytest.raises(ValueError):
        EvaluateModelOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_evaluate_model_operation_missing_label_attribute_failure():
    params = {

        EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'c',
        EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: '',
        EvaluateModelOperation.METRIC_PARAM: 'f1',
    }
    n_in = {'input data': 'input_1', 'model': 'df_model'}
    n_out = {'metric': 'df_metric', 'evaluator': 'df_evaluator'}
    with pytest.raises(ValueError):
        EvaluateModelOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    CrossValidationOperation Tests
'''


def test_cross_validation_partial_operation_success():
    params = {
        'task_id': 232,
        'operation_id': 1,
        CrossValidationOperation.NUM_FOLDS_PARAM: 3,
        CrossValidationOperation.EVALUATOR_PARAM: 'accuracy',
        CrossValidationOperation.LABEL_ATTRIBUTE_PARAM: ['label'],

    }
    n_in = {'algorithm': 'df_1', 'input data': 'df_2', 'evaluator': 'xpto'}
    n_out = {'scored data': 'output_1', 'evaluation': 'eval_1'}
    outputs = ['output_1']

    instance = CrossValidationOperation(params, named_inputs=n_in,
                                        named_outputs=n_out, )

    code = instance.generate_code()

    expected_code = dedent("""
            grid_builder = tuning.ParamGridBuilder()
            estimator, param_grid, metric = {algorithm}

            for param_name, values in param_grid.items():
                param = getattr(estimator, param_name)
                grid_builder.addGrid(param, values)

            evaluator = evaluation.MulticlassClassificationEvaluator(
                predictionCol='prediction',
                labelCol=label_col,
                metricName='{metric}')

            estimator.setLabelCol(label_col)
            estimator.setPredictionCol('prediction')

            cross_validator = tuning.CrossValidator(
                 estimator=estimator,
                 estimatorParamMaps=grid_builder.build(),
                 evaluator=evaluator, numFolds=3)


            cv_model = cross_validator.fit({input_data})
            fit_data = cv_model.transform({input_data})
            best_model_1  = cv_model.bestModel
            metric_result = evaluator.evaluate(fit_data)
            # {evaluation} = metric_result
            {output} = fit_data
            models_task_1 = None

            """.format(
        algorithm=n_in['algorithm'],
        input_data=n_in['input data'],
        evaluator=n_in['evaluator'],
        evaluation='eval_1',
        output=outputs[0],
        metric=params[CrossValidationOperation.EVALUATOR_PARAM],
        label=params[CrossValidationOperation.LABEL_ATTRIBUTE_PARAM][0],
        folds=params[CrossValidationOperation.NUM_FOLDS_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_cross_validation_complete_operation_success():
    params = {

        CrossValidationOperation.NUM_FOLDS_PARAM: 3,
        'task_id': '2323-afffa-343bdaff',
        'operation_id': 2793,
        CrossValidationOperation.EVALUATOR_PARAM: 'weightedRecall',
        CrossValidationOperation.LABEL_ATTRIBUTE_PARAM: ['label']

    }
    n_in = {'algorithm': 'algo1', 'input data': 'df_1', 'evaluator': 'ev_1'}
    n_out = {'evaluation': 'output_1', 'scored data': 'output_1'}
    outputs = ['output_1']

    instance = CrossValidationOperation(
        params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
            grid_builder = tuning.ParamGridBuilder()
            estimator, param_grid, metric = {algorithm}

            for param_name, values in param_grid.items():
                param = getattr(estimator, param_name)
                grid_builder.addGrid(param, values)

            evaluator = evaluation.MulticlassClassificationEvaluator(
                predictionCol='prediction',
                labelCol=label_col,
                metricName='{metric}')

            estimator.setLabelCol(label_col)
            estimator.setPredictionCol('prediction')

            cross_validator = tuning.CrossValidator(
                estimator=estimator, estimatorParamMaps=grid_builder.build(),
                evaluator=evaluator, numFolds={folds})
            cv_model = cross_validator.fit({input_data})
            fit_data = cv_model.transform({input_data})
            best_model_1  = cv_model.bestModel
            metric_result = evaluator.evaluate(fit_data)
            # {output} = metric_result
            {output} = fit_data
            models_task_1 = None

            """.format(
        algorithm=n_in['algorithm'],
        input_data=n_in['input data'],
        evaluator=n_in['evaluator'],
        output=outputs[0],
        label=params[CrossValidationOperation.LABEL_ATTRIBUTE_PARAM][0],
        metric=params[CrossValidationOperation.EVALUATOR_PARAM],
        folds=params[CrossValidationOperation.NUM_FOLDS_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_cross_validation_complete_operation_missing_input_failure():
    params = {
        CrossValidationOperation.EVALUATOR_PARAM: 'f1',
        CrossValidationOperation.NUM_FOLDS_PARAM: 3,
        CrossValidationOperation.LABEL_ATTRIBUTE_PARAM: 'label'

    }
    n_in = {'algorithm': 'algo1'}
    n_out = {'evaluation': 'output_1'}

    instance = CrossValidationOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)
    assert not instance.has_code


'''
    ClassificationModel tests
'''


def test_classification_model_operation_success():
    params = {
        ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',
        ClassificationModelOperation.LABEL_ATTRIBUTE_PARAM: 'l',
        ClassificationModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'prediction1',
        'task_id': 'a7asf788',
        'operation_id': 200,

    }
    n_in = {'algorithm': 'algo_param', 'train input data': 'train'}
    n_out = {'model': 'model_1', 'output': 'out_classification'}
    instance = ClassificationModelOperation(params, named_inputs=n_in,
                                            named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        emit = functools.partial(
                emit_event, name='update task',
                status='RUNNING', type='TEXT',
                identifier='{task_id}',
                operation={{'id': {operation_id}}}, operation_id={operation_id},
                task={{'id': '{task_id}'}},
                title='{title}')
        alg, param_grid, metrics = {algorithm}

        params = dict([(p.name, v) for p, v in
            alg.extractParamMap().items()])

        if isinstance(alg, MultilayerPerceptronClassifier):
            del params['rawPredictionCol']
        algorithm_cls = globals()[alg.__class__.__name__]
        algorithm = algorithm_cls()
        algorithm.setParams(**params)


        algorithm.setPredictionCol('{prediction}')
        requires_pipeline = False
        stages = []
        features = {features}
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

        setattr(model_1, 'ensemble_weights', [1.0])

        # Lazy execution in case of sampling the data in UI
        def call_transform(df):
            return model_1.transform(df)
        out_task_1 = dataframe_util.LazySparkTransformationDataframe(
            {model}, {train}, call_transform)

        display_text = True
        if display_text:
            rows = [[m, getattr(model_1, m)] for m in metrics
                if hasattr(model_1, m)]
            headers = [u'Parameter', u'Value']
            content = SimpleTableReport(
                'table table-striped table-bordered table-sm',
                headers, rows)

            result = '<h4>Generated classification model parameters</h4>'

            emit(status='COMPLETED', message=result + content.generate(),
                type='HTML', title='Generated classification model parameters')

        """.format(model=n_out['model'],
                   output=n_out['output'],
                   train=n_in['train input data'],
                   algorithm=n_in['algorithm'],
                   msg0=_(
                       'Features are not assembled as a vector. They will be '
                       'implicitly assembled and rows with null values will be '
                       'discarded. If this is undesirable, explicitly add a '
                       'feature assembler in the workflow.'),
                   msg1=_('Label attribute is categorical, it will be '
                          'implicitly indexed as string.'),
                   task_id=params['task_id'],
                   operation_id=params['operation_id'],
                   title=_('Generated classification model parameters'),
                   prediction=params[
                       ClassificationModelOperation.PREDICTION_ATTRIBUTE_PARAM],
                   features=repr(params[
                                     ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM]),
                   label=params[
                       ClassificationModelOperation.LABEL_ATTRIBUTE_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_classification_model_operation_failure():
    params = {
        ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',
        ClassificationModelOperation.LABEL_ATTRIBUTE_PARAM: 'l'

    }
    with pytest.raises(ValueError):
        n_in = {'train input data': 'train'}
        n_out = {'model': 'model_1'}
        instance = ClassificationModelOperation(params, named_inputs=n_in,
                                                named_outputs=n_out)
        instance.generate_code()


def test_classification_model_operation_missing_features_failure():
    params = {
        ClassificationModelOperation.LABEL_ATTRIBUTE_PARAM: 'label'
    }
    n_in = {'a': 'a', 'b': 'b'}
    n_out = {'c': 'c'}

    with pytest.raises(ValueError):
        ClassificationModelOperation(
            params, named_inputs=n_in, named_outputs=n_out)


def test_classification_model_operation_missing_label_failure():
    params = {
        ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM: 'features',

    }
    with pytest.raises(ValueError):
        n_in = {}
        n_out = {'model': 'model_1'}
        ClassificationModelOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)


def test_classification_model_operation_missing_inputs_failure():
    params = {}
    n_in = {'algorithm': 'algo_param', 'train input data': 'train'}
    n_out = {'model': 'model_1'}

    with pytest.raises(ValueError):
        classification_model = ClassificationModelOperation(
            params, named_inputs=n_in, named_outputs=n_out)

        classification_model.generate_code()


'''
    ClassifierOperation tests
'''


# @FIX-ME
def test_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    n_out = {'algorithm': 'classifier_1'}

    instance = ClassifierOperation(params, named_inputs={}, named_outputs=n_out)

    code = instance.generate_code()

    param_grid = {
        "labelCol": params[ClassifierOperation.GRID_PARAM]
        [ClassifierOperation.LABEL_PARAM],
        "featuresCol": params[ClassifierOperation.GRID_PARAM]
        [ClassifierOperation.FEATURES_PARAM]
    }

    expected_code = dedent("""

    param_grid = {{}}
    # Output result is the classifier and its parameters. Parameters are
    # need in classification model or cross validator.
    {output} = ({name}, param_grid, [])
    """.format(output=n_out['algorithm'], name='BaseClassifier()',
               param_grid=json.dumps(param_grid)))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_classifier_operation_missing_output_failure():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    n_out = {}

    with pytest.raises(ValueError):
        classifier = ClassifierOperation(params, named_inputs={},
                                         named_outputs=n_out)
        classifier.generate_code()


def test_svm_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    n_out = {'algorithm': 'classifier_1'}

    instance_svm = SvmClassifierOperation(params, named_inputs={},
                                          named_outputs=n_out)

    # Is not possible to generate_code(), because has_code is False
    assert (instance_svm.name ==
            "classification.LinearSVC(**{})")


def test_lr_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = LogisticRegressionClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    # Is not possible to generate_code(), because has_code is False
    # noinspection PyUnresolvedReferences
    assert instance_lr.name == 'classification.LogisticRegression(**{})'


def test_dt_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    n_out = {'algorithm': 'classifier_1'}

    instance_dt = DecisionTreeClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    # Is not possible to generate_code(), because has_code is False
    # noinspection PyUnresolvedReferences
    assert instance_dt.name == 'classification.DecisionTreeClassifier(**{})'


def test_gbt_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    n_out = {'algorithm': 'classifier_1'}

    instance_dt = GBTClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    # Is not possible to generate_code(), because has_code is False
    # noinspection PyUnresolvedReferences
    assert instance_dt.name == 'classification.GBTClassifier(**{})'


def test_nb_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    n_out = {'algorithm': 'classifier_1'}

    instance_nb = NaiveBayesClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    # Is not possible to generate_code(), because has_code is False
    # noinspection PyUnresolvedReferences
    assert instance_nb.name == 'classification.NaiveBayes(**{})'


def test_rf_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    n_out = {'algorithm': 'classifier_1'}
    instance_nb = RandomForestClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    # Is not possible to generate_code(), because has_code is False
    # noinspection PyUnresolvedReferences
    assert instance_nb.name == 'classification.RandomForestClassifier(**{})'


def test_perceptron_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    n_out = {'algorithm': 'classifier_1'}

    instance_pct = PerceptronClassifier(
        params, named_inputs={}, named_outputs=n_out)

    # Is not possible to generate_code(), because has_code is False
    # noinspection PyUnresolvedReferences
    assert (instance_pct.name ==
            'classification.MultilayerPerceptronClassifier(**{})')


"""
    Clustering tests
"""


def test_clustering_model_operation_success():
    params = {

        ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',
        'task_id': 232,
        'operation_id': 343

    }
    named_inputs = {'algorithm': 'algorithm1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_model'}
    outputs = ['output_1']

    instance = ClusteringModelOperation(params, named_inputs=named_inputs,
                                        named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
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
            algorithm.setPredictionCol('prediction')

        stages.append(algorithm)
        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(df_2)
        {model} = pipeline_model
        setattr({model}, 'features', u'f')

        # Lazy execution in case of sampling the data in UI
        def call_transform(df):
            if requires_pipeline:
                return pipeline_model.transform(df).drop(final_features)
            else:
                return pipeline_model.transform(df)
        output_1 = dataframe_util.LazySparkTransformationDataframe(
             {model}, df_2, call_transform)

        summary = getattr({model}, 'summary', None)
        def call_clusters(df):
            if hasattr({model}, 'clusterCenters'):
                return spark_session.createDataFrame(
                    [center.tolist()
                        for center in {model}.clusterCenters()])
            else:
                return spark_session.createDataFrame([],
                    types.StructType([]))

        centroids_task_1 = dataframe_util.LazySparkTransformationDataframe(
            {model}, df_2, call_clusters)

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
                identifier='232',
                message=summary_content.generate(),
                type='HTML', title='Clustering result',
                task={{'id': '{task_id}' }},
                operation={{'id': {operation_id} }},
                operation_id={operation_id})
        """.format(algorithm=named_inputs['algorithm'],
                   input=named_inputs['train input data'],
                   model=named_outputs['model'],
                   output=outputs[0],
                   operation_id=params['operation_id'],
                   task_id=params['task_id'],
                   title=_("Clustering result"),
                   features=repr(params[
                       ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM]),
                   msg1=_('Regression only support numerical features.'),
                   msg2=_('Features are not assembled as a vector. '
                          'They will be implicitly assembled and rows with '
                          'null values will be discarded. If this is '
                          'undesirable, explicitly add a feature assembler '
                          'in the workflow.'),
                   ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_clustering_model_operation_missing_features_failure():
    params = {}
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}

    with pytest.raises(ValueError):
        ClusteringModelOperation(params,
                                 named_inputs=named_inputs,
                                 named_outputs=named_outputs)


def test_clustering_model_operation_missing_input_failure():
    params = {
        ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',
    }
    named_inputs = {'algorithm': 'df_1'}
    named_outputs = {'output data': 'output_1', 'model': 'output_2'}

    clustering = ClusteringModelOperation(params,
                                          named_inputs=named_inputs,
                                          named_outputs=named_outputs)
    assert not clustering.has_code


def test_clustering_model_operation_missing_output_success():
    params = {
        ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',
    }
    named_inputs = {'algorithm': 'df_1', 'train input data': 'df_2'}
    named_outputs = {'model': 'output_2'}

    clustering = ClusteringModelOperation(params,
                                          named_inputs=named_inputs,
                                          named_outputs=named_outputs)
    assert clustering.has_code


def test_clustering_operation_success():
    # This test its not very clear, @CHECK
    params = {}
    named_outputs = {'algorithm': 'clustering_algo_1'}

    name = 'BaseClustering'
    set_values = []
    instance = ClusteringOperation(params, named_inputs={},
                                   named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=named_outputs['algorithm'],
        name=name))

    settings = (['{0}.set{1}({2})'.format(named_outputs['model'], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_lda_clustering_operation_optimizer_online_success():
    params = {
        LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM: 10,
        LdaClusteringOperation.OPTIMIZER_PARAM: 'online',
        LdaClusteringOperation.MAX_ITERATIONS_PARAM: 10,
        LdaClusteringOperation.DOC_CONCENTRATION_PARAM: 0.25,
        LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM: 0.1
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    name = "clustering.LDA"

    set_values = [
        ['K', params[LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM]],
        ['MaxIter', params[LdaClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['Optimizer',
         "'{}'".format(params[LdaClusteringOperation.OPTIMIZER_PARAM])],
        ['DocConcentration', [0.25]],
        ['TopicConcentration',
         params[LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM]]
    ]

    instance = LdaClusteringOperation(params, named_inputs={},
                                      named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=named_outputs['algorithm'],
        name=name))

    settings = (['{0}.set{1}({2})'.format(
        named_outputs['algorithm'], name,
        v if not isinstance(v, list) else json.dumps(v))
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


# def test_lda_clustering_operation_optimizer_em_success():
#     params = {
#         LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM: 10,
#         LdaClusteringOperation.OPTIMIZER_PARAM: 'em',
#         LdaClusteringOperation.MAX_ITERATIONS_PARAM: 10,
#         LdaClusteringOperation.DOC_CONCENTRATION_PARAM: 0.25,
#         LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM: 0.1
#         LdaClusteringOperation.ONLINE_OPTIMIZER: '',
#         LdaClusteringOperation.EM_OPTIMIZER: ''
#
# }
# inputs = ['df_1', 'df_2']
# outputs = ['output_1']
#
# instance = LdaClusteringOperation(params, inputs,
#                                   outputs,
#                                   named_inputs={},
#                                   named_outputs={})
#
# code = instance.generate_code()
#
# expected_code = dedent("""
#     {input_2}.setLabelCol('{label}').setFeaturesCol('{features}')
#     {output} = {input_2}.fit({input_1})
#     """.format(output=outputs[0],
#                input_1=inputs[0],
#                input_2=inputs[1],
#                features=params[ClassificationModel.FEATURES_ATTRIBUTE_PARAM],
#                label=params[ClassificationModel.LABEL_ATTRIBUTE_PARAM]))
#
# result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
# assert result, msg + debug_ast(code, expected_code)


def test_lda_clustering_operation_failure():
    params = {
        LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM: 10,
        LdaClusteringOperation.OPTIMIZER_PARAM: 'xXx',
        LdaClusteringOperation.MAX_ITERATIONS_PARAM: 10,
        LdaClusteringOperation.DOC_CONCENTRATION_PARAM: 0.25,
        LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM: 0.1
    }
    named_outputs = {'algorithm': 'clustering_algo_2'}
    with pytest.raises(ValueError):
        LdaClusteringOperation(params, named_inputs={},
                               named_outputs=named_outputs)


def test_kmeans_clustering_operation_random_type_kmeans_success():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'kmeans',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'random',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    name = "clustering.KMeans"

    set_values = [
        ['MaxIter', params[KMeansClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[KMeansClusteringOperation.K_PARAM]],
        ['Tol', params[KMeansClusteringOperation.TOLERANCE_PARAMETER]],
        ['InitMode',
         '"{}"'.format(params[KMeansClusteringOperation.INIT_MODE_PARAMETER])]
    ]

    instance = KMeansClusteringOperation(params, named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=named_outputs['algorithm'],
        name=name))

    settings = (['{0}.set{1}({2})'.format(named_outputs['algorithm'], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_kmeans_clustering_operation_random_type_bisecting_success():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'bisecting',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'random',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    name = "BisectingKMeans"

    set_values = [
        ['MaxIter', params[KMeansClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[KMeansClusteringOperation.K_PARAM]],
    ]

    instance = KMeansClusteringOperation(params, named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=named_outputs['algorithm'],
        name=name))

    settings = (['{0}.set{1}({2})'.format(named_outputs['algorithm'], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_kmeans_clustering_operation_kmeans_type_kmeans_success():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'kmeans',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'k-means||',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}
    name = "clustering.KMeans"

    set_values = [
        ['MaxIter', params[KMeansClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[KMeansClusteringOperation.K_PARAM]],
        ['Tol', params[KMeansClusteringOperation.TOLERANCE_PARAMETER]],
        ['InitMode',
         '"{}"'.format(params[KMeansClusteringOperation.INIT_MODE_PARAMETER])]
    ]

    instance = KMeansClusteringOperation(params, named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=named_outputs['algorithm'], name=name))

    settings = (
        ['{0}.set{1}({2})'.format(named_outputs['algorithm'], name, v)
         for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_kmeans_clustering_operation_kmeansdd_type_bisecting_success():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'bisecting',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'k-means||',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    name = "BisectingKMeans"

    set_values = [
        ['MaxIter', params[KMeansClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[KMeansClusteringOperation.K_PARAM]],
    ]

    instance = KMeansClusteringOperation(params,
                                         named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=named_outputs['algorithm'],
        name=name))

    settings = (['{0}.set{1}({2})'.format(named_outputs['algorithm'], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_kmeans_clustering_operation_random_type_failure():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'XxX',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'random',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    with pytest.raises(ValueError):
        KMeansClusteringOperation(params, named_inputs={},
                                  named_outputs=named_outputs)


def test_gaussian_mixture_clustering_operation_success():
    params = {
        GaussianMixtureClusteringOperation.K_PARAM: 10,
        GaussianMixtureClusteringOperation.MAX_ITERATIONS_PARAM: 10,
        GaussianMixtureClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}
    name = "clustering.GaussianMixture"

    set_values = [
        ['MaxIter',
         params[GaussianMixtureClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[GaussianMixtureClusteringOperation.K_PARAM]],
        ['Tol', params[GaussianMixtureClusteringOperation.TOLERANCE_PARAMETER]],
    ]

    instance = GaussianMixtureClusteringOperation(params, named_inputs={},
                                                  named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=named_outputs['algorithm'],
        name=name))

    settings = (['{0}.set{1}({2})'.format(named_outputs['algorithm'], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


@pytest.mark.skip
def test_topics_report_operation_success():
    params = {
        TopicReportOperation.TERMS_PER_TOPIC_PARAM: 20,
    }
    named_inputs = {'model': 'df_1',
                    'input data': 'df_2',
                    'vocabulary': 'df_3'}
    named_outputs = {'output data': 'output_1'}

    instance = TopicReportOperation(params, named_inputs=named_inputs,
                                    named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
            topic_df = {model}.describeTopics(maxTermsPerTopic={tpt})
            # See hack in ClusteringModelOperation
            features = {model}.uid.split('|')[1]
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
            {output} =  {input}
        """.format(model=named_inputs['model'],
                   tpt=params[TopicReportOperation.TERMS_PER_TOPIC_PARAM],
                   vocabulary=named_inputs['vocabulary'],
                   output=named_outputs['output data'],
                   input=named_inputs['input data']))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


"""
  Collaborative Filtering tests
"""


def test_collaborative_filtering_operation_success():
    params = {
    }
    named_inputs = {'algorithm': 'df_1', 'train input data': 'df_2',
                    'vocabulary': 'df_3'}
    named_outputs = {'output data': 'output_1', 'model': 'output_1_model'}

    name = "als"
    set_values = []

    instance = CollaborativeOperation(params, named_inputs=named_inputs,
                                      named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=named_outputs['output data'],
        name=name))

    settings = (['{0}.set{1}({2})'.format(named_outputs['output data'], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_als_operation_success():
    params = {
        AlternatingLeastSquaresOperation.RANK_PARAM: 10,
        AlternatingLeastSquaresOperation.MAX_ITER_PARAM: 10,
        AlternatingLeastSquaresOperation.USER_COL_PARAM: 'u',
        AlternatingLeastSquaresOperation.ITEM_COL_PARAM: 'm',
        AlternatingLeastSquaresOperation.RATING_COL_PARAM: 'r',
        AlternatingLeastSquaresOperation.REG_PARAM: 0.1,
        AlternatingLeastSquaresOperation.IMPLICIT_PREFS_PARAM: False,

        # Could be required
        # AlternatingLeastSquaresOperation.ALPHA_PARAM:'alpha',
        # AlternatingLeastSquaresOperation.SEED_PARAM:'seed',
        # AlternatingLeastSquaresOperation.NUM_USER_BLOCKS_PARAM:'numUserBlocks',
        # AlternatingLeastSquaresOperation.NUM_ITEM_BLOCKS_PARAM:'numItemBlocks',
    }

    named_inputs = {'algorithm': 'df_1',
                    'input data': 'df_2'}
    named_outputs = {'algorithm': 'algorithm_als'}

    instance = AlternatingLeastSquaresOperation(params,
                                                named_inputs=named_inputs,
                                                named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
                # Build the recommendation model using ALS on the training data
                {output} = ALS(maxIter={maxIter}, regParam={regParam},
                        userCol='{userCol}', itemCol='{itemCol}',
                        ratingCol='{ratingCol}',
                        coldStartStrategy='drop')
                """.format(
        output=named_outputs['algorithm'],
        input=named_inputs['input data'],
        maxIter=params[AlternatingLeastSquaresOperation.MAX_ITER_PARAM],
        regParam=params[AlternatingLeastSquaresOperation.REG_PARAM],
        userCol=params[AlternatingLeastSquaresOperation.USER_COL_PARAM],
        itemCol=params[AlternatingLeastSquaresOperation.ITEM_COL_PARAM],
        ratingCol=params[AlternatingLeastSquaresOperation.RATING_COL_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)
