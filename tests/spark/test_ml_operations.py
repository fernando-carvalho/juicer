# -*- coding: utf-8 -*-
import ast
import json
from itertools import izip_longest
from textwrap import dedent

import pytest
import difflib
# Import Operations to test
from juicer.runner import configuration
from juicer.spark.ml_operation import FeatureIndexerOperation, \
    FeatureAssemblerOperation, \
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
        col_alias = dict({alias})
        indexers = [feature.StringIndexer(inputCol=col, outputCol=alias,
                            handleInvalid='skip')
                    for col, alias in col_alias.items()]

        # Use Pipeline to process all attributes once
        pipeline = Pipeline(stages=indexers)
        models_task_1 = dict([(c, indexers[i].fit({in1}))
                  for i, c in enumerate(col_alias)])

        # labels = [model.labels for model in models.itervalues()]
        # Spark ML 2.0.1 do not deal with null in indexer.
        # See SPARK-11569

        # input_1_without_null = input_1.na.fill('NA', subset=col_alias.keys())
        {in1}_without_null = {in1}.na.fill('NA', subset=col_alias.keys())
        {out} = pipeline.fit({in1}_without_null).transform({in1}_without_null)

        """.format(attr=params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
                   in1=in1,
                   out=out,
                   alias=json.dumps(
                       zip(params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
                           params[FeatureIndexerOperation.ALIAS_PARAM]))))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg


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
            models = dict([(col, indexers[i].fit({1})) for i, col in
                        enumerate(col_alias)])
            labels = None

            # Spark ML 2.0.1 do not deal with null in indexer.
            # See SPARK-11569
            {1}_without_null = {1}.na.fill('NA', subset=col_alias.keys())

            {2} = pipeline.fit({1}_without_null).transform({1}_without_null)
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

    expected_code = "{output_1} = {model}.transform({input_1})".format(
        output_1=out, input_1=in1, model=model)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


'''
    EvaluateModel tests
'''


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
            {evaluator_out} = {evaluator}({predic_col}='{predic_atr}',
                                  labelCol='{label_atr}', metricName='{metric}')

            {output} = {evaluator_out}.evaluate({input_1})

            display_text = False
            if display_text:
                from juicer.spark.reports import SimpleTableReport
                headers = ['Parameter', 'Description', 'Value', 'Default']
                rows = [
                        [x.name, x.doc,
                            df_evaluator._paramMap.get(x, 'unset'),
                             df_evaluator._defaultParamMap.get(
                                 x, 'unset')] for x in
                    df_evaluator.extractParamMap()]

                content = SimpleTableReport(
                        'table table-striped table-bordered', headers, rows)

                result = '<h4>{{}}: {{}}</h4>'.format('f1',
                    metric_value)

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=result + content.generate(),
                    type='HTML', title='Evaluation result',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})

            from juicer.spark.ml_operation import ModelsEvaluationResultList
            model_task_1 = ModelsEvaluationResultList(
                [df_model], df_model, 'f1', metric_value)
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
    assert result, msg + format_code_comparison(code, expected_code) #+ '\n' + '\n'.join(difflib.unified_diff(code.split('\n'), expected_code.split('\n')))


    # @!BUG - Acessing 'task''order' in parameter attribute, but doesn't exist
    # def test_evaluate_model_operation_missing_output_param_failure():
    #     params = {
    #
    #         EvaluateModel.PREDICTION_ATTRIBUTE_PARAM: 'c',
    #         EvaluateModel.LABEL_ATTRIBUTE_PARAM: 'c',
    #         EvaluateModel.METRIC_PARAM: 'f1',
    #     }
    #     inputs = ['input_1', 'input_2']
    #     outputs = []
    #     with pytest.raises(ValueError):
    #         evaluator = EvaluateModel(params, inputs,
    #                                   outputs, named_inputs={},
    #                                   named_outputs={})
    #         evaluator.generate_code()


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

    }
    n_in = {'algorithm': 'df_1', 'input data': 'df_2', 'evaluator': 'xpto'}
    n_out = {'scored data': 'output_1', 'evaluation': 'eval_1'}
    outputs = ['output_1']

    instance = CrossValidationOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
            grid_builder = tuning.ParamGridBuilder()
            estimator, param_grid = {algorithm}

            for param_name, values in param_grid.items():
                param = getattr(estimator, param_name)
                grid_builder.addGrid(param, values)

            evaluator = {evaluator}

            cross_validator = tuning.CrossValidator(
                estimator=estimator, estimatorParamMaps=grid_builder.build(),
                evaluator=evaluator, numFolds={folds})
            cv_model = cross_validator.fit({input_data})
            fit_data = cv_model.transform({input_data})
            best_model_{output}  = cv_model.bestModel
            metric_result = evaluator.evaluate(fit_data)
            {evaluation} = metric_result
            {output} = fit_data
            models_task_1 = None

            grouped_result = fit_data.select(
                 evaluator.getLabelCol(), evaluator.getPredictionCol())\
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
                identifier='232', message='Result generated',
                type='TEXT', title='Evaluation result',
                task={{'id': '232' }},
                operation={{'id': 1 }},
                operation_id=1,
                content=json.dumps(eval_{output}))

            """.format(algorithm=n_in['algorithm'],
                       input_data=n_in['input data'],
                       evaluator=n_in['evaluator'],
                       evaluation='eval_1',
                       output=outputs[0],
                       folds=params[CrossValidationOperation.NUM_FOLDS_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_cross_validation_complete_operation_success():
    params = {

        CrossValidationOperation.NUM_FOLDS_PARAM: 3,
        'task_id': '2323-afffa-343bdaff',
        'operation_id': 2793

    }
    n_in = {'algorithm': 'algo1', 'input data': 'df_1', 'evaluator': 'ev_1'}
    n_out = {'evaluation': 'output_1', 'scored data': 'output_1'}
    outputs = ['output_1']

    instance = CrossValidationOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
            grid_builder = tuning.ParamGridBuilder()
            estimator, param_grid = {algorithm}

            for param_name, values in param_grid.items():
                param = getattr(estimator, param_name)
                grid_builder.addGrid(param, values)

            evaluator = {evaluator}

            cross_validator = tuning.CrossValidator(
                estimator=estimator, estimatorParamMaps=grid_builder.build(),
                evaluator=evaluator, numFolds={folds})
            cv_model = cross_validator.fit({input_data})
            fit_data = cv_model.transform({input_data})
            best_model_{output}  = cv_model.bestModel
            metric_result = evaluator.evaluate(fit_data)
            {output} = metric_result
            {output} = fit_data
            models_task_1 = None
            """.format(algorithm=n_in['algorithm'],
                       input_data=n_in['input data'],
                       evaluator=n_in['evaluator'],
                       output=outputs[0],
                       folds=params[CrossValidationOperation.NUM_FOLDS_PARAM]))

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
                type='TEXT', title='Evaluation result',
                task={{'id': '{task_id}' }},
                operation={{'id': {operation_id} }},
                operation_id={operation_id},
                content=json.dumps(eval_{output}))

            """.format(output=outputs[0],
                       task_id=params['task_id'],
                       operation_id=params['operation_id'])
    expected_code = '\n'.join([expected_code, dedent(eval_code)])

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_cross_validation_complete_operation_missing_input_failure():
    params = {

        CrossValidationOperation.NUM_FOLDS_PARAM: 3,

    }
    n_in = {'algorithm': 'algo1', 'evaluator': 'ev_1'}
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
        ClassificationModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'prediction1'

    }
    n_in = {'algorithm': 'algo_param', 'train input data': 'train'}
    n_out = {'model': 'model_1'}
    instance = ClassificationModelOperation(params, named_inputs=n_in,
                                            named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        algorithm, param_grid = {algorithm}
        algorithm.setPredictionCol('{prediction}')
        algorithm.setLabelCol('{label}')
        algorithm.setFeaturesCol('{features}')
        {output} = algorithm.fit({train})
        """.format(output=n_out['model'],
                   train=n_in['train input data'],
                   algorithm=n_in['algorithm'],
                   prediction=params[
                       ClassificationModelOperation.PREDICTION_ATTRIBUTE_PARAM],
                   features=params[
                       ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM],
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
    n_in = {}
    n_out = {}

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

    param_grid = {param_grid}
    # Output result is the classifier and its parameters. Parameters are
    # need in classification model or cross validator.
    {output} = ({name}(), param_grid)
    """.format(output=n_out['algorithm'], name='BaseClassifier',
               param_grid=json.dumps(param_grid)))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_classifier_operation_missing_param_grid_parameter_failure():
    params = {}
    n_out = {'algorithm': 'classifier_1'}

    with pytest.raises(ValueError):
        ClassifierOperation(params, named_inputs={}, named_outputs=n_out)


def test_classifier_operation_missing_label_failure():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
        }

    }
    n_out = {'algorithm': 'classifier_1'}

    with pytest.raises(ValueError):
        ClassifierOperation(params, named_inputs={}, named_outputs=n_out)


def test_classifier_operation_missing_features_failure():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.LABEL_PARAM: 'l'
        }
    }
    n_out = {'algorithm': 'classifier_1'}

    with pytest.raises(ValueError):
        ClassifierOperation(params, named_inputs={}, named_outputs=n_out)


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
    assert instance_svm.name == 'classification.SVM'


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
    assert instance_lr.name == 'classification.LogisticRegression'


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
    assert instance_dt.name == 'classification.DecisionTreeClassifier'


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
    assert instance_dt.name == 'classification.GBTClassifier'


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
    assert instance_nb.name == 'classification.NaiveBayes'


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
    assert instance_nb.name == 'classification.RandomForestClassifier'


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
    assert instance_pct.name == \
           'classification.MultilayerPerceptronClassificationModel'


"""
    Clustering tests
"""


def test_clustering_model_operation_success():
    params = {

        ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',

    }
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    instance = ClusteringModelOperation(params, named_inputs=named_inputs,
                                        named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        {algorithm}.setFeaturesCol('{features}')
        {model} = {algorithm}.fit({input})
        # There is no way to pass which attribute was used in clustering, so
        # this information will be stored in uid (hack).
        {model}.uid += '|{features}'
        {output} = {model}.transform({input})
        """.format(algorithm=named_inputs['algorithm'],
                   input=named_inputs['train input data'],
                   model=named_outputs['model'],
                   output=outputs[0],
                   features=params[
                       ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM]))

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
        ['DocConcentration',
         params[LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM] *
         [(params.get(LdaClusteringOperation.DOC_CONCENTRATION_PARAM,
                      LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM)) / 50.0]],
        ['K', params[LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM]],
        ['MaxIter', params[LdaClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['Optimizer',
         "'{}'".format(params[LdaClusteringOperation.OPTIMIZER_PARAM])],
        ['TopicConcentration',
         params[LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM]]
    ]

    instance = LdaClusteringOperation(params, named_inputs={},
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
        ['Tol', params[KMeansClusteringOperation.TOLERANCE_PARAMETER]],
        # ['InitMode', params[KMeansClusteringOperation.INIT_MODE_PARAMETER]]
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


def test_kmeans_clustering_operation_kmeansdd_type_kmeans_success():
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
        ['Tol', params[KMeansClusteringOperation.TOLERANCE_PARAMETER]],
        # ['InitMode', params[KMeansClusteringOperation.INIT_MODE_PARAMETER]]
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
                        ratingCol='{ratingCol}')

                ##model = als.fit({input})
                #predictions = model.transform(test)

                # Evaluate the model not support YET
                # evaluator = RegressionEvaluator(metricName="rmse",
                #                labelCol={ratingCol},
                #                predictionCol="prediction")

                # rmse = evaluator.evaluate(predictions)
                # print("Root-mean-square error = " + str(rmse))
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
