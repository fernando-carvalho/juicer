# -*- coding: utf-8 -*-
import ast
from textwrap import dedent

import pytest
from juicer.spark.etl_operation import SplitOperation, SortOperation, \
    RemoveDuplicatedOperation, \
    SampleOrPartitionOperation, AddRowsOperation, IntersectionOperation, \
    DifferenceOperation, \
    JoinOperation, DropOperation, \
    TransformationOperation, SelectOperation, AggregationOperation, \
    FilterOperation, \
    CleanMissingOperation, \
    AddColumnsOperation
from tests import compare_ast, format_code_comparison, debug_ast


def test_add_columns_minimum_params_success():
    params = {}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}

    instance = AddColumnsOperation(parameters=params,
                                   named_inputs=n_in,
                                   named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
    def _add_column_index(df, prefix):
        # Create new attribute names
        old_attrs = ['{{}}{{}}'.format(prefix, name)
            for name in df.schema.names]
        new_attrs = old_attrs + ['_inx']

        # Add attribute index
        return df.rdd.zipWithIndex().map(
            lambda (row, inx): row + (inx,)).toDF(new_attrs)

    input1_indexed = _add_column_index({in0}, 'ds0_')
    input2_indexed = _add_column_index({in1}, 'ds1_')

    out = input1_indexed.join(input2_indexed,
       input1_indexed._inx == input2_indexed._inx, 'inner').drop(
            input1_indexed._inx).drop(input2_indexed._inx)
    """.format(
        out=n_out['output data'],
        in0=n_in['input data 1'],
        in1=n_in['input data 2']))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_add_rows_minimal_params_success():
    params = {}

    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}

    instance = AddRowsOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = {in0}.unionAll({in1})".format(
        out=n_out['output data'], in0=n_in['input data 1'],
        in1=n_in['input data 2'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_aggregation_rows_minimal_params_success():
    params = {
        AggregationOperation.FUNCTION_PARAM: [
            {'attribute': 'income', 'f': 'AVG', 'alias': 'avg_income'}],
        AggregationOperation.ATTRIBUTES_PARAM: ['country']
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = AggregationOperation(params, named_inputs=n_in,
                                    named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent("""
         pivot_values = None
         pivot_attr = ''
         if pivot_attr:
              {out} = {in0}.groupBy(
                 functions.col('{agg}')).pivot(
                     pivot_attr, pivot_values).agg(
                         functions.avg('income').alias('avg_income'))
         else:
              {out} = {in0}.groupBy(
                 functions.col('{agg}')).agg(
                     functions.avg('income').alias('avg_income'))

        """.format(out=n_out['output data'], in0=n_in['input data'],
                   agg='country', ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_aggregation_rows_group_all_missing_attributes_success():
    params = {
        AggregationOperation.FUNCTION_PARAM: [
            {'attribute': 'income', 'f': 'AVG', 'alias': 'avg_income'}],
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = AggregationOperation(params, named_inputs=n_in,
                                    named_outputs=n_out)
    code = instance.generate_code()

    expected_code = """{out} = {in0}.agg(
                        functions.avg('income').alias('avg_income'))""".format(
        out=n_out['output data'], in0=n_in['input data'], agg='country', )
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_aggregation_missing_function_param_failure():
    params = {
        AggregationOperation.ATTRIBUTES_PARAM: ['country']
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    with pytest.raises(ValueError):
        AggregationOperation(params, named_inputs=n_in,
                             named_outputs=n_out)


def test_clean_missing_minimal_params_success():
    params = {
        CleanMissingOperation.ATTRIBUTES_PARAM: ['name'],
        CleanMissingOperation.MIN_MISSING_RATIO_PARAM: "0.0",
        CleanMissingOperation.MAX_MISSING_RATIO_PARAM: "1.0",
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output result': 'output_1'}
    instance = CleanMissingOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
    ratio_{input_1} = {input_1}.select(
        (functions.avg(functions.col('{attribute}').isNull().cast(
        'int'))).alias('{attribute}')).collect()
    attributes_{input_1} = [c for c in ["{attribute}"]
                 if 0.0 <= ratio_{input_1}[0][c] <= 1.0]
    if len(attributes_input_1) > 0:
        {output_1} = {input_1}.na.drop(how='any', subset=attributes_{input_1})
    else:
        {output_1} = {input_1}
    """.format(input_1=n_in['input data'], attribute=params['attributes'][0],
               output_1=n_out['output result']))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_clean_missing_without_missing_rating_params_success():
    params = {
        CleanMissingOperation.ATTRIBUTES_PARAM: ['name'],
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output result': 'output_1'}
    instance = CleanMissingOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
    attributes_{input_1} = ['{attribute}']
    if len(attributes_input_1) > 0:
        {output_1} = {input_1}.na.drop(how='any', subset=attributes_{input_1})
    else:
        {output_1} = {input_1}
    """.format(input_1=n_in['input data'], attribute=params['attributes'][0],
               output_1=n_out['output result']))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_clean_missing_minimal_params_type_value_success():
    params = {
        CleanMissingOperation.ATTRIBUTES_PARAM: ['name'],
        CleanMissingOperation.MIN_MISSING_RATIO_PARAM: "0.0",
        CleanMissingOperation.MAX_MISSING_RATIO_PARAM: "1.0",
        CleanMissingOperation.VALUE_PARAMETER: "200",
        CleanMissingOperation.CLEANING_MODE_PARAM: CleanMissingOperation.VALUE
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output result': 'output_1'}
    instance = CleanMissingOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
    ratio_{input_1} = {input_1}.select(
        (functions.avg(functions.col('{attribute}').isNull().cast(
        'int'))).alias('{attribute}')).collect()
    attributes_{input_1} = [c for c in ["{attribute}"]
                 if 0.0 <= ratio_{input_1}[0][c] <= 1.0]
    if len(attributes_input_1) > 0:
        {output_1} = {input_1}.na.fill(value={value},
                subset=attributes_{input_1})
    else:
        {output_1} = {input_1}
    """.format(input_1=n_in['input data'], attribute=params['attributes'][0],
               output_1=n_out['output result'],
               value=params[CleanMissingOperation.VALUE_PARAMETER]))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)

    # Test with value being number
    params[CleanMissingOperation.VALUE_PARAMETER] = 1200
    instance = CleanMissingOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)
    code = instance.generate_code()
    expected_code = expected_code.replace('200', '1200')
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_clean_missing_missing_attribute_param_failure():
    params = {}
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        CleanMissingOperation(params, named_inputs=n_in,
                              named_outputs=n_out)


def test_difference_minimal_params_success():
    params = {}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    class_name = DifferenceOperation
    instance = class_name(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = "{out} = {in1}.subtract({in2})".format(
        out=n_out['output data'], in1=n_in['input data 1'],
        in2=n_in['input data 2'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_remove_duplicated_minimal_params_success():
    params = {}
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = RemoveDuplicatedOperation(params, named_inputs=n_in,
                                         named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = {input}.dropDuplicates()".format(
        out=n_out['output data'], input=n_in['input data']
    )
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_remove_duplicated_by_attributes_success():
    params = {
        'attributes': ['name']
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = RemoveDuplicatedOperation(params, named_inputs=n_in,
                                         named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = {input}.dropDuplicates(subset=['name'])".format(
        out=n_out['output data'], input=n_in['input data']
    )
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_drop_minimal_params_success():
    params = {
        'column': 'TEST'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = DropOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = {in1}.drop('{drop}')".format(
        out=n_out['output data'], in1=n_in['input data'], drop=params['column'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_filter_minimum_params_success():
    params = {
        FilterOperation.FILTER_PARAM: [{
            'attribute': 'code',
            'f': '>',
            'value': '201'
        }],
        'config': {

        }
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    instance = FilterOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = ("{out} = {in1}.filter("
                     "functions.col('{attribute}') {f} '{value}')").format(
        out=n_out['output data'], in1=n_in['input data'],
        **params[FilterOperation.FILTER_PARAM][0])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_filter_missing_parameter_filter_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        FilterOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_intersection_minimal_params_success():
    params = {}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    instance = IntersectionOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent(
        """
        if len(df1.columns) != len(df2.columns):
            raise ValueError('{error}')
        {out} = {in1}.intersect({in2})
        """.format(
            out=n_out['output data'], in1=n_in['input data 1'],
            in2=n_in['input data 2'],
            error=(
                'For intersection operation, both input data '
                'sources must have the same number of attributes '
                'and types.')))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_join_inner_join_minimal_params_success():
    params = {
        'left_attributes': ['id', 'cod'],
        'right_attributes': ['id', 'cod'],
        'aliases': 'left_, right_  '
    }
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    instance = JoinOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        def _rename_attributes(df, prefix):
            result = df
            for col in df.columns:
                result = result.withColumnRenamed(col, '{{}}{{}}'.format(
                    prefix, col))
            return result

        in0_renamed = _rename_attributes({in0}, '{a0}')
        in1_renamed = _rename_attributes({in1}, '{a1}')

        condition = [in0_renamed['{a0}id'] == in1_renamed['{a1}id'],
            in0_renamed['{a0}cod'] == in1_renamed['{a1}cod']]

        {out} = in0_renamed.join(in1_renamed, on=condition, how='{how}').drop(
            in1_renamed['{a1}id']).drop(in1_renamed['{a1}cod'])""".format(
        out=n_out['output data'], in0=n_in['input data 1'],
        a0='left_', a1='right_',
        in1=n_in['input data 2'], how="inner"))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_join_left_join_keep_columns_minimal_params_success():
    params = {
        'left_attributes': ['id', 'cod'],
        'right_attributes': ['id', 'cod'],
        JoinOperation.JOIN_TYPE_PARAM: 'left',
        JoinOperation.KEEP_RIGHT_KEYS_PARAM: True,
        'aliases': 'left_, right_  '
    }
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    instance = JoinOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        def _rename_attributes(df, prefix):
            result = df
            for col in df.columns:
                result = result.withColumnRenamed(col, '{{}}{{}}'.format(
                    prefix, col))
            return result

        in0_renamed = _rename_attributes({in0}, '{a0}')
        in1_renamed = _rename_attributes({in1}, '{a1}')

        condition = [in0_renamed['{a0}id'] == in1_renamed['{a1}id'],
            in0_renamed['{a0}cod'] == in1_renamed['{a1}cod']]
        {out} = in0_renamed.join(in1_renamed, on=condition, how='left')
        """.format(
        out=n_out['output data'], in0=n_in['input data 1'],
        a0='left_', a1='right_',
        in1=n_in['input data 2'], type=params[JoinOperation.JOIN_TYPE_PARAM], ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_join_remove_right_columns_success():
    params = {
        'left_attributes': ['id', 'cod'],
        'right_attributes': ['id2', 'cod2'],
        JoinOperation.KEEP_RIGHT_KEYS_PARAM: 'False',
        'aliases': 'left_, right_  '
    }
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    instance = JoinOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        def _rename_attributes(df, prefix):
            result = df
            for col in df.columns:
                result = result.withColumnRenamed(col, '{{}}{{}}'.format(
                    prefix, col))
            return result
        in0_renamed = _rename_attributes({in0}, '{a0}')
        in1_renamed = _rename_attributes({in1}, '{a1}')

        condition = [in0_renamed['{a0}id'] == in1_renamed['{a1}id2'],
            in0_renamed['{a0}cod'] == in1_renamed['{a1}cod2']]
        {out} = in0_renamed.join(in1_renamed, on=condition, how='inner')\\
          .drop(in1_renamed['{a1}id2']).drop(in1_renamed['{a1}cod2'])""".format(
        out=n_out['output data'], in0=n_in['input data 1'],
        in1=n_in['input data 2'], a0='left_', a1='right_'))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_join_missing_left_or_right_param_failure():
    params = {
        'right_attributes': ['id', 'cod']
    }
    with pytest.raises(ValueError):
        n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
        n_out = {'output data': 'out'}
        JoinOperation(params, named_inputs=n_in, named_outputs=n_out)

    params = {
        'left_attributes': ['id', 'cod']
    }
    with pytest.raises(ValueError):
        JoinOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_random_split_minimal_params_success():
    params = {
        'weights': '40',
        'seed': '1234321'
    }
    n_in = {'input data': 'df1'}
    n_out = {'splitted data 1': 'out1', 'splitted data 2': 'out2'}

    instance = SplitOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out0}, {out1} = {input}.randomSplit({weights}, {seed})" \
        .format(out0=n_out['splitted data 1'], out1=n_out['splitted data 2'],
                input=n_in['input data'], weights='[40.0, 60.0]', seed=1234321)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_sample_or_partition_minimal_params_success():
    params = {
        'withReplacement': 'False',
        'fraction': '0.3',
        'seed': '0'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(withReplacement={}, " \
                    "fraction={}, seed={})".format(params['withReplacement'],
                                                   params['fraction'],
                                                   params['seed'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_type_value_success():
    params = {
        'withReplacement': 'False',
        'value': '400',
        'seed': '0',
        'type': SampleOrPartitionOperation.TYPE_VALUE
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = """output_1 = input_1.orderBy(
        functions.rand(0)).limit({})""".format(params['value'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_type_head_success():
    params = {
        'withReplacement': 'False',
        'value': '365',
        'seed': '0',
        'type': SampleOrPartitionOperation.TYPE_HEAD
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = """output_1 = input_1.limit({})""".format(params['value'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    # print "\n\n", code, '\n\n', expected_code, '\n\n'
    assert result, msg


def test_sample_or_partition_invalid_fraction_failure():
    params = {
        'withReplacement': 'False',
        'fraction': '101',
        'seed': '0'
    }

    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SampleOrPartitionOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)


def test_sample_or_partition_fraction_percentage_success():
    params = {
        'withReplacement': 'False',
        'fraction': 45,
        'seed': '0'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(withReplacement={}, " \
                    "fraction={}, seed={})".format(params['withReplacement'],
                                                   params['fraction'] * 0.01,
                                                   params['seed'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_fraction_missing_failure():
    params = {
        'withReplacement': 'False',
        'seed': '0'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SampleOrPartitionOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)


def test_select_minimal_params_success():
    params = {
        SelectOperation.ATTRIBUTES_PARAM: ['name', 'class']
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output projected data': 'output_1'}
    instance = SelectOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    select = ', '.join(
        ['"{}"'.format(x) for x in params[SelectOperation.ATTRIBUTES_PARAM]])
    expected_code = '{out} = {in1}.select({select})'.format(
        out=n_out['output projected data'], in1=n_in['input data'],
        select=select)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_select_missing_attribute_param_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SelectOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_sort_minimal_params_success():
    params = {
        'attributes': [{'attribute': 'name', 'f': 'asc'},
                       {'attribute': 'class', 'f': 'desc'}],
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = SortOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = '{out} = {input}.orderBy(["name", "class"], ' \
                    'ascending=[1, 0])'.format(out=n_out['output data'],
                                               input=n_in['input data'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sort_missing_attributes_failure():
    params = {}
    with pytest.raises(ValueError):
        n_in = {'input data': 'df1'}
        n_out = {'output data': 'out'}
        SortOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_transformation_minumum_params_success():
    alias = 'result_1'
    expr = [{'tree': {
        "type": "CallExpression",
        "arguments": [
            {
                "type": "Literal",
                "value": "attr_name",
                "raw": "'attr_name'"
            }
        ],
        "callee": {
            "type": "Identifier",
            "name": "lower"
        },
    }, 'alias': alias, 'expression': "lower(attr_name)"}]
    params = {
        TransformationOperation.EXPRESSION_PARAM: expr,
        'input': 'input_x',
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = TransformationOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent(
        """
        from juicer.spark.ext import CustomExpressionTransformer
        expr_alias = [
            [functions.lower('attr_name'), '{alias}']
        ]
        tmp_out = {in1}
        for expr, alias in expr_alias:
            transformer = CustomExpressionTransformer(
                outputCol=alias, expression=expr)
            tmp_out = transformer.transform(tmp_out)
        {out} = tmp_out
        """)

    expected_code = expected_code.format(
        out=n_out['output data'], in1=n_in['input data'], alias=alias)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_transformation_math_expression_success():
    alias = 'result_2'
    expr = [{'tree': {
        "type": "BinaryExpression",
        "operator": "*",
        "left": {
            "type": "Identifier",
            "name": "a"
        },
        "right": {
            "type": "Literal",
            "value": 100,
            "raw": "100"
        }
    }, 'alias': alias, 'expression': "lower(a)"}]

    params = {
        TransformationOperation.EXPRESSION_PARAM: expr,
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = TransformationOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent(
        """
        from juicer.spark.ext import CustomExpressionTransformer
        expr_alias = [
            [{in1}['a'] * 100, '{alias}']
        ]
        tmp_out = {in1}
        for expr, alias in expr_alias:
            transformer = CustomExpressionTransformer(
                outputCol=alias, expression=expr)
            tmp_out = transformer.transform(tmp_out)
        {out} = tmp_out
        """)

    expected_code = expected_code.format(
        out=n_out['output data'], in1=n_in['input data'], alias=alias)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_transformation_complex_expression_success():
    alias = 'result_3'
    expr = [{'tree': {
        "type": "BinaryExpression",
        "operator": "+",
        "left": {
            "type": "UnaryExpression",
            "operator": "-",
            "argument": {
                "type": "Identifier",
                "name": "a"
            },
            "prefix": True
        },
        "right": {
            "type": "Identifier",
            "name": "b"
        }
    }, 'alias': alias, 'expression': "a + b "}]

    params = {
        TransformationOperation.EXPRESSION_PARAM: expr,
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = TransformationOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent(
        """
        from juicer.spark.ext import CustomExpressionTransformer
        expr_alias = [
            [(- {in1}['a'] + {in1}['b']), '{alias}']
        ]
        tmp_out = {in1}
        for expr, alias in expr_alias:
            transformer = CustomExpressionTransformer(
                outputCol=alias, expression=expr)
            tmp_out = transformer.transform(tmp_out)
        {out} = tmp_out
        """)

    expected_code = expected_code.format(
        out=n_out['output data'], in1=n_in['input data'], alias=alias)

    debug_ast(code, expected_code)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_transformation_missing_expr_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'df1'}
        n_out = {'output data': 'out'}
        TransformationOperation(params, named_inputs=n_in,
                                named_outputs=n_out)
