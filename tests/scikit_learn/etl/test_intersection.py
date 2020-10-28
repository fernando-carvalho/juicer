from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import IntersectionOperation
import pytest
import pandas as pd
# Intersection
# 
def test_intersection_success():
    slice_size1 = 10
    slice_size2 = 20
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
        'petallength','petalwidth'], slice_size1)]
    df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                            'petallength','petalwidth'], slice_size2)]

    expectedkeys=df1[1].columns.tolist()
    expectedresult=pd.merge(df1[1],df2[1],how='inner',on=expectedkeys,indicator=False,copy=False)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IntersectionOperation(**arguments)
    print(instance.generate_code())
    result = util.execute(instance.generate_code(),
                          {'df1':df1[1],'df2':df2[1]})
    #print(expectedresult)
    print(result['out'])
    assert result['out'].equals(expectedresult)



def test_intersection_fail_different_colunms():
    slice_size1 = 10
    slice_size2 = 20
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size1)]
    df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'class'], slice_size2)]

    # expectedkeys=df1[1].columns.tolist()
    # expectedresult=pd.merge(df1[1],df2[1],how='inner',on=expectedkeys,indicator=False,copy=False)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]

        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IntersectionOperation(**arguments)
    # print(instance.generate_code())
    with pytest.raises(KeyError) as exc_info:
        result = util.execute(instance.generate_code(),
                          {'df1': df1[1], 'df2': df2[1]})
    print(exc_info.value)
    # print(expectedresult)
    # print(result['out'])
    assert 'petalwidth' in str(exc_info.value)



def test_3_input():
    slice_size1 = 10
    slice_size2 = 20
    slice_size3 = 15
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size1)]
    df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size2)]
    df3 = ['df3', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size3)]
    #expectedkeys = df1[1].columns.tolist()
    #expectedresult = pd.merge(df1[1], df2[1], how='inner', on=expectedkeys, indicator=False, copy=False)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0],
            'input data 3': df3[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as exc_info:
        instance = IntersectionOperation(**arguments)
    # print(instance.generate_code())
    #result = util.execute(instance.generate_code(),
    #                      {'df1': df1[1], 'df2': df2[1],'df3': df3[1]})
    print(exc_info.value)
    # print(expectedresult)
    # print(result['out'])
    assert 'Parameter \'input data 1\' and \'input data 2\' must be informed for task' in str(exc_info.value)



def test_1_input():
    slice_size1 = 10
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size1)]
    #expectedkeys = df1[1].columns.tolist()
    #expectedresult = pd.merge(df1[1], df2[1], how='inner', on=expectedkeys, indicator=False, copy=False)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as exc_info:
        instance = IntersectionOperation(**arguments)
    # print(instance.generate_code())
    #result = util.execute(instance.generate_code(),
    #                      {'df1': df1[1], 'df2': df2[1],'df3': df3[1]})
    print(exc_info.value)
    # print(expectedresult)
    # print(result['out'])
    assert 'Parameter \'input data 1\' and \'input data 2\' must be informed for task' in str(exc_info.value)



def test_intersection_fail_different_number_of_colunms():
        slice_size1 = 10
        slice_size2 = 20
        df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                                 'petallength', 'petalwidth'], slice_size1)]
        df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                                 'petallength'], slice_size2)]

        #expectedkeys=df1[1].columns.tolist()
        #expectedresult=pd.merge(df1[1],df2[1],how='inner',on=expectedkeys,indicator=False,copy=False)
        arguments = {
            'parameters': {},
            'named_inputs': {
                'input data 1': df1[0],
                'input data 2': df2[0]
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        instance = IntersectionOperation(**arguments)
        # print(instance.generate_code())
        with pytest.raises(ValueError) as exc_info:
            result = util.execute(instance.generate_code(),
                             {'df1': df1[1], 'df2': df2[1]})
        print(exc_info.value)
        #print(expectedresult)
        #print(result['out'])
        assert 'For intersection operation, both input data sources must have the same number of attributes and types.' in str(exc_info.value)



def test_interssection_fail_input_data_1_empty():
    slice_size1 = 0
    slice_size2 = 20
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size1)]
    df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size2)]

    expectedkeys = df1[1].columns.tolist()
    expectedresult = pd.merge(df1[1], df2[1], how='inner', on=expectedkeys, indicator=False, copy=False)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IntersectionOperation(**arguments)
    #print(instance.generate_code())
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1], 'df2': df2[1]})
    #print(expectedresult)
    #print(result['out'])
    assert result['out'].equals(expectedresult)



def test_interssection_fail_input_data_2_empty():
    slice_size1 = 10
    slice_size2 = 0
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size1)]
    df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size2)]

    expectedkeys = df1[1].columns.tolist()
    expectedresult = pd.merge(df1[1], df2[1], how='inner', on=expectedkeys, indicator=False, copy=False)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IntersectionOperation(**arguments)
    #print(instance.generate_code())
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1], 'df2': df2[1]})
    #print(expectedresult)
    #print(result['out'])
    assert result['out'].equals(expectedresult)



def test_intersection_success_missing_input_inplies_no_code():
    #deve dar sucessso porem sem gerar codigo, has_code() deve retornar False
    slice_size1 = 10
    slice_size2 = 20
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size1)]
    df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size2)]

    arguments = {
        'parameters': {},
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as exc_info:
        instance = IntersectionOperation(**arguments)
    print(exc_info.value)
    assert 'Parameter \'input data 1\' and \'input data 2\' must be informed for task' in str(exc_info.value)



def test_intersection_success_missing_output_inplies_no_code():
    #deve dar sucessso porem sem gerar codigo
    slice_size1 = 10
    slice_size2 = 20
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size1)]
    df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth'], slice_size2)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
        }
    }
    with pytest.raises(ValueError) as exc_info:
        instance = IntersectionOperation(**arguments)
    """ 
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    assert 'Parameter \'input data 1\' and \'input data 2\' must be informed for task ' in str(exc_info.value)
