from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import TransformationOperation
import pytest

# Transformation
# 
def test_transformation_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth', 
        'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': { "expression":
                            {"value":
                                 [
                                     {"expression": "upper(petallength)",
                                      "alias": "",
                                      "tree":
                                         {"type": "CallExpression",
                                          "arguments":
                                             [
                                                 {"type": "Identifier", "name": "coluna"}
                                             ],
                                          "callee":
                                             {"type": "Identifier", "name": "upper"}
                                          },
                                      "error": ''
                                      }
                                 ]
                            }
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = TransformationOperation(**arguments)
    print(instance.generate_code())
    #result = util.execute(instance.generate_code(),
                          #dict([df]))
    #assert result['out'].equals(util.iris(size=slice_size))
