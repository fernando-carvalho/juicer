# -*- coding: utf-8 -*-
import pytest
from juicer.spark.operation import Operation


def test_base_operation_generate_code_failure():
    parameters = {
        'data_source': 1
    }
    with pytest.raises(NotImplementedError):
        instance = Operation(parameters, inputs=[], outputs=['output_1'],
                             named_inputs={}, named_outputs={})
        instance.generate_code()
