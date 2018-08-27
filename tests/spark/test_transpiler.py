# -*- coding: utf-8 -*-
from io import StringIO

import mock
import pytest
from juicer.operation import Operation
from juicer.spark.transpiler import SparkTranspiler, TranspilerUtils
from juicer.workflow.workflow import Workflow


@pytest.fixture
def basic_wf_fixture():
    workflow = {
        "id": 11,
        "name": "Sort titanic data by passenger age",
        "description": None,
        "enabled": True,
        "created": "2016-12-07T16:53:06+00:00",
        "updated": "2016-12-12T16:20:03+00:00",
        "version": 2,
        "tasks": [
            {
                "id": "922ca54d-98d3-4e2f-a008-b02c93ca823f",
                "left": 127,
                "top": 49,
                "z_index": 11,
                "forms": {
                    "header": {
                        "category": "Execution",
                        "value": "1"
                    },
                    "data_source": {
                        "category": "Execution",
                        "value": "4"
                    }
                },
                "operation": {
                    "id": 18,
                    "name": "Data reader",
                    "slug": "data-reader"
                }
            },
            {
                "id": "9dad14aa-c191-4d60-8045-6623af29ffc9",
                "left": 126,
                "top": 158,
                "z_index": 12,
                "forms": {
                    "attributes": {
                        "category": "Execution",
                        "value": [
                            {
                                "attribute": "age",
                                "alias": "",
                                "f": "asc"
                            }
                        ]
                    },
                },
                "operation": {
                    "id": 32,
                    "name": "Sort",
                    "slug": "sort"
                }
            },
            {
                "id": "dd99334c-954d-4881-a711-e71e80c25b91",
                "left": 127,
                "top": 268,
                "z_index": 13,
                "forms": {
                    "name": {
                        "category": "Execution",
                        "value": "titanic_sorted"
                    },
                    "format": {
                        "category": "Execution",
                        "value": "CSV"
                    },
                    "storage": {
                        "category": "Execution",
                        "value": "1"
                    },
                    "mode": {
                        "category": "Execution",
                        "value": "overwrite"
                    },
                    "path": {
                        "category": "Execution",
                        "value": "/walter/lixo"
                    }
                },
                "operation": {
                    "id": 30,
                    "name": "Data writer",
                    "slug": "data-writer"
                }
            }
        ],
        "flows": [
            {
                "source_port": 35,
                "target_port": 61,
                "source_port_name": "output data",
                "target_port_name": "input data",
                "source_id": "922ca54d-98d3-4e2f-a008-b02c93ca823f",
                "target_id": "9dad14aa-c191-4d60-8045-6623af29ffc9"
            }
        ],
        "platform": {
            "id": 1,
            "name": "Spark",
            "slug": "spark",
            "description": "Apache Spark 2.0 execution platform",
            "icon": "/static/spark.png"
        },
        "user": {
            "login": "admin",
            "id": 0,
            "name": "admin"
        }
    }
    return workflow


@pytest.fixture
def workflow_with_disabled_tasks_fixture():
    workflow = basic_wf_fixture()

    for i, t in enumerate(workflow['tasks']):
        t['enabled'] = (i % 2) == 0
    return workflow


# noinspection PyShadowingNames,PyProtectedMember
def test_transpiler_utils__get_enabled_tasks_to_execute_success(
        workflow_with_disabled_tasks_fixture):
    workflow = workflow_with_disabled_tasks_fixture
    transpiler = SparkTranspiler({})

    # Mock in order to do not read config file
    with mock.patch(
            'juicer.workflow.workflow.Workflow._build_initial_workflow_graph'):
        loader = Workflow(workflow, config={})
        instances = []
        total_enabled = 0
        for i, task in enumerate(loader.workflow['tasks']):
            parameters = dict(
                [(k, v['value']) for k, v in task['forms'].items()])
            parameters['task'] = task
            class_name = transpiler.operations[task['operation']['slug']]
            instance = class_name(parameters, {}, {})
            # Force enabled, because flows are not set
            if (i % 2) == 0:
                instance.has_code = True
                total_enabled += 1
            instances.append(instance)
        assert len(
            TranspilerUtils._get_enabled_tasks_to_execute(
                instances)) == total_enabled


# noinspection PyUnusedLocal, PyShadowingNames
def test_transpiler_basic_flow_success(basic_wf_fixture):
    workflow = basic_wf_fixture
    with mock.patch(
            'juicer.workflow.workflow.Workflow._build_initial_workflow_graph') \
            as mocked_fn:
        mocked_fn.side_effect = lambda: ""
        out = StringIO()
        loader = Workflow(workflow, config={})
        transpiler = SparkTranspiler({})

        class FakeOp(Operation):
            name = u'# Fake'

            def generate_code(self):
                x = self.get_output_names('|').split('|')
                return u'{} = {}'.format(self.get_output_names(),
                                         ', '.join(['None' for _ in x]))

        # import pdb
        # pdb.set_trace()
        for op in transpiler.operations:
            transpiler.operations[op] = FakeOp

        transpiler.transpile(loader.workflow, loader.graph, {}, out=out)
        out.seek(0)
        # print out.read()
