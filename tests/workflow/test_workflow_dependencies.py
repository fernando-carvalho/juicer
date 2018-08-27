# -*- coding: utf-8 -*-
import gzip
import json

import pytest
from juicer.workflow.workflow import Workflow

fake_conf = {
    'juicer': {
        'services': {
            'tahiti': {'url': 'http://tahiti', 'auth_token': 'xpto'},
            'limonero': {'url': 'http://limonero', 'auth_token': 'xpto'},
            'caipirinha': {'url': 'http://caipirinha', 'auth_token': 'xpto'},
            'stand': {'url': 'http://stand', 'auth_token': 'xpto'},
        }
    }
}


def debug_instance(instance_wf):
    print '*' * 20
    print instance_wf.graph.nodes(data=False)
    print '*' * 21
    print instance_wf.graph.edges()
    print '*' * 22
    print instance_wf.graph.is_multigraph()
    print '*' * 23
    print instance_wf.graph.number_of_edges()
    print '*' * 24
    print instance_wf.sorted_tasks
    print '*' * 25
    test = instance_wf.graph.reverse()
    print test.edges()
    print '*' * 26
    print instance_wf.graph.in_degree()
    print instance_wf.check_in_degree_edges()
    print '*' * 27
    print instance_wf.graph.out_degree()
    print instance_wf.check_out_degree_edges()
    print '*' * 28
    x = instance_wf._get_operations()[0]
    print x['ports']

    # print instance_wf.get_ports_from_operation_tasks('')
    # Show image
    # pos = nx.spring_layout(instance_wf.graph)
    # pos = nx.fruchterman_reingold_layout(instance_wf.graph)
    # nx.draw(instance_wf.graph, pos, node_color='#004a7b', node_size=2000,
    #         edge_color='#555555', width=1.5, edge_cmap=None,
    #         with_labels=True, style='dashed',
    #         label_pos=50.3, alpha=1, arrows=True, node_shape='s',
    #         font_size=8,
    #         font_color='#FFFFFF')
    # plt.show()
    # plt.savefig(filename, dpi=300, orientation='landscape', format=None,
    # bbox_inches=None, pad_inches=0.1)


def fake_query_operations(operations):
    def q():
        return operations

    return q


def fake_query_data_sources(data_sources):
    def q():
        for v in data_sources:
            yield v

    return q


@pytest.mark.skip(reason="Not working")
def test_workflow_sequence_success():
    # Complete workflow
    fixture = "./tests/workflow/fixtures/"
    with open(fixture + 'workflow_correct_changedid.txt') as f:
        workflow_test = json.load(f, encoding='utf-8')

    with gzip.open(fixture + 'operations.json.gz') as g:
        ops = json.load(g, encoding='utf-8')

    instance_wf = Workflow(
        workflow_test, fake_conf,
        query_operations=fake_query_operations(ops),
        query_data_sources=fake_query_data_sources(
            [
                {"id": 1, 'attributes': []}
            ])
    )
    assert instance_wf


@pytest.mark.skip(reason="Not working")
def test_workflow_sequence_missing_targetid_value_failure():
    # workflow with missing target_id
    workflow_test = json.load(
        open("./tests/workflow/fixtures/workflow_missing_targetid.txt"),
        encoding='utf-8')
    with pytest.raises(AttributeError):
        Workflow(workflow_test, fake_conf)


@pytest.mark.skip(reason="Not working")
def test_workflow_sequence_missing_sourceid_value_failure():
    # workflow with missing target_id
    workflow_test = json.load(
        open("./tests/workflow/fixtures/workflow_missing_sourceid.txt"),
        encoding='utf-8')
    with pytest.raises(AttributeError):
        Workflow(workflow_test, fake_conf)


@pytest.mark.skip(reason="Not working")
# @DEPENDS fields of JSON - In Progress
def test_workflow_parcial_execution_success():
    # workflow with missing target_id
    workflow_test = json.load(
        open("./tests/workflow/fixtures/workflow_parcial_execution.txt"),
        # open("./tests/workflow/fixtures/workflow_parcial_execution_tasks.txt"),
        # open("./tests/workflow/fixtures/workflow_parcial_execution_missing_1_input.txt"),
        encoding='utf-8')

    Workflow._get_operations = lambda s, conf: {}
    instance_wf = Workflow(workflow_test, fake_conf)
    assert instance_wf
