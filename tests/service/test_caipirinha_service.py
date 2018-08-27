# coding=utf-8
import json

import pytest
import requests
from juicer.service import caipirinha_service
from mock import patch, call
from . import fake_req


# noinspection PyProtectedMember, PyUnresolvedReferences
def test_update_caipirinha_success():
    text = {
        'id': 1,
        'name': 'Visualization'
    }
    with patch.object(requests, 'post',
                      new_callable=fake_req(200, json.dumps(text))):
        resp = caipirinha_service._update_caipirinha(
            "http://caipirinha", "/visualizations", "OK", 1,
            {"id": 1, "name": "Dummt"})
        for k, v in resp.items():
            assert v == text[k]

    text = {
        'name': 'Visualization'
    }
    with patch.object(requests, 'post',
                      new_callable=fake_req(200, json.dumps(text))):
        resp = caipirinha_service._update_caipirinha(
            "http://caipirinha", "/visualizations", "OK", '',
            {"id": 1, "name": "Dummt"})
        for k, v in resp.items():
            assert v == text[k]


# noinspection PyProtectedMember, PyUnresolvedReferences
def test_update_caipirinha_fail():
    text = "Not found"
    with patch.object(requests, 'post',
                      new_callable=fake_req(404, json.dumps(text))):
        with pytest.raises(RuntimeError):
            resp = caipirinha_service._update_caipirinha(
                "http://caipirinha", "/visualizations", "OK", 1,
                {"id": 1, "name": "Dummt"})
            for k, v in resp.items():
                assert v == text[k]


# noinspection PyUnusedLocal
def emit(*args, **kwargs):
    pass


class VisualizationModel(object):
    def __init__(self, title, type_id=1):
        self.title = title
        self.type_id = type_id


# noinspection PyProtectedMember, PyUnresolvedReferences
@patch('tests.service.test_caipirinha_service.emit')
@patch('requests.post')
def test_new_dashboard(mocked_post, mocked_emit):
    text = {
        'id': 1,
        'name': 'Visualization'
    }
    mocked_post.side_effect = fake_req(200, json.dumps(text))()
    config = {
        'juicer': {
            'services': {
                'caipirinha': {
                    'url': 'http://caipirinha',
                    'auth_token': '0000'
                }
            }
        }
    }

    title = 'Dashboard title'
    user = {'id': 1, 'name': 'Lemon'}
    workflow_id = 1999
    workflow_name = 'Test'
    job_id = 982
    task_id = 555
    visualizations = []
    emit_event_fn = mocked_emit

    resp = caipirinha_service.new_dashboard(
        config, title, user, workflow_id, workflow_name, job_id, task_id,
        visualizations, emit_event_fn)
    for k, v in resp.items():
        assert v == text[k]

    mocked_post.assert_called_with(
        'http://caipirinha/dashboards',
        data='{"workflow_id": 1999, "job_id": 982, "task_id": 555, '
             '"title": "Dashboard title", "visualizations": [], '
             '"workflow_name": "Test", "user": {"id": 1, "name": "Lemon"}}',
        headers={'X-Auth-Token': '0000'})
    calls = [call('update task', identifier=555,
                  message='Saving visualizations',
                  status='RUNNING', task={'id': 555},
                  type='STATUS'),
             call('update task', identifier=555,
                  message='Visualizations saved',
                  status='COMPLETED', task={'id': 555},
                  type='STATUS'),
             ]
    mocked_emit.assert_has_calls(calls)

    mocked_emit.reset_mock()
    mocked_post.reset_mock()

    model = VisualizationModel('Title')
    visualizations.append({'id': 1, 'type': 'chart', 'task_id': 'aaa-bbb',
                           'model': model})

    caipirinha_service.new_dashboard(
        config, title, user, workflow_id, workflow_name, job_id, task_id,
        visualizations, emit_event_fn)

    mocked_post.assert_called_with(
        'http://caipirinha/dashboards',
        data='{"workflow_id": 1999, "job_id": 982, "task_id": 555, '
             '"title": "Dashboard title", '
             '"visualizations": [{"type": "chart", "id": 1, '
             '"task_id": "aaa-bbb"}], '
             '"workflow_name": "Test", "user": {"id": 1, "name": "Lemon"}}',
        headers={'X-Auth-Token': '0000'})


# noinspection PyProtectedMember, PyUnresolvedReferences
@patch('tests.service.test_caipirinha_service.emit')
@patch('requests.post')
def test_new_visualization(mocked_post, mocked_emit):
    text = {
        'id': 1,
        'name': 'Visualization'
    }
    mocked_post.side_effect = fake_req(200, json.dumps(text))()
    config = {
        'juicer': {
            'services': {
                'caipirinha': {
                    'url': 'http://caipirinha',
                    'auth_token': '0000'
                }
            }
        }
    }

    user = {'id': 1, 'name': 'Lemon'}
    workflow_id = 1999
    job_id = 982
    task_id = 555
    emit_event_fn = mocked_emit
    visualization = {'model': VisualizationModel('Title'), 'task_id': 1}

    resp = caipirinha_service.new_visualization(
        config, user, workflow_id, job_id, task_id, visualization,
        emit_event_fn, 'chart')
    for k, v in resp.items():
        assert v == text[k]

    mocked_post.assert_called_with(
        'http://caipirinha/visualizations',
        data='{"task_id": 1}',
        headers={'X-Auth-Token': '0000'})
    calls = [call('update task', identifier=555,
                  message='Saving visualizations',
                  status='RUNNING', task={'id': 555},
                  type='STATUS'),
             call('task result', identifier=1, message='Result generated',
                  operation={'id': 1}, operation_id=1, status='COMPLETED',
                  task={'id': 1}, title='Title', type='chart'),
             call('update task', identifier=555, message='Visualizations saved',
                  status='COMPLETED', task={'id': 555}, type='STATUS')]
    mocked_emit.assert_has_calls(calls)
