# coding=utf-8
import json

import pytest
from juicer.service import stand_service
from mock import patch
from . import fake_req


@patch('requests.patch')
def test_job_source_code_success(mocked_patch):
    text = {
        'id': 1,
        'name': 'Source'
    }
    mocked_patch.side_effect = fake_req(200, json.dumps(text))()
    url = 'http://stand'
    token = '00000'
    job_id = 982
    source = 'def f()\n    print("OK")'

    resp = stand_service.save_job_source_code(url, token, job_id, source)
    for k, v in resp.items():
        assert v == text[k]

    mocked_patch.assert_called_with(
        'http://stand/jobs/982/source-code',
        data='{"source": "def f()\\n    print(\\"OK\\")", "secret": "00000"}',
        headers={'Content-Type': 'application/json', 'X-Auth-Token': '00000'})


@patch('requests.patch')
def test_job_source_code_failure(mocked_patch):
    text = {
        'id': 1,
        'name': 'Source'
    }
    mocked_patch.side_effect = fake_req(201, json.dumps(text))()
    url = 'http://stand'
    token = '00000'
    job_id = 982
    source = 'def f()\n    print("OK")'

    data = '{"source": "def f()\\n    print(\\"OK\\")", "secret": "00000"}'
    headers = {'Content-Type': 'application/json', 'X-Auth-Token': '00000'}
    with pytest.raises(RuntimeError):
        resp = stand_service.save_job_source_code(url, token, job_id, source)
        mocked_patch.assert_called_with('http://stand/jobs/982/source-code',
                                        data=data, headers=headers)
        for k, v in resp.items():
            assert v == text[k]
