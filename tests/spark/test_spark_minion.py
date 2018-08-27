# coding=utf-8
from __future__ import print_function

import json
from textwrap import dedent

import mock
import os
import pytest
from dummy_spark import SparkConf, SparkContext
from juicer.runner.control import StateControlRedis
from juicer.spark.spark_minion import SparkMinion
from juicer.util import dataframe_util
from mockredis import mock_strict_redis_client

config = {
    'juicer': {
        'servers': {
            'redis_url': 'redis://invalid:2923'
        }
    }
}


# This functions are used to prevent minons
# from using unsupported operations in test mode
# noinspection PyUnusedLocal
def dummy_get_or_create_spark_session(b, c, d):
    return None


# noinspection PyUnusedLocal
def dummy_emit_event(room, namespace):
    # noinspection PyUnusedLocal
    def _dummy_emit_event(name, message, status, identifier):
        return None

    return _dummy_emit_event


# Auxiliary functions
def get_records():
    return [
        ['Brazil', 'UFMG', 223],
        ['Spain', 'UPV', 52],
        ['Italy', 'POLIMI', 921],
    ]


def get_side_effect(records, task_id, index=0):
    # noinspection PyUnusedLocal
    def side_effect0(w, g, c, out, j, state):
        print(dedent("""
            from dummy_spark import SparkConf, SparkContext
            import datetime
            sconf = SparkConf()
            sc = SparkContext(master='', conf=sconf)
            records = {records}
            rdd = sc.parallelize(records)
            class DataFrame:
                def __init__(self, rdd):
                    self.rdd = rdd

                def take(self, total):
                    return self
            def main(spark_session, cached_data, emit_event):
                return {{
                    '{task_id}': {{
                        'port0': {{'output': DataFrame(rdd), 'sample': []}},
                        'time': 20.0
                    }}
                }}
            """.format(records=json.dumps(records), task_id=task_id)), file=out)

    # noinspection PyUnusedLocal
    def side_effect1(w, g, c, out, j):
        print(dedent("""
            def main(spark_session, cached_data, emit_event):
                return {
                    'xyz-647': {
                        'port0': {'output': 'df', 'sample': []},
                        'time': 27.27
                    }
                }"""), file=out)

    # noinspection PyUnusedLocal
    def side_effect2(w, g, c, out, j):
        print(dedent("""
            def main(spark_session, cached_data, emit_event):
                return {'res': 'version 1.0'} """), file=out)

    # noinspection PyUnusedLocal
    def side_effect3(w, g, c, out, j):
        print(dedent("""
            def main(spark_session, cached_data, emit_event):
                return {'res': 'version 2.1'} """), file=out)

    # noinspection PyUnusedLocal
    def side_effect4(w, g, c, out, j):
        print(dedent("""
            a = 4
            def main(spark_session, cached_data, emit_event):
                return Invalid Code """), file=out)

    # noinspection PyUnusedLocal
    def side_effect5(w, g, c, out, j):
        print(dedent("""
            def main(spark_session, cached_data, emit_event):
                import time
                def infinity_loop(n):
                    while True:
                        time.sleep(1)
                    return 0
                spark_session.sparkContext.parallelize(range(0,2)). \
                        map(infinity_loop).count()
                return {'res', 'version 1.0'}"""), file=out)

    return \
        [side_effect0, side_effect1, side_effect2, side_effect3,
         side_effect4, side_effect5][index]


class DataFrame:
    def __init__(self, rdd):
        self.rdd = rdd

    # noinspection PyUnusedLocal
    def take(self, total):
        return self


# noinspection PyProtectedMember
def test_minion_ping_success():
    workflow_id = 6666
    app_id = 897987

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        minion = SparkMinion(redis_conn=redis_conn,
                             workflow_id=workflow_id, app_id=app_id,
                             config=config)
        minion._emit_event = dummy_emit_event
        minion._perform_ping()

        state_control = StateControlRedis(redis_conn)

        assert json.loads(state_control.get_minion_status(app_id)) == {
            'status': 'READY', 'pid': os.getpid()}
        assert state_control.get_app_output_queue_size(app_id) == 0


# noinspection PyProtectedMember
def test_minion_generate_output_success():
    workflow_id = 6666
    app_id = 897987

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        minion = SparkMinion(redis_conn=redis_conn,
                             workflow_id=workflow_id, app_id=app_id,
                             config=config)
        minion._emit_event = dummy_emit_event

        state_control = StateControlRedis(redis_conn)

        msgs = ["Message being sent \n{}", {'msg': 'Dictionary being sent'}]
        for msg in msgs:
            minion._generate_output(msg)
            result = json.loads(
                state_control.pop_app_output_queue(app_id, False))
            assert sorted(result.keys()) == sorted(
                ['date', 'status', 'message', 'workflow_id', 'app_id', 'code'])
            assert result['app_id'] == app_id
            assert result['message'] == msg
        assert state_control.get_app_output_queue_size(app_id) == 0


# noinspection PyProtectedMember
@pytest.mark.skip(reason="Not working")
def test_minion_perform_execute_success():
    workflow_id = '6666'
    app_id = '897447'
    job_id = '1'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = get_side_effect(None, 0, 1)

            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn,
                                 workflow_id=workflow_id, app_id=app_id,
                                 config=config)
            minion.get_or_create_spark_session = \
                dummy_get_or_create_spark_session
            minion._emit_event = dummy_emit_event
            # Configure mocked redis
            state_control = StateControlRedis(redis_conn)
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())

            msg = {
                'workflow_id': workflow_id,
                'app_id': app_id,
                'job_id': job_id,
                'type': 'execute',
                'workflow': data
            }

            state_control.push_app_queue(app_id, json.dumps(msg))

            minion._process_message()
            assert minion._state == {
                "xyz-647": {
                    'port0': {'output': "df", 'sample': []},
                    'time': 27.27
                }}, 'Invalid state'

            assert state_control.get_app_output_queue_size(
                app_id) == 1, 'Wrong number of output messages'


# noinspection PyProtectedMember
@pytest.mark.skip(reason="Not working")
def test_minion_perform_execute_reload_code_success():
    workflow_id = '6666'
    app_id = '667788'
    job_id = '1'
    workflow = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'job_id': job_id,
        'type': 'execute',
        'workflow': ''
    }
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            with mock.patch('juicer.workflow.workflow.Workflow'
                            '._build_initial_workflow_graph') as mocked_fn:
                mocked_fn.side_effect = lambda: ""
                # Setup for mocked_transpile
                mocked_transpile.side_effect = get_side_effect(None, None, 2)

                redis_conn = mocked_redis()
                minion = SparkMinion(redis_conn=redis_conn,
                                     workflow_id=workflow_id, app_id=app_id,
                                     config=config)
                minion.get_or_create_spark_session = \
                    dummy_get_or_create_spark_session
                minion._emit_event = dummy_emit_event
                # Configure mocked redis
                state_control = StateControlRedis(redis_conn)
                with open(os.path.join(os.path.dirname(__file__),
                                       'fixtures/simple_workflow.json')) as f:
                    data = json.loads(f.read())
                    workflow['workflow'] = data

                state_control.push_app_queue(app_id, json.dumps(workflow))

                minion._process_message()
                assert minion._state == {'res': 'version 1.0'}, 'Invalid state'

                # Executes the same workflow, but code should be different
                state_control.push_app_queue(app_id, json.dumps(workflow))
                assert state_control.get_app_output_queue_size(
                    app_id) == 1, 'Wrong number of output messages'

                state_control.pop_app_queue(app_id, True, 0)
                minion.transpiler.transpile = get_side_effect(None, None, 3)
                minion._process_message()

                assert minion._state == {'res': 'version 2.1'}, 'Invalid state'

                assert state_control.get_app_output_queue_size(
                    app_id) == 2, 'Wrong number of output messages'


# noinspection PyProtectedMember
@pytest.mark.skip(reason="Not working")
def test_minion_generate_invalid_code_failure():
    workflow_id = '6666'
    app_id = '667788'
    job_id = '1'
    workflow = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'job_id': job_id,
        'type': 'execute',
        'workflow': ''
    }
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            with mock.patch('juicer.workflow.workflow.Workflow'
                            '._build_initial_workflow_graph') as mocked_fn:
                mocked_fn.side_effect = lambda: ""
                # Setup for mocked_transpile
                mocked_transpile.side_effect = get_side_effect(None, None, 4)
                redis_conn = mocked_redis()
                minion = SparkMinion(redis_conn=redis_conn,
                                     workflow_id=workflow_id, app_id=app_id,
                                     config=config)
                minion._emit_event = dummy_emit_event
                # Configure mocked redis
                with open(os.path.join(os.path.dirname(__file__),
                                       'fixtures/simple_workflow.json')) as f:
                    data = json.loads(f.read())
                    workflow['workflow'] = data

                state_control = StateControlRedis(redis_conn)
                state_control.push_app_queue(app_id, json.dumps(workflow))
                minion._process_message()

                assert state_control.get_app_output_queue_size(
                    app_id) == 2, 'Wrong number of output messages'
                # Discards
                state_control.pop_app_output_queue(app_id)

                msg = json.loads(state_control.pop_app_output_queue(app_id))
                assert msg['status'] == 'ERROR'
                assert msg['message'][:19] == 'Invalid Python code'


# noinspection PyProtectedMember
def test_minion_perform_deliver_success():
    workflow_id = '6666'
    app_id = '1000'
    job_id = '1'
    out_queue = 'queue_2000'
    sconf = SparkConf()
    sc = SparkContext(master='', conf=sconf)

    rdd = sc.parallelize(get_records())

    df0 = DataFrame(rdd=rdd)
    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        state_control = StateControlRedis(redis_conn)

        data = {
            'workflow_id': workflow_id,
            'app_id': app_id,
            'job_id': job_id,
            'type': 'deliver',
            'task_id': '033f-284ab-28987e',
            'port': 'port0',
            'output': out_queue,
            'workflow': ''
        }
        state_control.push_app_queue(app_id, json.dumps(data))
        minion = SparkMinion(redis_conn=redis_conn,
                             workflow_id=workflow_id, app_id=app_id,
                             config=config)
        minion._emit_event = dummy_emit_event
        minion._state = {
            data['task_id']: {
                'port0': {'output': df0, 'sample': []},
                'time': 35.92
            }
        }
        minion._process_message()

        # Discard first status message
        state_control.pop_app_output_queue(app_id, False)

        msg = json.loads(state_control.pop_app_output_queue(app_id, False))
        assert msg['status'] == 'SUCCESS', 'Invalid status'
        assert msg['code'] == minion.MNN002[0], 'Invalid code'

        # CSV data
        csv_records = '\n'.join(
            map(dataframe_util.convert_to_csv, get_records()))

        result = json.loads(state_control.pop_queue(out_queue, False))
        assert result['sample'] == csv_records, 'Wrong CSV generated'


# noinspection PyProtectedMember
def test_minion_perform_deliver_missing_state_process_app_with_success():
    workflow_id = '6666'
    app_id = '1000'
    job_id = '1'
    out_queue = 'queue_2000'
    task_id = '033f-284ab-28987e'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            with mock.patch('juicer.workflow.workflow.Workflow'
                            '._build_initial_workflow_graph') as mocked_fn:
                mocked_fn.side_effect = lambda: ""
                # Setup for mocked_transpile
                mocked_transpile.side_effect = get_side_effect(get_records(),
                                                               task_id)
                redis_conn = mocked_redis()
                state_control = StateControlRedis(redis_conn)

                data = {
                    'workflow_id': workflow_id,
                    'app_id': app_id,
                    'job_id': job_id,
                    'type': 'deliver',
                    'task_id': task_id,
                    'port': 'port0',
                    'output': out_queue,
                    'workflow': {"tasks": [], "flows": []}
                }

                state_control.push_app_queue(app_id, json.dumps(data))
                minion = SparkMinion(redis_conn=redis_conn,
                                     workflow_id=workflow_id, app_id=app_id,
                                     config=config)
                minion.get_or_create_spark_session = \
                    dummy_get_or_create_spark_session
                minion._emit_event = dummy_emit_event
                minion._state = {
                }
                minion._process_message()

                # Discard first status message
                state_control.pop_app_output_queue(app_id, False)

                msg = json.loads(
                    state_control.pop_app_output_queue(app_id, False))
                assert msg['status'] == 'WARNING', 'Invalid status'
                assert msg['code'] == minion.MNN003[0], 'Invalid code'

                # CSV data
                csv_records = '\n'.join(
                    map(dataframe_util.convert_to_csv, get_records()))

                result = json.loads(state_control.pop_queue(out_queue, False))
                assert result['sample'] == csv_records, 'Wrong CSV generated'


# noinspection PyProtectedMember
@pytest.mark.skip(reason="Not working")
def test_minion_perform_deliver_missing_state_process_app_with_failure():
    workflow_id = '6666'
    app_id = '6000'
    job_id = '1'
    out_queue = 'queue_2000'
    task_id = '033f-284ab-28987e'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            with mock.patch('juicer.workflow.workflow.Workflow'
                            '._build_initial_workflow_graph') as mocked_fn:
                mocked_fn.side_effect = lambda: ""
                # Setup for mocked_transpile
                # Invalid code
                mocked_transpile.side_effect = get_side_effect(get_records(),
                                                               task_id, 4)
                redis_conn = mocked_redis()
                state_control = StateControlRedis(redis_conn)

                data = {
                    'workflow_id': workflow_id,
                    'app_id': app_id,
                    'job_id': job_id,
                    'type': 'deliver',
                    'task_id': task_id,
                    'port': 'port0',
                    'output': out_queue,
                    'workflow': {"tasks": [], "flows": []}
                }

                state_control.push_app_queue(app_id, json.dumps(data))
                minion = SparkMinion(redis_conn=redis_conn,
                                     workflow_id=workflow_id, app_id=app_id,
                                     config=config)
                minion._emit_event = dummy_emit_event
                minion._state = {
                }
                minion._process_message()

                # Discard first status message
                state_control.pop_app_output_queue(app_id, False)

                # First message is about missing state
                msg = json.loads(
                    state_control.pop_app_output_queue(app_id, False))
                assert msg['status'] == 'WARNING', 'Invalid status'
                assert msg['code'] == minion.MNN003[0], 'Invalid code'

                # Second message is about invalid Python code
                msg = json.loads(
                    state_control.pop_app_output_queue(app_id, False))
                assert msg['status'] == 'ERROR', 'Invalid status'
                assert msg.get('code') == minion.MNN006[0], 'Invalid code'

                # Third message is about unable to read data
                msg = json.loads(
                    state_control.pop_app_output_queue(app_id, False))
                assert msg['status'] == 'ERROR', 'Invalid status'
                assert msg.get('code') == minion.MNN005[0], 'Invalid code'

                assert state_control.get_app_output_queue_size(
                    app_id) == 0, 'There are messages in app output queue!'

                result = json.loads(state_control.pop_queue(out_queue, False))
                assert not result['sample'], 'Wrong CSV generated'


# noinspection PyProtectedMember
def test_minion_perform_deliver_missing_state_invalid_port_failure():
    workflow_id = '6666'
    app_id = '5001'
    job_id = '1'
    out_queue = 'queue_50001'
    task_id = 'f033f-284ab-28987e-232add'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            with mock.patch('juicer.workflow.workflow.Workflow'
                            '._build_initial_workflow_graph') as mocked_fn:
                mocked_fn.side_effect = lambda: ""
                # Setup for mocked_transpile
                mocked_transpile.side_effect = get_side_effect(get_records(),
                                                               task_id, 0)
                redis_conn = mocked_redis()
                state_control = StateControlRedis(redis_conn)

                data = {
                    'workflow_id': workflow_id,
                    'app_id': app_id,
                    'job_id': job_id,
                    'type': 'deliver',
                    'task_id': task_id,
                    'port': 'port2',  # This port is invalid
                    'output': out_queue,
                    'workflow': {"tasks": [], "flows": []}
                }

                state_control.push_app_queue(app_id, json.dumps(data))
                minion = SparkMinion(redis_conn=redis_conn,
                                     workflow_id=workflow_id, app_id=app_id,
                                     config=config)
                minion.get_or_create_spark_session = \
                    dummy_get_or_create_spark_session
                minion._emit_event = dummy_emit_event
                minion._state = {}
                minion._process_message()

                # Discard first status message
                state_control.pop_app_output_queue(app_id, False)

                msg = json.loads(
                    state_control.pop_app_output_queue(app_id, False))
                assert msg['status'] == 'WARNING', 'Invalid status'
                assert msg['code'] == minion.MNN003[0], 'Invalid code'

                msg = json.loads(
                    state_control.pop_app_output_queue(app_id, False))
                assert msg['status'] == 'ERROR', 'Invalid status'
                assert msg.get('code') == minion.MNN004[0], 'Invalid code'

                result = json.loads(state_control.pop_queue(out_queue, False))
                assert not result['sample'], 'Wrong CSV generated'


# noinspection PyProtectedMember
def test_minion_perform_deliver_missing_state_unsupported_output_failure():
    workflow_id = '6666'
    app_id = '4001'
    job_id = '1'
    out_queue = 'queue_40001'
    task_id = 'f033f-284ab-28987e-232add'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            with mock.patch('juicer.workflow.workflow.Workflow'
                            '._build_initial_workflow_graph') as mocked_fn:
                mocked_fn.side_effect = lambda: ""
                # Setup for mocked_transpile
                mocked_transpile.side_effect = get_side_effect(get_records(),
                                                               task_id)
                redis_conn = mocked_redis()
                state_control = StateControlRedis(redis_conn)

                data = {
                    'workflow_id': workflow_id,
                    'app_id': app_id,
                    'job_id': job_id,
                    'type': 'deliver',
                    'task_id': task_id,
                    'port': 'port1',
                    'output': out_queue,
                    'workflow': {"tasks": [], "flows": []}
                }

                state_control.push_app_queue(app_id, json.dumps(data))
                minion = SparkMinion(redis_conn=redis_conn,
                                     workflow_id=workflow_id, app_id=app_id,
                                     config=config)
                minion._emit_event = dummy_emit_event
                minion.get_or_create_spark_session = \
                    dummy_get_or_create_spark_session
                minion._emit_event = dummy_emit_event
                minion._state = {}

                minion._process_message()

                # Discard first status message
                state_control.pop_app_output_queue(app_id, False)

                msg = json.loads(
                    state_control.pop_app_output_queue(app_id, False))
                assert msg['status'] == 'WARNING', 'Invalid status'
                assert msg['code'] == minion.MNN003[0], 'Invalid code'

                msg = json.loads(
                    state_control.pop_app_output_queue(app_id, False))
                assert msg['status'] == 'ERROR', 'Invalid status'
                assert msg.get('code') == minion.MNN004[0], 'Invalid code'

                result = json.loads(state_control.pop_queue(out_queue, False))
                assert not result['sample'], 'Wrong CSV generated'

                # check proper termination
                minion.terminate()
                assert not minion.is_spark_session_available()


# noinspection PyProtectedMember
@pytest.mark.skip(reason="Not working")
def test_minion_spark_configuration():
    """
    - Start a juicer server
    - Instanciate one minion passing specific configurations w.r.t. runtime
      environments (app_configs)
    - Assert that the spark context within the minion inherited the same configs
      that it was supposed to inherit
    """

    try:
        # noinspection PyUnresolvedReferences
        from pyspark.sql import SparkSession
    except ImportError:
        # we will skip this test because pyspark is not installed
        return

    workflow_id = '6666'
    app_id = '667788'
    job_id = '1'
    app_configs = {'spark.master': 'local[3]',
                   'config1': '1', 'config2': '2'}
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            with mock.patch('juicer.workflow.workflow.Workflow'
                            '._build_initial_workflow_graph') as mocked_fn:
                mocked_fn.side_effect = lambda: ""
                # Setup for mocked_transpile
                mocked_transpile.side_effect = get_side_effect(None, 0, 1)

                redis_conn = mocked_redis()
                minion = SparkMinion(redis_conn=redis_conn,
                                     workflow_id=workflow_id, app_id=app_id,
                                     config=config)
                minion._emit_event = dummy_emit_event

                # Configure mocked redis
                state_control = StateControlRedis(redis_conn)
                with open(os.path.join(os.path.dirname(__file__),
                                       'fixtures/simple_workflow.json')) as f:
                    data = json.loads(f.read())

                state_control.push_app_queue(app_id, json.dumps({
                    'workflow_id': workflow_id,
                    'app_id': app_id,
                    'job_id': job_id,
                    'type': 'execute',
                    'app_configs': app_configs,
                    'workflow': data
                }))

                minion._process_message()

                # check spark session health
                assert minion.spark_session is not None
                assert minion.is_spark_session_available()

                # check configs
                ctx_configs = \
                    minion.spark_session.sparkContext.getConf().getAll()
                ctx_configs = {k: v for k, v in ctx_configs}
                for k, v in app_configs.items():
                    assert ctx_configs[k] == v

                # check app name
                name = minion.spark_session.sparkContext.appName
                assert name == u'{}(workflow_id={},app_id={})'.format(
                    data['name'], workflow_id, app_id)

                # check proper termination
                minion.terminate()
                assert not minion.is_spark_session_available()

                state_control.pop_app_output_queue(app_id, False)
                msg = json.loads(
                    state_control.pop_app_output_queue(app_id, False))
                assert msg['status'] == 'SUCCESS', 'Invalid status'
                assert msg['code'] == minion.MNN008[0], 'Invalid code'


# noinspection PyUnresolvedReferences,PyProtectedMember
@pytest.mark.skip(reason="Not working")
def test_minion_terminate():
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        # we will skip this test because pyspark is not installed
        return

    workflow_id = '6666'
    app_id = '897447'
    job_id = '1'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = get_side_effect(None, 0, 5)

            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn,
                                 workflow_id=workflow_id, app_id=app_id,
                                 config=config)
            minion._emit_event = dummy_emit_event

            # Configure mocked redis
            state_control = StateControlRedis(redis_conn)
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())

            # execute message
            state_control.push_app_queue(app_id, json.dumps({
                'workflow_id': workflow_id,
                'app_id': app_id,
                'job_id': job_id,
                'type': 'execute',
                'workflow': data
            }))
            minion._process_message_nb()
            # discard extra message
            state_control.pop_app_output_queue(app_id, False)

            # job termination
            state_control.push_app_queue(app_id, json.dumps({
                'workflow_id': workflow_id,
                'app_id': app_id,
                'job_id': job_id,
                'type': 'terminate'
            }))
            minion._process_message()

            state_control.pop_app_output_queue(app_id, False)

            # first the spark app will throw an exception regarding the job
            # canceling
            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'ERROR', 'Invalid status'
            assert msg['code'] == 1000, 'Invalid code'

            # second the minion will report success for the job canceling
            # operation
            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'SUCCESS', 'Invalid status'
            assert msg['code'] == SparkMinion.MNN007[0], 'Invalid code'

            # assert app still alive
            assert minion.spark_session is not None
            assert minion.is_spark_session_available()

            # app termination
            state_control.push_app_queue(app_id, json.dumps({
                'workflow_id': workflow_id,
                'app_id': app_id,
                'type': 'terminate'
            }))
            minion._process_message()
            # discard extra message
            state_control.pop_app_output_queue(app_id, False)

            msg = json.loads(state_control.pop_app_output_queue(app_id, False))
            assert msg['status'] == 'SUCCESS', 'Invalid status'
            assert msg['code'] == SparkMinion.MNN008[0], 'Invalid code'

            # assert app still alive
            assert minion.spark_session is None
            assert not minion.is_spark_session_available()

            minion.terminate()
            assert not minion.is_spark_session_available()


def test_minion_global_configuration():
    workflow_id = '6666'
    app_id = '897447'

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        SparkMinion(redis_conn=redis_conn,
                    workflow_id=workflow_id, app_id=app_id,
                    config=config)

        # the configuration should be set by now, let's check it
        from juicer.runner import configuration
        assert configuration.get_config() == config
