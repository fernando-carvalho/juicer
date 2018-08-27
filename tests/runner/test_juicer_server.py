# -*- coding: utf-8 -*-
import errno
import json
import signal
import sys

import mock
import os
from juicer.runner.control import StateControlRedis
from juicer.runner.server import JuicerServer
from mockredis.client import mock_strict_redis_client


def test_runner_read_start_queue_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = '1'
    workflow_id = '1000'
    workflow = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'type': 'execute',
        'workflow': {}
    }

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = server = JuicerServer(config, 'faked_minions.py',
                                           config_file_path='config.yaml')
            mocked_redis_conn = mocked_redis()
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(
                app_id) == '{"pid": 1, "port": 36000}'

            assert mocked_popen.call_args_list[0][0][0] == [
                'nohup', sys.executable, 'faked_minions.py',
                '-w', workflow_id, '-a', app_id,
                '-t', 'spark', '-c', 'config.yaml']
            assert mocked_popen.called

            # Was command removed from the queue?
            assert state_control.pop_start_queue(False) is None

            assert json.loads(state_control.pop_app_queue(app_id)) == workflow

            assert state_control.get_workflow_status(
                workflow_id) == JuicerServer.STARTED
            assert json.loads(state_control.pop_app_output_queue(app_id)) == {
                'code': 0, 'message': 'Minion is processing message execute'}


def test_runner_read_start_queue_workflow_not_started_failure():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = '1'
    workflow_id = '1000'
    workflow = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'type': 'execute',
        'workflow': {}
    }

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))

            # This workflow is being processed, should not start it again
            # state_control.set_workflow_status(workflow_id, JuicerServer.STARTED)
            server.active_minions[(workflow_id, app_id)] = '_'

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(app_id) is None
            assert not mocked_popen.called
            # Was command removed from the queue?
            assert state_control.pop_start_queue(False) is None

            assert state_control.get_app_queue_size(workflow_id) == 0


def test_runner_read_start_queue_minion_already_running_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    workflow_id = 1000
    workflow = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'type': 'execute',
        'workflow': {}
    }

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))
            state_control.set_minion_status(app_id, JuicerServer.STARTED)

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(
                app_id) == JuicerServer.STARTED

            assert not mocked_popen.called
            # Was command removed from the queue?
            assert mocked_redis_conn.lpop('start') is None
            assert json.loads(state_control.pop_app_queue(app_id)) == workflow

            assert state_control.get_workflow_status(
                workflow_id) == JuicerServer.STARTED
            assert json.loads(state_control.pop_app_output_queue(app_id)) == {
                'code': 0, 'message': 'Minion is processing message execute'}


def test_runner_read_start_queue_missing_details_failure():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    workflow_id = 1000
    # incorrect key, should raise exception
    workflow = {
        'workflow_id': workflow_id,
        'xapp_id': app_id,
        'type': 'execute',
        'workflow': {}
    }

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()
            # Publishes a message to process data
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))
            state_control.set_minion_status(app_id, JuicerServer.STARTED)

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(
                app_id) == JuicerServer.STARTED

            assert not mocked_popen.called
            # Was command removed from the queue?
            assert state_control.pop_start_queue(block=False) is None
            assert state_control.pop_app_queue(app_id, block=False) is None
            assert state_control.pop_app_output_queue(app_id,
                                                      block=False) is None

            assert mocked_redis_conn.hget(workflow_id, 'status') is None


def test_runner_master_queue_client_shutdown_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    ticket = {"app_id": app_id, 'reason': JuicerServer.HELP_UNHANDLED_EXCEPTION}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            with mock.patch('os.kill') as mocked_kill:
                server = JuicerServer(config, 'faked_minions.py')
                mocked_redis_conn = mocked_redis()
                # Publishes a message to process data
                state_control = StateControlRedis(mocked_redis_conn)

                # Configure minion
                status = {'app_id': app_id, 'pid': 9999}
                state_control.set_minion_status(app_id, json.dumps(status))
                error = OSError()
                error.errno = errno.ESRCH
                mocked_kill.side_effect = error
                # Publishes a message to master queue
                state_control.push_master_queue(json.dumps(ticket))

                # Start of testing
                server.read_minion_support_queue(mocked_redis_conn)

                assert state_control.get_minion_status(
                    app_id) == '{"pid": 1, "port": 36000}'

                assert mocked_popen.called

                mocked_kill.assert_called_once_with(status['pid'],
                                                    signal.SIGKILL)


# def test_runner_master_watch_minion_process_success():
#     config = {
#         'juicer': {
#             'servers': {
#                 'redis_url': "nonexisting.mock"
#             }
#         }
#     }
#
#     class PubSub:
#         def __init__(self, m_redis):
#             self.mocked_redis = m_redis
#             self.channel = ""
#
#         def psubscribe(self, channel):
#             self.channel = channel
#
#         def listen(self):
#             for v in self.mocked_redis.pubsub_channels[self.channel]:
#                 yield v
#
#     class CustomMockRedis(MockRedis):
#         def __init__(self, strict=False, clock=None, load_lua_dependencies=True,
#                      blocking_timeout=1000, blocking_sleep_interval=0.01,
#                      **kwargs):
#             super(CustomMockRedis, self).__init__(strict, clock,
#                                                   load_lua_dependencies,
#                                                   blocking_timeout,
#                                                   blocking_sleep_interval,
#                                                   **kwargs)
#             self.pubsub = lambda: PubSub(self)
#             self.pubsub_channels = collections.defaultdict(list)
#
#         def publish(self, channel, message):
#             self.pubsub_channels[channel].append(message)
#
#         def script_kill(self):
#             pass
#
#     with mock.patch('redis.StrictRedis',
#                     CustomMockRedis) as mocked_redis:
#         server = JuicerServer(config, 'faked_minions.py')
#         mocked_redis_conn = mocked_redis()
#
#         mocked_redis_conn.pubsub_channels['__keyevent@*__:expired'].append(
#             {'pattern': None, 'type': 'psubscribe',
#              'channel': '__keyevent@*__:expired', 'data': 1L}
#         )
#         mocked_redis_conn.pubsub_channels['__keyevent@*__:expired'].append(
#             {'pattern': '__keyevent@*__:expired', 'type': 'pmessage',
#              'channel': '__keyevent@0__:expired', 'data': 'a'}
#         )
#         # Configure minion
#
#         # Start of testing
#         server.watch_minion_process(mocked_redis_conn)


def test_runner_multiple_jobs_single_app():
    """
    - Start a juicer server
    - Instanciate a minion for an application
    - Submit more than one job to the same (workflow_id,app_id)
    - Assert that just one minion was launched
    """

    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    workflow_id = 1000
    workflow = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'type': 'execute',
        'workflow': {}
    }

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()

            # Publishes a message to process data
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))
            state_control.push_start_queue(json.dumps(workflow))

            # Start of testing
            server.read_start_queue(mocked_redis_conn)
            server.read_start_queue(mocked_redis_conn)

            assert len(server.active_minions) == 1
            assert mocked_popen.called


def test_runner_multiple_jobs_multiple_apps():
    """
    - Start a juicer server
    - Instanciate two minions for two different aplications
    - Submit jobs for both minions
    - Assert that two minions were launched
    """

    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }

    app_id = 1
    workflow_id = 1000
    workflow1 = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'type': 'execute',
        'workflow': {}
    }
    workflow2 = {
        'workflow_id': workflow_id + 1,
        'app_id': app_id + 1,
        'type': 'execute',
        'workflow': {}
    }

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()

            # Publishes a message to process data
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow1))
            state_control.push_start_queue(json.dumps(workflow2))
            state_control.push_start_queue(json.dumps(workflow2))
            state_control.push_start_queue(json.dumps(workflow1))

            # Start of testing
            server.read_start_queue(mocked_redis_conn)
            server.read_start_queue(mocked_redis_conn)
            server.read_start_queue(mocked_redis_conn)
            server.read_start_queue(mocked_redis_conn)

            assert len(server.active_minions) == 2
            assert mocked_popen.called


def test_runner_minion_termination():
    """
    - Start a juicer server
    - Instanciate two minions
    - Kill the first, assert that it was killed and the other remains
    - Kill the second, assert that all minions were killed and their state
      cleaned
    """

    try:
        from pyspark.sql import SparkSession
    except ImportError as ie:
        # we will skip this test because pyspark is not installed
        return

    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    workflow_id = 1000
    workflow1 = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'type': 'execute',
        'workflow': {}
    }

    workflow1_kill = {
        'workflow_id': workflow_id,
        'app_id': app_id,
        'type': 'terminate',
    }

    workflow2 = {
        'workflow_id': workflow_id + 1,
        'app_id': app_id + 1,
        'type': 'execute',
        'workflow': {}
    }

    workflow2_kill = {
        'workflow_id': workflow_id + 1,
        'app_id': app_id + 1,
        'type': 'terminate',
    }

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        config_file_path = os.path.join(os.path.dirname(__file__),
                                        'fixtures', 'juicer-server-config.yaml')
        server = JuicerServer(config, 'faked_minions.py',
                              config_file_path=config_file_path)
        mocked_redis_conn = mocked_redis()

        # Publishes a message to process data
        state_control = StateControlRedis(mocked_redis_conn)

        # Publishes a message to process data
        state_control.push_start_queue(json.dumps(workflow1))
        state_control.push_start_queue(json.dumps(workflow2))

        # Start of testing
        server.read_start_queue(mocked_redis_conn)
        server.read_start_queue(mocked_redis_conn)

        assert len(server.active_minions) == 2

        # kill first minion
        state_control.push_start_queue(json.dumps(workflow1_kill))
        server.read_start_queue(mocked_redis_conn)
        assert len(server.active_minions) == 1

        # kill second minion
        state_control.push_start_queue(json.dumps(workflow2_kill))
        server.read_start_queue(mocked_redis_conn)
        assert len(server.active_minions) == 0


def test_global_configuration():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        server = JuicerServer(config, 'faked_minions.py')

        # the configuration should be set by now, let's check it
        from juicer.runner import configuration
        assert configuration.get_config() == config
