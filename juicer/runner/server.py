# coding=utf-8
"""
"
"""
import argparse
import errno
import gettext
import json
import logging.config
import multiprocessing
import signal
import subprocess
import sys
import time
import urlparse

import os
import redis
import yaml
from juicer.exceptions import JuicerException
from juicer.runner import configuration
from juicer.runner import protocol as juicer_protocol
from juicer.runner.control import StateControlRedis
from redis.exceptions import ConnectionError

locales_path = os.path.join(os.path.dirname(__file__), '..', 'i18n', 'locales')

os.chdir(os.environ.get('JUICER_HOME', '.'))
logging.config.fileConfig('logging_config.ini')
log = logging.getLogger('juicer.runner.server')


class JuicerServer:
    """
    The JuicerServer is responsible for managing the lifecycle of minions.
    A minion controls a application, i.e., an active instance of an workflow.
    Thus, the JuicerServer receives launch request from clients, launches and
    manages minion processes and takes care of their properly termination.
    """
    STARTED = 'STARTED'
    LOADED = 'LOADED'
    TERMINATED = 'TERMINATED'
    HELP_UNHANDLED_EXCEPTION = 1
    HELP_STATE_LOST = 2

    def __init__(self, config, minion_executable, log_dir='/tmp',
                 config_file_path=None):

        self.minion_support_process = None
        self.new_minion_watch_process = None
        self.start_process = None
        self.minion_status_process = None
        self.state_control = None
        self.minion_watch_process = None

        self.active_minions = {}

        self.config = config
        configuration.set_config(config)
        self.config_file_path = config_file_path
        self.minion_executable = minion_executable
        self.log_dir = log_dir or self.config['juicer'].get('log', {}).get(
            'path', '/tmp')

        signal.signal(signal.SIGTERM, self._terminate)
        self.platform = 'spark'

        self.port_range = range(*(config['juicer'].get('minion', {}).get(
            'libprocess_port_range', [36000, 36500])))
        self.advertise_ip = config['juicer'].get('minion', {}).get(
            'libprocess_advertise_ip')

        # Minion requires 3 different ports:
        # 1 for libprocess/Mesos communication
        # 1 for driver port
        # 1 for block manager
        self.port_offset = config['juicer'].get('minion', {}).get(
            'port_offset', 100)

    def start(self):
        signal.signal(signal.SIGTERM, self._terminate_minions)
        log.info(_('Starting master process. Reading "start" queue'))

        parsed_url = urlparse.urlparse(
            self.config['juicer']['servers']['redis_url'])
        redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                       port=parsed_url.port)

        # Start pending minions
        apps = [q.split('_')[-1] for q in redis_conn.keys('queue_app_*')]
        self.state_control = StateControlRedis(redis_conn)

        for app in apps:
            log.warn(_('Starting pending app {}').format(app))
            self._start_minion(app, app, self.state_control, self.platform)
        while True:
            self.read_start_queue(redis_conn)

    # noinspection PyMethodMayBeStatic
    def read_start_queue(self, redis_conn):
        app_id = None
        try:
            self.state_control = StateControlRedis(redis_conn)
            # Process next message
            msg = self.state_control.pop_start_queue()
            msg_info = json.loads(msg)

            # Extract message type and common parameters
            msg_type = msg_info['type']
            workflow_id = str(msg_info['workflow_id'])
            app_id = str(msg_info['app_id'])

            if msg_type in (juicer_protocol.EXECUTE, juicer_protocol.DELIVER):
                self.platform = msg_info['workflow'].get('platform', {}).get(
                        'slug', 'spark')
                self._forward_to_minion(msg_type, workflow_id, app_id, msg,
                                        self.platform)

            elif msg_type == juicer_protocol.TERMINATE:
                self._forward_to_minion(msg_type, workflow_id, app_id, msg,
                                        self.platform)
                self._terminate_minion(workflow_id, app_id)

            else:
                log.warn(_('Unknown message type %s'), msg_type)

        except ConnectionError as cx:
            log.exception(cx)
            time.sleep(1)

        except JuicerException as je:
            log.exception(je)
            if app_id:
                self.state_control.push_app_output_queue(app_id, json.dumps(
                    {'code': je.code, 'message': je.message}))
        except KeyboardInterrupt:
            pass
        except Exception as ex:
            log.exception(ex)
            if app_id:
                self.state_control.push_app_output_queue(
                    app_id, json.dumps({'code': 500, 'message': ex.message}))

    def _forward_to_minion(self, msg_type, workflow_id, app_id, msg, platform):
        # Get minion status, if it exists
        minion_info = self.state_control.get_minion_status(app_id)
        log.info(_('Minion status for (workflow_id=%s,app_id=%s): %s'),
                 workflow_id, app_id, minion_info)

        # If there is status registered for the application then we do not
        # need to launch a minion for it, because it is already running.
        # Otherwise, we launch a new minion for the application.
        if minion_info:
            log.info(_('Minion (workflow_id=%s,app_id=%s) is running on %s.'),
                     workflow_id, app_id, platform)
        else:
            # This is a special case when the minion timed out.
            # In this case we kill it before starting a new one
            if (workflow_id, app_id) in self.active_minions:
                self._terminate_minion(workflow_id, app_id)

            minion_process = self._start_minion(
                workflow_id, app_id, self.state_control, platform)
            self.active_minions[(workflow_id, app_id)] = {
                'pid': minion_process.pid, 'process': minion_process,
                'port': self._get_next_available_port()}

        # Forward the message to the minion, which can be an execute or a
        # deliver command
        self.state_control.push_app_queue(app_id, msg)
        self.state_control.set_workflow_status(workflow_id, self.STARTED)

        log.info(_('Message %s forwarded to minion (workflow_id=%s,app_id=%s)'),
                 msg_type, workflow_id, app_id)
        # log.info(_('Message content (workflow_id=%s,app_id=%s): %s'),
        #          workflow_id, app_id, msg)
        self.state_control.push_app_output_queue(app_id, json.dumps(
            {'code': 0,
             'message': 'Minion is processing message %s' % msg_type}))

    def _start_minion(self, workflow_id, app_id, state_control, platform,
                      restart=False):

        minion_id = 'minion_{}_{}'.format(workflow_id, app_id)
        stdout_log = os.path.join(self.log_dir, minion_id + '_out.log')
        stderr_log = os.path.join(self.log_dir, minion_id + '_err.log')
        log.debug(_('Forking minion %s.'), minion_id)

        port = self._get_next_available_port()
        # Setup command and launch the minion script. We return the subprocess
        # created as part of an active minion.
        # spark.driver.port and spark.driver.blockManager.port are required
        # when running the driver inside a docker container.
        open_opts = ['nohup', sys.executable, self.minion_executable,
                     '-w', str(workflow_id), '-a', str(app_id), '-t', platform,
                     '-c',
                     self.config_file_path, ]
        log.debug(_('Minion command: %s'), json.dumps(open_opts))

        # Mesos / libprocess configuration. See:
        # http://mesos.apache.org/documentation/latest/configuration/libprocess/
        cloned_env = os.environ.copy()
        cloned_env['LIBPROCESS_PORT'] = str(port)
        cloned_env['SPARK_DRIVER_PORT'] = str(port + self.port_offset)
        cloned_env['SPARK_DRIVER_BLOCKMANAGER_PORT'] = str(
            port + 2 * self.port_offset)

        if self.advertise_ip is not None:
            cloned_env['LIBPROCESS_ADVERTISE_IP'] = self.advertise_ip

        proc = subprocess.Popen(open_opts,
                                stdout=open(stdout_log, 'a'),
                                stderr=open(stderr_log, 'a'),
                                env=cloned_env)

        # Expires in 30 seconds and sets only if it doesn't exist
        proc_id = int(proc.pid)
        state_control.set_minion_status(
            app_id, json.dumps({'pid': proc_id, 'port': port}), ex=30,
            nx=False)
        return proc

    def _terminate_minion(self, workflow_id, app_id):
        # In this case we got a request for terminating this workflow
        # execution instance (app). Thus, we are going to explicitly
        # terminate the workflow, clear any remaining metadata and return
        if not (workflow_id, app_id) in self.active_minions:
            log.warn('(%s, %s) not in active minions ', workflow_id, app_id)
        log.info(_("Terminating (workflow_id=%s,app_id=%s)"),
                 workflow_id, app_id)
        if (workflow_id, app_id) in self.active_minions:
            os.kill(self.active_minions[(workflow_id, app_id)].get('pid'),
                    signal.SIGTERM)
            del self.active_minions[(workflow_id, app_id)]

    def minion_support(self):
        parsed_url = urlparse.urlparse(
            self.config['juicer']['servers']['redis_url'])
        redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                       port=parsed_url.port)
        while True:
            self.read_minion_support_queue(redis_conn)

    def read_minion_support_queue(self, redis_conn):
        try:
            state_control = StateControlRedis(redis_conn)
            ticket = json.loads(state_control.pop_master_queue())
            workflow_id = ticket.get('workflow_id')
            app_id = ticket.get('app_id', ticket.get('workflow_id'))
            reason = ticket.get('reason')
            log.info(_("Master received a ticket for app %s"), app_id)
            if reason == self.HELP_UNHANDLED_EXCEPTION:
                # Let's kill the minion and start another
                minion_info = json.loads(
                    state_control.get_minion_status(app_id))
                while True:
                    try:
                        os.kill(minion_info['pid'], signal.SIGKILL)
                    except OSError as err:
                        if err.errno == errno.ESRCH:
                            break
                    time.sleep(.5)

                self._start_minion(workflow_id, app_id, state_control,
                                   self.platform)

            elif reason == self.HELP_STATE_LOST:
                pass
            else:
                log.warn(_("Unknown help reason %s"), reason)
        except KeyboardInterrupt:
            pass
        except ConnectionError as cx:
            log.exception(cx)
            time.sleep(1)

        except Exception as ex:
            log.exception(ex)

    def _get_next_available_port(self):
        used_ports = set(
            [minion['port'] for minion in self.active_minions.values()])
        for i in self.port_range:
            if i not in used_ports:
                return i
        raise ValueError(_('Unable to launch minion: there is not available '
                           'port for libprocess.'))

    def watch_new_minion(self):
        try:
            log.info(_('Watching minions events.'))

            parsed_url = urlparse.urlparse(
                self.config['juicer']['servers']['redis_url'])
            redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                           port=parsed_url.port)
            redis_conn.config_set('notify-keyspace-events', 'KE$gx')
            pub_sub = redis_conn.pubsub()
            pub_sub.psubscribe('__keyspace*__:key_minion_app*')
            for msg in pub_sub.listen():
                # print '|{}|'.format(msg.get('channel'))
                app_id = msg.get('channel', '').split('_')[-1]
                if app_id.isdigit():
                    app_id = int(app_id)
                    key = (app_id, app_id)
                    data = msg.get('data', '')
                    if key in self.active_minions:
                        if data == 'del' or data == 'expired':
                            del self.active_minions[key]
                            log.info(_('Minion {} finished.').format(app_id))
                            if redis_conn.lrange('queue_app_{}'.format(app_id),
                                                 0, 0):
                                log.warn(
                                    _('There are messages to process in app {} '
                                      'queue, starting minion.').format(app_id))
                                if self.state_control is None:
                                    self.state_control = StateControlRedis(
                                        redis_conn)
                                self._start_minion(
                                    app_id, app_id, self.state_control,
                                    self.platform)

                    elif data == 'set':
                        # Externally launched minion
                        minion_info = json.loads(redis_conn.get(
                            'key_minion_app_{}'.format(app_id)))
                        port = self._get_next_available_port()
                        self.active_minions[key] = {
                            'pid': minion_info.get('pid'), 'port': port}
                        log.info(
                            _('Minion {} joined (pid: {}, port: {}).').format(
                                app_id, minion_info.get('pid'), port))
        except KeyboardInterrupt:
            pass
        except ConnectionError as cx:
            log.exception(cx)
            time.sleep(1)

    def process(self):
        log.info(_('Juicer server started (pid=%s)'), os.getpid())
        self.start_process = multiprocessing.Process(
            name="master", target=self.start)
        self.start_process.daemon = False

        self.minion_support_process = multiprocessing.Process(
            name="help_desk", target=self.minion_support)
        self.minion_support_process.daemon = False

        self.new_minion_watch_process = multiprocessing.Process(
            name="minion_status", target=self.watch_new_minion)
        self.new_minion_watch_process.daemon = False

        self.start_process.start()
        self.minion_support_process.start()
        self.new_minion_watch_process.start()

        try:
            self.start_process.join()
            self.minion_support_process.join()
            self.new_minion_watch_process.join()
        except KeyboardInterrupt:
            self._terminate(None, None)

    # noinspection PyUnusedLocal
    def _terminate_minions(self, _signal, _frame):
        log.info(_('Terminating %s active minions'), len(self.active_minions))
        minions = [m for m in self.active_minions]
        for (wid, aid) in minions:
            self._terminate_minion(wid, aid)
        sys.exit(0)

    # noinspection PyUnusedLocal
    def _terminate(self, _signal, _frame):
        """
        This is a handler that reacts to a sigkill signal.
        """
        log.info(_('Killing juicer server subprocesses and terminating'))
        if self.start_process:
            os.kill(self.start_process.pid, signal.SIGTERM)
        if self.minion_support_process:
            os.kill(self.minion_support_process.pid, signal.SIGKILL)
        # if self.minion_watch_process:
        #     os.kill(self.minion_watch_process.pid, signal.SIGKILL)
        if self.new_minion_watch_process:
            os.kill(self.new_minion_watch_process.pid, signal.SIGKILL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file", required=True)
    parser.add_argument("--lang", help="Minion messages language (i18n)",
                        required=False, default="en_US")
    args = parser.parse_args()

    t = gettext.translation('messages', locales_path, [args.lang],
                            fallback=True)
    t.install(unicode=True)

    with open(args.config) as config_file:
        juicer_config = yaml.load(config_file.read())

    # Every minion starts with the same script.
    _minion_executable = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'minion.py'))

    server = JuicerServer(juicer_config, _minion_executable,
                          config_file_path=args.config)
    server.process()
