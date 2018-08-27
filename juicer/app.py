# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import print_function

import argparse
import gettext
import json
import logging.config

import os
import redis
import requests
import yaml
from juicer.runner import configuration
from juicer.spark.transpiler import SparkTranspiler
from juicer.sklearn.transpiler import SklearnTranspiler
from juicer.compss.transpiler import COMPSsTranspiler
from juicer.workflow.workflow import Workflow

logging.config.fileConfig('logging_config.ini')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Statuses:
    def __init__(self):
        pass

    EMPTY = 'EMPTY'
    START = 'START'
    RUNNING = 'RUNNING'


# noinspection SpellCheckingInspection
class JuicerSparkService:
    def __init__(self, redis_conn, workflow_id, execute_main, params, job_id,
                 config):
        self.redis_conn = redis_conn
        self.config = config
        self.workflow_id = workflow_id
        self.state = "LOADING"
        self.params = params
        self.job_id = job_id
        self.execute_main = execute_main
        self.states = {
            "EMPTY": {
                "START": self.start
            },
            "START": {

            }
        }

    def start(self):
        pass

    def run(self):
        # _id = 'status_{}'.format(self.workflow_id)
        # status = self.redis_conn.hgetall(_id)
        # print '>>>', status

        log.debug(_('Processing workflow queue %s'), self.workflow_id)

        # msg = self.redis_conn.brpop(str(self.workflow_id))

        # self.redis_conn.hset(_id, 'status', Statuses.RUNNING)
        tahiti_conf = self.config['juicer']['services']['tahiti']

        r = requests.get("{url}/workflows/{id}?token={token}".format(
            id=self.workflow_id, url=tahiti_conf['url'],
            token=tahiti_conf['auth_token']))

        loader = None
        if r.status_code == 200:
            loader = Workflow(json.loads(r.text), self.config)
        else:
            print(tahiti_conf['url'], r.text)
            exit(-1)
        # FIXME: Implement validation
        # loader.verify_workflow()
        configuration.set_config(self.config)

        try:
            if loader.platform == "spark":
                transpiler = SparkTranspiler(configuration.get_config())
            elif loader.platform == "compss":
                transpiler = COMPSsTranspiler(configuration.get_config())
            elif loader.platform == "scikit-learn":
                transpiler = SklearnTranspiler(configuration.get_config())
            else:
                raise ValueError(
                    _('Invalid platform value: {}').format(loader.platform))

            self.params['execute_main'] = self.execute_main
            transpiler.execute_main = self.execute_main
            transpiler.transpile(loader.workflow,
                                 loader.graph,
                                 params=self.params,
                                 job_id=self.job_id)

        except ValueError as ve:
            log.exception(_("At least one parameter is missing"), exc_info=ve)
        except:
            raise


def main(workflow_id, execute_main, params, job_id, config):
    redis_conn = redis.StrictRedis()
    service = JuicerSparkService(redis_conn, workflow_id, execute_main, params,
                                 job_id, config)
    service.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False,
                        help="Configuration file")

    parser.add_argument("-w", "--workflow", type=int, required=True,
                        help="Workflow identification number")

    parser.add_argument("-j", "--job_id", type=int,
                        help="Job identification number")

    parser.add_argument("-e", "--execute-main", action="store_true",
                        help="Write code to run the program (it calls main()")

    parser.add_argument("-s", "--service", required=False,
                        action="store_true",
                        help="Indicates if workflow will run as a service")
    parser.add_argument("--lang", help="Minion messages language (i18n)",
                        required=False, default="en_US")
    parser.add_argument(
        "-p", "--plain", required=False, action="store_true",
        help="Indicates if workflow should be plain PySpark, "
             "without Lemonade extra code")
    args = parser.parse_args()

    locales_path = os.path.join(os.path.dirname(__file__), 'i18n', 'locales')
    t = gettext.translation('messages', locales_path, [args.lang],
                            fallback=True)
    t.install(unicode=True)

    juicer_config = {}
    if args.config:
        with open(args.config) as config_file:
            juicer_config = yaml.load(config_file.read())

    main(args.workflow, args.execute_main,
         {"service": args.service, "plain": args.plain},
         args.job_id, juicer_config)
    '''
    if True:
        app.run(debug=True, port=8000)
    else:
        wsgi.server(eventlet.listen(('', 8000)), app)
    '''
