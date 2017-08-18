# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import logging.config;

import redis
import requests
import yaml
from juicer.runner import configuration
from juicer.spark.transpiler import SparkTranspiler
from juicer.workflow.workflow import Workflow
from juicer.workflow.workflow_webservice import WorkflowWebService

logging.config.fileConfig('logging_config.ini')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Statuses:
    def __init__(self):
        pass

    EMPTY = 'EMPTY'
    START = 'START'
    RUNNING = 'RUNNING'


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
        _id = 'status_{}'.format(self.workflow_id)
        # status = self.redis_conn.hgetall(_id)
        # print '>>>', status

        log.debug('Processing workflow queue %s', self.workflow_id)
        while True:
            # msg = self.redis_conn.brpop(str(self.workflow_id))

            # self.redis_conn.hset(_id, 'status', Statuses.RUNNING)
            tahiti_conf = self.config['juicer']['services']['tahiti']

            r = requests.get(
                "{url}/workflows/{id}?token={token}".format(id=self.workflow_id,
                                                            url=tahiti_conf[
                                                                'url'],
                                                            token=tahiti_conf[
                                                                'auth_token']))
            if r.status_code == 200:
                loader = Workflow(json.loads(r.text), self.config)
            else:
                print tahiti_conf['url'], r.text
                exit(-1)
            # FIXME: Implement validation
            # loader.verify_workflow()
            configuration.set_config(self.config)
            spark_instance = SparkTranspiler(configuration.get_config())
            self.params['execute_main'] = self.execute_main

            # generated = StringIO()
            # spark_instance.output = generated
            try:
                spark_instance.transpile(loader.workflow,
                                         loader.graph,
                                         params=self.params,
                                         job_id=self.job_id)
            except ValueError as ve:
                log.exception("At least one parameter is missing", exc_info=ve)
                break
            except:
                raise

            # WebService
            workflow_as_web_service = True

            if workflow_as_web_service:

                ###########
                # Get Params for Juicer WebServices - Generate from Workflow
                params_ws = {
                    'inputs': [
                        {'id': 18,
                         'operation_id': "603d19e6-c420-4116-85cc-2490cba18133"}
                    ],
                    'outputs': [
                        {'id': 42,
                         'operation_id': "082d2558-58eb-47c9-baeb-15b051bc256a"}
                    ],
                    'models': [
                        {'id': 1,
                         'operation_id': "ad0ed19f-b939-4074-8d97-c06643114a24"}
                    ]
                }

                # Lookup table - pre defined
                dict_lkt = {
                    1: 'Read Model',
                    18: 'WS Input',
                    42: 'WS Output',
                    26: 'WS Visualization'
                }
                try:
                    webservice_workflow_instance = WorkflowWebService(loader.workflow,
                                                                      loader.graph,
                                                                      params_ws,
                                                                      dict_lkt,
                                                                      configuration.get_config())

                    configuration.set_config(self.config)
                    spark_instance_2 = SparkTranspiler(configuration.get_config())
                    self.params['execute_main'] = self.execute_main

                    print webservice_workflow_instance.graph_ws.edges()
                    spark_instance_2.transpile(webservice_workflow_instance.workflow_ws,
                                               webservice_workflow_instance.graph_ws,
                                               params=self.params,
                                               job_id=self.job_id)
                except ValueError as ve:
                    log.exception("At least one parameter is missing", exc_info=ve)
                    break
                except:
                    raise

            # generated.seek(0)
            # print generated.read()
            # raw_input('Pressione ENTER')
            break


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
    parser.add_argument(
        "-p", "--plain", required=False, action="store_true",
        help="Indicates if workflow should be plain PySpark, "
             "without Lemonade extra code")
    args = parser.parse_args()

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
