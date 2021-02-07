# coding=utf-8


import datetime
import hashlib
import inspect
import jinja2
import json
import logging
import networkx as nx
import redis
import sys
import uuid
from collections import OrderedDict
from rq import Queue
from urllib.parse import urlparse
from textwrap import dedent

from juicer import auditing
from juicer.util.jinja2_custom import AutoPep8Extension
from .service import stand_service
from .util.template_util import HandleExceptionExtension

AUDITING_QUEUE_NAME = 'auditing'
AUDITING_JOB_NAME = 'seed.jobs.auditing'

log = logging.getLogger(__name__)


class DependencyController(object):
    """ Evaluates if a dependency is met when generating code. """

    def __init__(self, requires):
        self._satisfied = set()
        self.requires = requires

    def satisfied(self, _id):
        self._satisfied.add(_id)

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def is_satisfied(self, _id):
        return True  # len(self.requires[_id].difference(self._satisfied)) == 0


# noinspection PyMethodMayBeStatic
class Transpiler(object):
    """Base class for transpilers (converts workflow into platform specific
    code).
    """
    VISITORS = []
    DATA_SOURCE_OPS = ['data-reader']
    __slots__ = (
        'configuration', 'current_task_id', 'operations', 'port_id_to_port',
        'slug_to_op_id', 'template_dir', 'sample_size'
    )

    def __init__(self, configuration, template_dir, slug_to_op_id=None,
                 port_id_to_port=None):
        self.operations = {}
        if slug_to_op_id is None:
            self.slug_to_op_id = {}
        else:
            self.slug_to_op_id = slug_to_op_id
        if port_id_to_port is None:
            self.port_id_to_port = {}
        else:
            self.port_id_to_port = port_id_to_port
        self.operations = {}
        self._assign_operations()
        self.configuration = configuration
        self.template_dir = template_dir
        self.current_task_id = None
        self.sample_size = 50

    def _assign_operations(self):
        raise NotImplementedError()

    def get_context(self):
        return {}

    def get_code_template(self):
        return "templates/operation.tmpl"

    def get_notebook_template(self):
        return "templates/notebook.tmpl"

    def get_deploy_template(self):
        return "templates/deploy.tmpl"

    def get_plain_template(self):
        return "templates/plain.tmpl"

    def get_audit_info(self, graph, workflow, task, parameters):
        result = []
        task['ancestors'] = list(nx.ancestors(graph, task['id']))
        ancestors = [graph.node[task_id] for task_id in task['ancestors']]
        ancestors_data_source = [int(p['forms']['data_source'].get('value', 0))
                                 for p in ancestors if p['is_data_source']]

        # If it doesn't have a data source implies no auditing info generated
        if not ancestors_data_source:
            return result
        events = parameters['audit_events'] or []
        if parameters['display_sample']:
            events.append(auditing.DISPLAY_DATA)

        if parameters['display_schema']:
            events.append(auditing.DISPLAY_SCHEMA)

        for event in events:
            result.append({
                'module': 'JUICER',
                'platform_id': parameters['workflow']['platform']['id'],
                'event': event,
                'date': datetime.datetime.now(),
                'context': parameters['configuration']['juicer'].get(
                    'context', 'NOT_SET'),
                'data_sources': ancestors_data_source,
                'workflow': {
                    'id': workflow['id'],
                    'name': workflow['name'],
                },
                'job': {'id': parameters['job_id']},
                'task': {
                    'id': task['id'],
                    'name': task['operation']['name'],
                    'type': task['operation']['slug']
                },
                'user': workflow['user'],
            })
        return result

    def generate_code(self, graph, job_id, out, params, ports,
                      sorted_tasks_id, state, task_hash, using_stdout,
                      workflow, deploy=False, export_notebook=False,
                      plain=False, persist=True):

        if deploy:
            # To be able to convert, workflow must obey all these rules:
            # - 1 and exactly 1 data source;
            # - Data source must be defined in Limonero with its attributes in
            # order to define the schema for data input;
            # - For ML models, it is required to have a Save Model operation;
            total_ds = 0
            for task in workflow['tasks']:
                if not task.get('enabled', False):
                    continue
                if task['operation']['slug'] in self.DATA_SOURCE_OPS:
                    total_ds += 1

            if total_ds < 1:
                raise ValueError(_(
                    'Workflow must have at least 1 data source to be deployed.')
                )
            tasks_ids = reversed(sorted_tasks_id)
        else:
            tasks_ids = sorted_tasks_id

        instances = OrderedDict()
        transpiler_utils = TranspilerUtils(self)

        audit_events = []
        for i, task_id in enumerate(tasks_ids):
            task = graph.node[task_id]['attr_dict']
            task['parents'] = graph.node[task_id]['parents']
            self.current_task_id = task_id

            class_name = self.operations[task['operation']['slug']]

            parameters = {}
            not_empty_params = [(k, d) for k, d in
                                list(task['forms'].items()) if d['value']]

            task['forms'] = dict(not_empty_params)
            for parameter, definition in list(task['forms'].items()):
                # @FIXME: Fix wrong name of form category
                # (using name instead of category)
                cat = definition.get('category',
                                     'execution').lower()  # FIXME!!!
                cat = 'paramgrid' if cat == 'param grid' else cat
                cat = 'logging' if cat == 'execution logging' else cat

                if all([cat in ["execution", 'paramgrid', 'param grid',
                                'execution logging', 'logging', 'save',
                                'transformation'],
                        definition['value'] is not None]):

                    task_hash.update(str(definition['value']).encode(
                        'utf8', errors='ignore'))
                    if cat in ['paramgrid', 'logging']:
                        if cat not in parameters:
                            parameters[cat] = {}
                        parameters[cat][parameter] = definition['value']
                    else:
                        parameters[parameter] = definition['value']
                # escape invalid characters for code generation
                # except JSON (starting with {)
                if definition['value'] is not None and not isinstance(
                        definition['value'], bool) and not isinstance(
                    definition['value'], int):
                    if '"' in definition['value'] or "'" in definition['value']:
                        if definition['value'][0] != '{':
                            definition['value'] = TranspilerUtils.escape_chars(
                                definition['value'])

            if state is None or state.get(task_id) is None:
                parameters['execution_date'] = None
            else:
                gen_source_code = state.get(task_id, [{}])[0]
                if gen_source_code:
                    parameters['execution_date'] = gen_source_code.get(
                        'execution_date')
                else:
                    parameters['execution_date'] = None
            true_values = (1, '1', True, 'true', 'True')
            parameters.update({
                'configuration': self.configuration,
                'display_sample': task['forms'].get('display_sample', {}).get(
                    'value') in true_values,
                'display_schema': task['forms'].get('display_schema', {}).get(
                    'value') in true_values,
                # Hash is used in order to avoid re-run task.
                'export_notebook': export_notebook,
                'hash': task_hash.hexdigest(),
                'job_id': job_id,
                'operation_id': task['operation']['id'],
                'operation_slug': task['operation']['slug'],
                # Some temporary variables need to be identified by a sequential
                # number, so it will be stored in this field
                'order': i,
                'task': task,
                'task_id': task['id'],
                'transpiler': self,  # Allows operation to notify transpiler
                'transpiler_utils': transpiler_utils,
                'plain': plain,
                'user': workflow['user'],
                'workflow': workflow,
                'workflow_id': workflow['id'],
                # Some operations require the complete workflow data
                'workflow_name': TranspilerUtils.escape_chars(workflow['name']),
            })
            port = ports.get(task['id'], {})
            parameters['parents'] = port.get('parents', [])
            parameters['parents_slug'] = port.get('parents_slug', [])
            parameters['parents_by_port'] = port.get('parents_by_port', [])
            parameters['my_ports'] = port.get('my_ports', [])

            # print task['name'], parameters['parents'] # port.get('parents', [])

            instance = class_name(parameters, port.get('named_inputs', {}),
                                  port.get('named_outputs', {}))

            graph.node[task['id']]['is_data_source'] = instance.is_data_source
            parameters['audit_events'] = instance.get_audit_events()

            if self.configuration['juicer'].get('auditing', False):
                audit_events.extend(self.get_audit_info(graph, workflow, task,
                                                        parameters))

            instance.out_degree = graph.out_degree(task_id)
            instances[task['id']] = instance

        if audit_events:

            redis_url = self.configuration['juicer']['servers']['redis_url']
            parsed = urlparse(redis_url)
            redis_conn = redis.Redis(host=parsed.hostname,
                                     port=parsed.port)
            q = Queue(AUDITING_QUEUE_NAME, connection=redis_conn)
            for event in audit_events:
                event['date'] = event['date'].isoformat()
            q.enqueue(AUDITING_JOB_NAME, json.dumps(audit_events))

        # adding information about the parents's multiplicity
        for task_id in instances:
            instances[task_id].parameters['multiplicity'] = dict()
            for p_id in instances[task_id].parameters['task']['parents']:
                for flow in workflow['flows']:
                    if flow['target_id'] == task_id and \
                            flow['source_id'] == p_id:
                        in_port = flow['target_port_name']
                        source_port = flow['source_port']
                        instances[task_id].parameters['multiplicity'][
                            in_port] = sum([1 for f in workflow['flows']
                                            if f['source_port'] == source_port])

        env_setup = {
            'dependency_controller': DependencyController(
                params.get('requires_info', False)),
            'disabled_tasks': workflow['disabled_tasks'],
            'execute_main': params.get('execute_main', False),
            'instances': list(instances.values()),
            'instances_by_task_id': instances,
            'job_id': job_id,
            'now': datetime.datetime.now(), 'user': workflow['user'],
            'plain': plain,
            'export_notebook': export_notebook,
            'transpiler': transpiler_utils,
            'workflow_name': workflow['name'],
            'workflow': workflow,
        }
        env_setup.update(self.get_context())

        template_loader = jinja2.FileSystemLoader(
            searchpath=self.template_dir)
        template_env = jinja2.Environment(loader=template_loader,
                                          extensions=[AutoPep8Extension,
                                                      HandleExceptionExtension,
                                                      'jinja2.ext.do'])
        template_env.globals.update(zip=zip)

        if deploy:
            env_setup['slug_to_op_id'] = self.slug_to_op_id
            # env_setup['slug_to_port_id'] = self.slug_to_port_id
            env_setup['id_mapping'] = {}
            template = template_env.get_template(self.get_deploy_template())
            out.write(template.render(env_setup))
        elif export_notebook:
            template = template_env.get_template(self.get_notebook_template())
            out.write(template.render(env_setup))
        elif plain:
            template = template_env.get_template(self.get_plain_template())
            out.write(template.render(env_setup))
        else:
            template = template_env.get_template(self.get_code_template())
            gen_source_code = template.render(env_setup)
            if using_stdout:
                out.write(gen_source_code)
            else:
                out.write(gen_source_code)
            stand_config = self.configuration.get('juicer', {}).get(
                'services', {}).get('stand')
            if stand_config and job_id and persist:
                # noinspection PyBroadException
                try:
                    stand_service.save_job_source_code(
                        stand_config['url'], stand_config['auth_token'], job_id,
                        gen_source_code)
                except Exception as ex:
                    log.exception(str(ex))

    def transpile(self, workflow, graph, params, out=None, job_id=None,
                  state=None, deploy=False, export_notebook=False, plain=False, 
                  persist=True):
        """ Transpile the tasks from Lemonade's workflow into code """

        using_stdout = out is None
        if using_stdout:
            out = sys.stdout
        ports = {}
        sequential_ports = {}
        counter = 0
        for edge_key in list(graph.edges.keys()):
            source_id, target_id, index = edge_key
            source_name = graph.node[source_id]['name']
            source_slug = graph.node[source_id]['operation']['slug']
            flow = graph.edges[edge_key]['attr_dict']
            flow_id = '[{}:{}]'.format(source_id, flow['source_port'], )

            if flow_id not in sequential_ports:
                sequential_ports[flow_id] = \
                    TranspilerUtils.gen_port_name(flow, counter)
                counter += 1
            if source_id not in ports:
                ports[source_id] = {'outputs': [], 'inputs': [],
                                    'parents': [],
                                    'parents_slug': [],
                                    'parents_by_port': [],
                                    'my_ports': [],
                                    'named_inputs': {},
                                    'named_outputs': {}}
            if target_id not in ports:
                ports[target_id] = {'outputs': [], 'inputs': [],
                                    'parents': [],
                                    'parents_by_port': [],
                                    'my_ports': [],
                                    'parents_slug': [],
                                    'named_inputs': {},
                                    'named_outputs': {}}
            ports[target_id]['parents'].append(source_name)
            ports[target_id]['parents_slug'].append(source_slug)
            ports[target_id]['parents_by_port'].append(
                (flow['source_port_name'], source_name))
            ports[target_id]['my_ports'].append(
                (flow['target_port_name'], source_name))
            sequence = sequential_ports[flow_id]

            source_port = ports[source_id]
            if sequence not in source_port['outputs']:
                source_port['named_outputs'][
                    flow['source_port_name']] = sequence
                source_port['outputs'].append(sequence)

            target_port = ports[target_id]
            if sequence not in target_port['inputs']:
                flow_name = flow['target_port_name']
                # Test if multiple inputs connects to a port
                # because it may have multiplicity MANY
                if flow_name in target_port['named_inputs']:
                    if not isinstance(
                            target_port['named_inputs'][flow_name],
                            list):
                        target_port['named_inputs'][flow_name] = [
                            target_port['named_inputs'][flow_name],
                            sequence]
                    else:
                        target_port['named_inputs'][flow_name].append(
                            sequence)
                else:
                    target_port['named_inputs'][flow_name] = sequence
                target_port['inputs'].append(sequence)

        self.generate_code(graph, job_id, out, params,
                           ports, nx.topological_sort(graph), state,
                           hashlib.sha1(),
                           using_stdout, workflow, deploy, export_notebook,
                           plain=plain,
                           persist=persist)

    def get_data_sources(self, workflow):
        return len(
            [t['slug'] in self.DATA_SOURCE_OPS for t in workflow['tasks']]) == 1

    def generate_auxiliary_code(self):
        return ""


class TranspilerUtils(object):
    """ Utilities for using in Jinja2 related to transpiling and other useful
     functions.
     """

    def __init__(self, transpiler=None):
        self.transpiler = transpiler
        self.imports = set()
        self.custom_functions = dict()
        if transpiler:
            self.sample_size = transpiler.sample_size

    @staticmethod
    def _get_enabled_tasks_to_execute(instances):
        dependency_controller = DependencyController([])
        result = []
        for instance in TranspilerUtils._get_enabled_tasks(instances):
            task = instance.parameters['task']
            is_satisfied = dependency_controller.is_satisfied(task['id'])
            if instance.must_be_executed(is_satisfied):
                result.append(instance)
        return result

    @staticmethod
    def _get_enabled_tasks(instances):
        return [instance for instance in instances if
                instance.has_code and instance.enabled]

    @staticmethod
    def get_auxiliary_code(instances):
        result = []
        for instance in instances:
            result.extend(instance.get_auxiliary_code())
        return set(result)

    @staticmethod
    def _get_parent_tasks(instances_map, instance, only_enabled=True,
                          only_direct=True):
        if only_enabled:
            result = []
            if only_direct:
                parents = instance.parameters['task']['parents']
            else:
                parents = instance.parameters['task']['ancestors']
            for parent_id in parents:
                parent = instances_map[parent_id]
                if parent.has_code and parent.enabled:
                    method = '{}_{}'.format(
                        parent.parameters['task']['operation']['slug'].replace(
                            '-', '_'), parent.order)
                    result.append((parent_id, method))
            return result
        else:
            return [instances_map[parent_id] for parent_id in
                    instance.parameters['task']['parents']]

    @staticmethod
    def get_imports(instances):
        # "from keras.layers import "
        layer_import = "from keras.layers import "
        layer_list = []
        callbacks_import = "from keras.callbacks import "
        callbacks_list = []
        model_import = "from keras.models import "
        model_list = []
        preprocessing_image_import = "from keras.preprocessing.image import "
        preprocessing_image_list = []
        others_import = ""
        others_list = []

        for instance in instances:
            if instance.import_code:
                if instance.import_code['layer']:
                    if not instance.import_code['layer'] in layer_list:
                        layer_list.append(instance.import_code['layer'])
                if instance.import_code['callbacks']:
                    for callback in instance.import_code['callbacks']:
                        if callback not in callbacks_list:
                            callbacks_list.append(callback)
                if instance.import_code['model']:
                    if not instance.import_code['model'] in model_list:
                        model_list.append(instance.import_code['model'])
                if instance.import_code['preprocessing_image']:
                    if not instance.import_code[
                               'preprocessing_image'] in preprocessing_image_list:
                        preprocessing_image_list.append(
                            instance.import_code['preprocessing_image'])
                if instance.import_code['others']:
                    for other in instance.import_code['others']:
                        if other not in others_list:
                            others_list.append(other)

        imports = ""
        if len(layer_list) > 0:
            imports += layer_import + ', '.join(layer_list) + '\n'
        if len(callbacks_list) > 0:
            imports += callbacks_import + ', '.join(callbacks_list) + '\n'
        if len(model_list) > 0:
            imports += model_import + ', '.join(model_list) + '\n'
        if len(preprocessing_image_list) > 0:
            imports += preprocessing_image_import + ', '.join(
                preprocessing_image_list) + '\n'
        if len(others_list) > 0:
            imports += others_import + '\n'.join(others_list)

        imports = imports.replace(' , ', ', ')

        return imports

    @staticmethod
    def get_ids_and_methods(instances):
        result = OrderedDict()
        for instance in TranspilerUtils._get_enabled_tasks_to_execute(
                instances):
            task = instance.parameters['task']
            result[task['id']] = '{}_{}'.format(
                task['operation']['slug'].replace('-', '_'), instance.order)
        return result

    @staticmethod
    def get_disabled_tasks(instances):
        return [instance for instance in instances if
                not instance.has_code or not instance.enabled]

    @staticmethod
    def get_new_task_id():
        return uuid.uuid1()

    @staticmethod
    def escape_chars(text):
        return text.replace('"', '\\"').replace("'", "\\'").encode(
            'unicode-escape').decode('utf-8')

    @staticmethod
    def gen_port_name(flow, seq):
        name = flow.get('source_port_name', 'data')
        parts = name.split()
        if len(parts) == 1:
            name = name[:5]
        elif name[:3] == 'out':
            name = 'var_'  # name[:3]
        else:
            name = ''.join([p[0] for p in parts])
        return '{}{}'.format(name, seq)

    def add_import(self, name):
        """ Add an import to the generated code. More than one operation may add
        the same import. This method handles it, by removing duplicates.
        """
        self.imports.add(name)

    def add_custom_function(self, name, f):
        """ Add a custom function to the generated code. More than one operation
        may add same function. This method handles it, by removing duplicates.
        """
        code = inspect.getsource(f)
        self.custom_functions[name] = dedent(code)

    def __unicode__(self):
        return 'TranspilerUtils object'

    def render_template(self, template: str, context: dict):
        """
        Render a Jinja2 template using the information provided by context.
        """
        tm = jinja2.Template(template)
        return tm.render(**context)
