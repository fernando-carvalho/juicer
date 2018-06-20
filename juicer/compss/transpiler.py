# -*- coding: utf-8 -*-
import json
import sys

import jinja2
import juicer.compss.data_operation
import juicer.compss.etl_operation
import juicer.compss.geo_operation
import juicer.compss.graph_operation
import juicer.compss.associative_operation
import juicer.compss.feature_operation
import juicer.compss.model_operation
import juicer.compss.text_operation
import juicer.compss.classification_operation
import juicer.compss.regression_operation
import juicer.compss.clustering_operation
import juicer.compss.optimizated_operation

import networkx as nx
import os
from juicer import operation
from juicer.service import stand_service
from juicer.util.jinja2_custom import AutoPep8Extension
from juicer.util.template_util import HandleExceptionExtension

from optimizated_operation import OptimizatedOperation


class DependencyController:
    """ Evaluates if a dependency is met when generating code. """

    def __init__(self, requires):
        self._satisfied = set()
        self.requires = requires

    def satisfied(self, _id):
        self._satisfied.add(_id)

    @staticmethod
    def is_satisfied(_id):
        return True  # len(self.requires[_id].difference(self._satisfied)) == 0


# noinspection SpellCheckingInspection
class COMPSsTranspiler(object):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    COMPSs.
    """

    def __init__(self, configuration):
        self.configuration = configuration
        self.operations = {}
        self._assign_operations()
        self.numFrag = self.configuration.get('juicer', {}).get(
            'compss', {}).get('numFrag')
        self.enable_optimization = self.configuration.get('juicer', {}).get(
            'compss', {}).get('optimization')

        # self.graph = graph
        # self.params = params if params is not None else {}
        #
        # self.using_stdout = out is None
        # if self.using_stdout:
        #     self.out = sys.stdout
        # else:
        #     self.out = out
        #
        # self.job_id = job_id
        # workflow_json = json.dumps(workflow)
        # workflow_name = workflow['name']
        # workflow_id = workflow['id']
        # workflow_user = workflow.get('user', {})

        self.execute_main = False

    def get_optimization_information(self, id_task, task, workflow, ports):
        class_name = self.operations[id_task]

        if getattr(class_name, "get_optimization_information", None):
            parameters = self.generate_parameters(task, workflow, 0)
            port = ports.get(task['id'], {})
            instance = class_name(parameters,
                                  port.get('named_inputs', {}),
                                  port.get('named_outputs', {}))
            return instance.get_optimization_information()
        return dict()

    def generate_parameters(self, task, workflow, i):
        parameters = {}

        for parameter, definition in task['forms'].iteritems():
            # @FIXME: Fix wrong name of form category
            # (using name instead of category)
            # print definition.get('category')
            # raw_input()
            cat = definition.get('category',
                                 'execution').lower()  # FIXME!!!
            cat = 'paramgrid' if cat == 'param grid' else cat
            cat = 'logging' if cat == 'execution logging' else cat

            if all([cat in ["execution", 'paramgrid', 'param grid',
                            'execution logging', 'logging'],
                    definition['value'] is not None]):

                if cat in ['paramgrid', 'logging']:
                    if cat not in parameters:
                        parameters[cat] = {}
                    parameters[cat][parameter] = definition['value']
                else:
                    parameters[parameter] = definition['value']

        # Operation SAVE requires the complete workflow
        if task['operation']['name'] == 'SAVE':
            parameters['workflow'] = workflow

        parameters['workflow_json'] = json.dumps(workflow)
        parameters['user'] = workflow['user']
        parameters['workflow_id'] = workflow['id']
        parameters['workflow_name'] = workflow['name']
        # Some temporary variables need to be identified by a sequential
        # number, so it will be stored in this field
        task['order'] = i

        parameters['task'] = task
        parameters['task_id'] = task['id']
        parameters['operation_id'] = task['operation']['id']
        parameters['operation_slug'] = task['operation']['slug']

        return parameters

    def check_optimization(self, sorted_tasks_id, graph, workflow, port):
        """Check optimizations."""

        otm_candidate = {}
        new_sorted_tasks_id = []
        map_candidates = {}

        for i, task_id in enumerate(sorted_tasks_id):
            task_source = graph.node[task_id]
            target = graph.edge[task_id]
            has_single_edge = len(target.keys()) == 1
            source_candidate = task_source['operation']['slug']
            source_info = self.get_optimization_information(
                    source_candidate, task_source, workflow, port)
            # print 'source_candidate', source_candidate

            s_one_stage = source_info.get('one_stage', False)
            s_bifurcation = source_info.get('bifurcation', False)
            s_first = source_info.get('if_first', False)
            s_need_keep = source_info.get("need_keeped_data", False)
            s_keep = source_info.get("keep_balanced", False)
            s_apply_model = source_info.get("apply_model", False)

            # Optimize only if source has a single edge
            # Optimize = merge with the next function
            if not has_single_edge:
                # add to sorted list if if task is not already in one group
                if not task_id in map_candidates:
                    new_sorted_tasks_id.append(task_id)
            else:
                target = target[target.keys()[0]][0]['target_id']
                target_candidate = graph.node[target]['operation']['slug']
                task_target = graph.node[target]
                target_info = self.get_optimization_information(
                        target_candidate, task_target, workflow, port)

                t_one_stage = target_info.get('one_stage', False)
                t_bifurcation = target_info.get('bifurcation', False)
                t_need_keep = target_info.get("need_keeped_data", False)

                # print 'target_candidate', target_candidate
                conditions = []
                # one_stage --> one_stage
                cond = s_one_stage and t_one_stage and \
                    not (s_bifurcation or t_bifurcation)
                conditions.append(cond)

                # keep_balanced --> SampleOperation
                # or
                # sample at begin of an optimization
                cond = (s_keep and t_need_keep) or \
                       (s_need_keep and t_one_stage)
                conditions.append(cond)

                # when the box is the first of its group
                cond = (s_first and source_candidate not in map_candidates) \
                    and t_one_stage
                conditions.append(cond)

                # apply_model --> one_stage
                cond = s_apply_model and t_one_stage
                conditions.append(cond)

                # print "conditions::", conditions

                if any(conditions):
                    # add the task in the its existent group
                    found = task_id in map_candidates
                    if found:
                        id_group = map_candidates[task_id]
                        otm_candidate[id_group].append(target)
                        map_candidates[task_id] = id_group
                        map_candidates[target] = id_group

                    # if does not have, create a new one
                    if not found:
                        new_sorted_tasks_id.append(task_id)
                        otm_candidate[task_id] = [task_id, target]
                        map_candidates[task_id] = task_id
                        map_candidates[target] = task_id

                else:
                    # if cant be merged, check if the source is already merged
                    # with a previous function, if not, add in the sorted list
                    found = False
                    for k, v in otm_candidate.items():
                        if task_id in v:
                            found = True

                            break
                    if not found:
                        new_sorted_tasks_id.append(task_id)

        # print "*" * 20
        # print "OTM Groups:"
        # for k in otm_candidate:
        #     print "{}: {}".format(k, otm_candidate[k])
        # print "N of tasks w/o optimization: ", len(sorted_tasks_id)
        # print "N of tasks w optimization: ", len(new_sorted_tasks_id)
        # print "sorted_tasks_id: ", new_sorted_tasks_id
        # print "MAP_Groups:"
        # for k in map_candidates:
        #     print "{} --> {}".format(k, map_candidates[k])
        # print "*" * 20

        return otm_candidate, new_sorted_tasks_id, map_candidates

    def update_parents(self, parameters, conv_parents):
        old_parents = parameters['task']['parents']
        for i, old_one in enumerate(old_parents):
            if old_one in conv_parents:
                parameters['task']['parents'][i] = conv_parents[old_one]

        return parameters

    def transpile(self, workflow, graph, params, out=None, job_id=None):
        """ Transpile the tasks from Lemonade's workflow into COMPSs code """

        ports = {}
        sequential_ports = {}

        for source_id in graph.edge:
            for target_id in graph.edge[source_id]:
                # Nodes accept multiple edges from same source
                for flow in graph.edge[source_id][target_id].values():
                    flow_id = '[{}:{}]'.format(source_id, flow['source_port'])

                    if flow_id not in sequential_ports:
                        sequential_ports[flow_id] = 'data{}'.format(
                            len(sequential_ports))

                    if source_id not in ports:
                        ports[source_id] = {'outputs': [], 'inputs': [],
                                            'named_inputs': {},
                                            'named_outputs': {}}
                    if target_id not in ports:
                        ports[target_id] = {'outputs': [], 'inputs': [],
                                            'named_inputs': {},
                                            'named_outputs': {}}

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
                                target_port['named_inputs'][flow_name] = \
                                    [target_port['named_inputs'][flow_name],
                                     sequence]
                            else:
                                target_port['named_inputs'][flow_name].append(
                                    sequence)
                        else:
                            target_port['named_inputs'][flow_name] = sequence
                        target_port['inputs'].append(sequence)

        sorted_tasks_id = nx.topological_sort(graph)

        env_setup = {'instances': [], 'instances_by_task_id': {},
                     'workflow_name': workflow['name']}

        imports = []
        if self.enable_optimization:
            # print "Optimization Enabled"
            otm, sorted_tasks_id, conversion_parents = \
                self.check_optimization(sorted_tasks_id, graph, workflow, ports)
        else:
            conversion_parents = {}
            otm = {}

        for i, task_id in enumerate(sorted_tasks_id):
            task = graph.node[task_id]

            if task_id in otm:
                otm_group = otm[task_id]

                code_0 = ""
                code_1 = []
                parameters = dict()
                for j, task_idd in enumerate(otm_group):
                    task = graph.node[task_idd]

                    port = ports.get(task['id'], {})
                    parameters = self.generate_parameters(task, workflow, j)
                    parameters['configuration'] = self.configuration
                    parameters['job_id'] = job_id
                    parameters['numFrag'] = self.numFrag

                    if j == 0:
                        first_task_id_group = parameters['task_id']
                        first_operation_group = parameters['operation_id']
                        first_slug_group = 'otm'
                        first_id = i
                        first_parents = task['parents']
                        first_slug = parameters['operation_slug']
                        first_task = task
                        first_port_input = port.get('named_inputs', {})

                    class_name = self.operations[task['operation']['slug']]
                    instance = class_name(parameters,
                                          port.get('named_inputs', {}),
                                          port.get('named_outputs', {}))

                    code_0 += instance.generate_preoptimization_code()
                    code_1.append(instance.generate_optimization_code())
                    if instance.has_import not in imports:
                        imports.append(instance.has_import)

                parameters['task'] = first_task
                parameters['first_slug'] = first_slug
                parameters['task']['order'] = first_id
                parameters['task']['operation']['slug'] = first_slug_group
                parameters['task']['id'] = first_task_id_group
                parameters['task']['parents'] = first_parents
                parameters['task_id'] = first_task_id_group
                parameters['operation_id'] = first_operation_group
                parameters['operation_slug'] = first_slug_group
                parameters['code_0'] = code_0
                parameters['code_1'] = code_1
                parameters['number_tasks'] = len(otm_group)
                parameters['fist_id'] = i
                task['id'] = first_task_id_group

                parameters = self.update_parents(parameters, conversion_parents)
                instance = OptimizatedOperation(parameters,
                                                first_port_input,
                                                port.get('named_outputs', {}))

                env_setup['dependency_controller'] = DependencyController(
                         params.get('requires_info', False))
                env_setup['instances'].append(instance)
                env_setup['instances_by_task_id'][task['id']] = instance
                env_setup['execute_main'] = params.get('execute_main', False)
                env_setup['plain'] = params.get('plain', False)
                env_setup['imports_list'] = imports

            else:
                port = ports.get(task['id'], {})

                parameters = self.generate_parameters(task, workflow, i)
                parameters['configuration'] = self.configuration
                parameters['job_id'] = job_id
                parameters['numFrag'] = self.numFrag

                class_name = self.operations[task['operation']['slug']]

                parameters = self.update_parents(parameters, conversion_parents)
                instance = class_name(parameters,
                                      port.get('named_inputs', {}),
                                      port.get('named_outputs', {}))

                env_setup['dependency_controller'] = DependencyController(
                    params.get('requires_info', False))

                env_setup['instances'].append(instance)
                env_setup['instances_by_task_id'][task['id']] = instance

            env_setup['execute_main'] = params.get('execute_main', False)
            env_setup['plain'] = params.get('plain', False)

            dict_msgs = dict()
            dict_msgs['task_completed'] = _('Task completed')
            dict_msgs['task_running'] = _('Task running')
            dict_msgs['lemonade_task_completed'] = \
                _('Lemonade task %s completed')
            dict_msgs['lemonade_task_parents'] = \
                _('Parents completed, submitting %s')
            dict_msgs['lemonade_task_started'] = \
                _('Lemonade task %s started')
            dict_msgs['lemonade_task_afterbefore'] = \
                _("Submitting parent task {} before {}")

            env_setup['dict_msgs'] = dict_msgs
            env_setup['numFrag'] = self.numFrag
            env_setup['imports'] = []

        template_loader = jinja2.FileSystemLoader(
            searchpath=os.path.dirname(__file__))
        template_env = jinja2.Environment(
            loader=template_loader, extensions=[AutoPep8Extension,
                                                HandleExceptionExtension])
        template_env.globals.update(zip=zip)
        template = template_env.get_template("operation.tmpl")

        v = template.render(env_setup)

        if out is None:
            sys.stdout.write(v.encode('utf8'))
        else:
            out.write(v)

        stand_config = self.configuration.get('juicer', {}).get(
            'services', {}).get('stand')
        if stand_config and job_id:
            # noinspection PyBroadException
            try:
                stand_service.save_job_source_code(stand_config['url'],
                                                   stand_config['auth_token'],
                                                   job_id, v.encode('utf8'))
            except:
                pass

    def _assign_operations(self):
        etl_ops = {
            'add-columns':
                juicer.compss.etl_operation.AddColumnsOperation,
            'add-rows':
                juicer.compss.etl_operation.UnionOperation,
            'aggregation':
                juicer.compss.etl_operation.AggregationOperation,
            'clean-missing':
                juicer.compss.etl_operation.CleanMissingOperation,
            'difference':
                juicer.compss.etl_operation.DifferenceOperation,
            'drop':
                juicer.compss.etl_operation.DropOperation,
            'filter-selection':
                juicer.compss.etl_operation.FilterOperation,
            'join':
                juicer.compss.etl_operation.JoinOperation,
            'projection':
                juicer.compss.etl_operation.SelectOperation,
            'remove-duplicated-rows':
                juicer.compss.etl_operation.DistinctOperation,
            'replace-value':
                juicer.compss.etl_operation.ReplaceValuesOperation,
            'sample':
                juicer.compss.etl_operation.SampleOrPartition,
            'set-intersection':
                juicer.compss.etl_operation.Intersection,
            'sort':
                juicer.compss.etl_operation.SortOperation,
            'split':
                juicer.compss.etl_operation.SplitOperation,
            'transformation':
                juicer.compss.etl_operation.TransformationOperation,

        }

        data_ops = {
            'data-reader':
                juicer.compss.data_operation.DataReaderOperation,
            'data-writer':
                juicer.compss.data_operation.SaveOperation,
            'save':
                juicer.compss.data_operation.SaveOperation,
            'balance-data':
                juicer.compss.data_operation.WorkloadBalancerOperation,
            'change-attribute':
                juicer.compss.data_operation.ChangeAttributesOperation,
        }

        geo_ops = {
            'read-shapefile':
                juicer.compss.geo_operation.ReadShapefileOperation,
            'within':
                juicer.compss.geo_operation.GeoWithinOperation,
            'stdbscan':
                juicer.compss.geo_operation.STDBSCANOperation,
        }

        graph_ops = {
            'page-rank':
                juicer.compss.graph_operation.PageRankOperation,
        }

        ml_ops = {
            # ------ Associative ------#
            'frequent-item-set':
                juicer.compss.associative_operation.AprioriOperation,
            'association-rules':
                juicer.compss.associative_operation.AssociationRulesOperation,

            # ------ Feature Extraction Operations  ------#

            'feature-assembler':
                juicer.compss.feature_operation.FeatureAssemblerOperation,
            'feature-indexer':
                juicer.compss.feature_operation.StringIndexerOperation,
            'pca':
                juicer.compss.feature_operation.PCAOperation,
            'max-abs-scaler':
                juicer.compss.feature_operation.MaxAbsScalerOperation,
            'min-max-scaler':
                juicer.compss.feature_operation.MinMaxScalerOperation,
            'standard-scaler':
                juicer.compss.feature_operation.StandardScalerOperation,


            # ------ Model Operations  ------#
            'apply-model':
                juicer.compss.model_operation.ApplyModel,
            'evaluate-model':
                juicer.compss.model_operation.EvaluateModelOperation,
            'load-model':
                juicer.compss.model_operation.LoadModelOperation,
            'save-model':
                juicer.compss.model_operation.SaveModelOperation,

            # ------ Clustering      -----#
            'clustering-model':
                juicer.compss.clustering_operation.ClusteringModelOperation,
            'k-means-clustering':
                juicer.compss.clustering_operation.KMeansClusteringOperation,
            'dbscan-clustering':
                juicer.compss.clustering_operation.DBSCANClusteringOperation,

            # ------ Classification  -----#
            'classification-model':
                juicer.compss.classification_operation.ClassificationModelOperation,
            'knn-classifier':
                juicer.compss.classification_operation.KNNClassifierOperation,
            'logistic-regression':
                juicer.compss.classification_operation.LogisticRegressionOperation,
            'naive-bayes-classifier':
                juicer.compss.classification_operation.NaiveBayesClassifierOperation,
            'svm-classification':
                juicer.compss.classification_operation.SvmClassifierOperation,


            # ------ Regression  -----#
            'regression-model':
                juicer.compss.regression_operation.RegressionModelOperation,
            'linear-regression':
                juicer.compss.regression_operation.LinearRegressionOperation,

        }

        text_ops = {
            'remove-stop-words':
                juicer.compss.text_operation.RemoveStopWordsOperation,
            'tokenizer': juicer.compss.text_operation.TokenizerOperation,
            'word-to-vector': juicer.compss.text_operation.WordToVectorOperation
        }

        other_ops = {
            'comment': operation.NoOp,
        }

        ws_ops = {}
        vis_ops = {}

        self.operations = {}
        for ops in [data_ops, etl_ops, geo_ops, graph_ops, ml_ops,
                    other_ops, text_ops, ws_ops, vis_ops]:
            self.operations.update(ops)
