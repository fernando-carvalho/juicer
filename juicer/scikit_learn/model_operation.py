from textwrap import dedent
import uuid
from jinja2 import Environment, BaseLoader
from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.service.limonero_service import query_limonero

from juicer import auditing

try:
    from urllib.request import urlopen
    from urllib.parse import urlparse, parse_qs
except ImportError:
    from urllib.parse import urlparse, parse_qs
    from urllib.request import urlopen

import string
from .util import get_X_train_data, get_label_data


class SafeDict(dict):
    # noinspection PyMethodMayBeStatic
    def __missing__(self, key):
        return '{' + key + '}'


class AlgorithmOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs,
                 model, algorithm):
        super(AlgorithmOperation, self).\
            __init__(parameters, named_inputs, named_outputs)
        self.algorithm = algorithm
        self.model = model
        self.has_code = len(self.named_inputs) and any(
            [len(self.named_outputs) > 0, self.contains_results()])

    def generate_code(self):
        if self.has_code:
            algorithm_code = self.algorithm.generate_code() or ''
            model_code = self.model.generate_code() or ''
            return "\n".join([algorithm_code, model_code])

    def get_output_names(self, sep=','):
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))
        models = self.named_outputs.get('model',
                                        'model_task_{}'.format(self.order))
        return sep.join([output, models])


class ApplyModelOperation(Operation):
    FEATURES_PARAM = 'features'
    PREDICTION_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = len(named_inputs) == 2 and any(
            [len(self.named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.output = self.named_outputs.get(
                    'output data', 'out_data_{}'.format(self.order))

            self.model = self.named_inputs.get(
                'model', 'model_{}'.format(self.order))

            self.prediction = self.parameters.get(self.PREDICTION_PARAM,
                                                  self.PREDICTION_PARAM)

            if self.FEATURES_PARAM not in self.parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM, self.__class__.__name__))
            self.features = self.parameters.get(self.FEATURES_PARAM)
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', f=get_X_train_data)
        else:
            if len(self.named_outputs) > 0:
                raise ValueError(
                    _('Model is being used, but at least one input is missing'))

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            code = dedent("""
                {out} = {in1}{copy_code}
                X_train = get_X_train_data({in1}, {features})
                if hasattr({in2}, 'predict'):
                    {out}['{new_attr}'] = {in2}.predict(X_train).tolist()
                else:
                    # to handle scaler operations
                    {out}['{new_attr}'] = {in2}.transform(X_train).tolist()
                """.format(copy_code=copy_code, out=self.output,
                           in1=self.named_inputs['input data'],
                           in2=self.model, new_attr=self.prediction,
                           features=self.features))

            return dedent(code)


class LoadModel(Operation):
    """LoadModel.
    """
    MODEL_PARAM = 'model'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.parameters = parameters

        self.model = parameters.get(self.MODEL_PARAM)
        if not self.model:
            msg = 'Missing parameter model'
            raise ValueError(msg)

        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.output = named_outputs.get(
            'model', 'model_{}'.format(self.order))

    def generate_code(self):
        """Generate code."""
        limonero_config = self.parameters.get('configuration') \
            .get('juicer').get('services').get('limonero')

        url = limonero_config['url']
        token = str(limonero_config['auth_token'])

        model_data = query_limonero(url, '/models', token, self.model)
        url = model_data['storage']['url']
        if url[-1] != '/':
            url += '/'

        path = '{}{}'.format(url, model_data['path'])
        parsed = urlparse(path)
        if parsed.scheme == 'file':
            hostname = 'file:///'
            port = 0
        else:
            hostname = parsed.hostname
            port = parsed.port

        code = """
        path = '{path}'        
        fs = pa.hdfs.connect('{hdfs_server}', {hdfs_port})
        exists = fs.exists(path)
        if not exists:
            raise ValueError('{error_file_not_exists}')
        
        import pickle
        from io import BytesIO
        with fs.open(path, 'rb') as f:
            b = BytesIO(f.read()) 
            {model} = pickle.load(b)
        """.format(model=self.output,
                   path=path,
                   hdfs_server=hostname,
                   hdfs_port=port,
                   error_file_not_exists=_('Model does not exist'))
        return dedent(code)


class SaveModel(Operation):
    """SaveModel.
    """
    NAME_PARAM = 'name'
    PATH_PARAM = 'path'
    STORAGE_PARAM = 'storage'
    SAVE_CRITERIA_PARAM = 'save_criteria'
    WRITE_MODE_PARAM = 'write_mode'

    CRITERIA_BEST = 'BEST'
    CRITERIA_ALL = 'ALL'
    CRITERIA_OPTIONS = [CRITERIA_BEST, CRITERIA_ALL]

    WRITE_MODE_ERROR = 'ERROR'
    WRITE_MODE_OVERWRITE = 'OVERWRITE'
    WRITE_MODE_OPTIONS = [WRITE_MODE_ERROR,
                          WRITE_MODE_OVERWRITE]

    WORKFLOW_NAME_PARAM = 'workflow_name'
    JOB_ID_PARAM = 'job_id'
    USER_PARAM = 'user'
    WORKFLOW_ID_PARAM = 'workflow_id'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.parameters = parameters

        self.filename = parameters.get(self.NAME_PARAM)
        self.storage_id = parameters.get(self.STORAGE_PARAM)

        if self.filename is None or self.storage_id is None:
            msg = _('Missing parameters. Check if values for parameters {} '
                    'were informed')
            raise ValueError(
                msg.format(', '.join([self.NAME_PARAM, self.STORAGE_PARAM])))

        self.path = parameters.get(self.PATH_PARAM, '/limonero/models').rstrip('/')

        self.write_mode = parameters.get(self.WRITE_MODE_PARAM, self.WRITE_MODE_ERROR)
        if self.write_mode not in self.WRITE_MODE_OPTIONS:
            raise ValueError(
                _('Invalid value for parameter {param}: value').format(
                    param=self.WRITE_MODE_PARAM, value=self.write_mode))

        self.criteria = parameters.get(self.SAVE_CRITERIA_PARAM, self.CRITERIA_ALL)
        if self.criteria not in self.CRITERIA_OPTIONS:
            raise ValueError(
                _('Invalid value for parameter {param}: {value}').format(
                    param=self.SAVE_CRITERIA_PARAM, value=self.criteria))

        if self.NAME_PARAM not in self.parameters:
            msg = _("Parameters '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.NAME_PARAM, self.__class__))

        self.workflow_id = parameters.get(self.WORKFLOW_ID_PARAM)
        self.workflow_name = parameters.get(self.WORKFLOW_NAME_PARAM)
        self.job_id = parameters.get(self.JOB_ID_PARAM)

        self.has_code = any([len(named_inputs) > 0, self.contains_results()])

    def get_audit_events(self):
        return [auditing.SAVE_MODEL]

    def generate_code(self):
        limonero_config = self.parameters.get('configuration') \
            .get('juicer').get('services').get('limonero')

        url = '{}'.format(limonero_config['url'], self.write_mode)
        token = str(limonero_config['auth_token'])
        storage = limonero_service.get_storage_info(url, token, self.storage_id)

        if storage['type'] != 'HDFS':
            raise ValueError(_('Storage type not supported: {}').format(
                storage['type']))

        if storage['url'].endswith('/'):
            storage['url'] = storage['url'][:-1]

        parsed = urlparse(storage['url'])
        if parsed.scheme == 'file':
            hostname = 'file:///'
            port = 0
        else:
            hostname = parsed.hostname
            port = parsed.port
        models = self.named_inputs['models']
        if not isinstance(models, list):
            models = [models]

        user = self.parameters.get('user', {})
        code = dedent("""
            from juicer.scikit_learn.model_operation import ModelsEvaluationResultList
            from juicer.service.limonero_service import register_model

            all_models = [{models}]
            criteria = '{criteria}'
            if criteria == 'ALL':
                models_to_save = list(itertools.chain.from_iterable(
                    map(lambda m: m.models if isinstance(m,
                        ModelsEvaluationResultList) else [m], all_models)))
            elif criteria == 'BEST':
                raise ValueError('{msg2}')

            import pickle
            from io import BytesIO
            fs = pa.hdfs.connect('{hdfs_server}', {hdfs_port})
            
            def _save_model(model_to_save, model_path, model_name):
                final_model_path = '{final_url}/{{}}'.format(model_path)
                overwrite = '{overwrite}'
                exists = fs.exists(final_model_path)
                if exists:
                    if overwrite == 'OVERWRITE':
                        fs.delete(final_model_path, False)
                    else:
                        raise ValueError('{error_file_exists}')
                        
                with fs.open(final_model_path, 'wb') as f:
                    b = BytesIO()
                    pickle.dump(model_to_save, b)
                    f.write(b.getvalue())
                                
                # Save model information in Limonero
                model_type = '{{}}.{{}}'.format(model_to_save.__module__,
                    model_to_save.__class__.__name__)
                
                model_payload = {{
                    "user_id": {user_id},
                    "user_name": '{user_name}',
                    "user_login": '{user_login}',
                    "name": model_name,
                    "class_name": model_type,
                    "storage_id": {storage_id},
                    "path":  model_path,
                    "type": "UNSPECIFIED",
                    "task_id": '{task_id}',
                    "job_id": {job_id},
                    "workflow_id": {workflow_id},
                    "workflow_name": '{workflow_name}'
                }}
                # Save model information in Limonero
                register_model('{url}', model_payload, '{token}')

            for i, model in enumerate(models_to_save):
                if isinstance(model, dict): # For instance, it's a Indexer
                    for k, v in model.items():
                        name = '{name} - {{}}'.format(k)
                        path = '{path}/{name}.{{0}}.{{1:04d}}'.format(k, i)
                        _save_model(v, path, name)
                else:
                    name = '{name}'
                    path = '{path}/{name}.{{0:04d}}'.format(i)
                    _save_model(model, path, name)
        """.format(models=', '.join(models),
                   overwrite=self.write_mode,
                   path=self.path,
                   final_url=storage['url'],
                   url=url,
                   token=token,
                   storage_id=self.storage_id,
                   name=self.filename.replace(' ', '_'),
                   criteria=self.criteria,
                   msg0=_('You cannot mix models with and without '
                          'evaluation (e.g. indexers) when saving models '
                          'and criteria is different from ALL'),
                   msg1=_('You cannot mix models built using with '
                          'different metrics ({}).'),
                   msg2=_('Invalid criteria.'),
                   error_file_exists=_('Model already exists'),
                   job_id=self.job_id,
                   task_id=self.parameters['task_id'],
                   workflow_id=self.workflow_id,
                   workflow_name=self.workflow_name,
                   user_id=user.get('id'),
                   user_name=user.get('name'),
                   user_login=user.get('login'),
                   hdfs_server=hostname,
                   hdfs_port=port,
                   ))
        return code

    def to_deploy_format(self, id_mapping):
        return []


class EvaluateModelOperation(Operation):

    PREDICTION_ATTRIBUTE_PARAM = 'prediction_attribute'
    LABEL_ATTRIBUTE_PARAM = 'label_attribute'
    FEATURE_ATTRIBUTE_PARAM = 'feature'
    METRIC_PARAM = 'model_type'

    MODEL_TO_METRIC = {
        'classification': 'classification_metric',
        'clustering': 'clustering_metric',
        'regression': 'regression_metric'
    }

    # first index value represents the needs of a feature attribute
    # second index value represents a model type
    METRICS_LIST = {
        'balanced_accuracy_score': (0, 0), 'f1_score': (0, 0),
        'precision_score': (0, 0), 'recall_score': (0, 0),
        'jaccard_score': (0, 0), 'roc_auc_score': (0, 0),
        'matthews_corrcoef': (0, 0), 'cohen_kappa_score': (0, 0),

        'homogeneity_completeness_v_measure': (0, 1),
        'calinski_harabasz_score': (1, 1), 'davies_bouldin_score': (1, 1),
        'silhouette_score': (1, 1),  'fowlkes_mallows_score': (0, 1),
        'adjusted_mutual_info_score': (0, 1),

        'explained_variance_score': (0, 2), 'max_error': (0, 2),
        'mean_absolute_error': (0, 2), 'mean_squared_error': (0, 2),
        'mean_squared_log_error': (0, 2), 'median_absolute_error': (0, 2),
        'r2_score': (0, 2), }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(self.named_inputs) == 2])

        self.type_model = parameters.get(self.METRIC_PARAM) or ''
        if self.type_model not in self.MODEL_TO_METRIC:
            msg = \
                _("Parameter '{}' must be informed for task {}")
            raise ValueError(msg.format(self.METRIC_PARAM, self.__class__))
        else:
            self.metric = parameters.get(self.MODEL_TO_METRIC[self.type_model])\
                          or ['']
            if self.metric not in self.METRICS_LIST:
                msg = \
                    _("Invalid metric value informed in task {}")
                raise ValueError(msg.format(self.METRIC_PARAM, self.__class__))

            if self.METRICS_LIST[self.metric][0] == 1:
                self.second_attribute = (parameters.get(
                        self.FEATURE_ATTRIBUTE_PARAM) or [''])
                second_name = self.FEATURE_ATTRIBUTE_PARAM
            else:
                self.second_attribute = (parameters.get(
                        self.LABEL_ATTRIBUTE_PARAM) or [''])[0]
                second_name = self.LABEL_ATTRIBUTE_PARAM

            self.prediction_attribute = (parameters.get(
                        self.PREDICTION_ATTRIBUTE_PARAM) or [''])[0]
            if any([self.prediction_attribute == "",
                   len(self.second_attribute) == 0]):
                msg = \
                    _("Parameters '{}' and '{}' must be informed for task {}")
                raise ValueError(msg.format(self.PREDICTION_ATTRIBUTE_PARAM,
                                            second_name, self.__class__))

        self.model = self.named_inputs.get(
            'model', 'model_{}'.format(self.order))

        self.model_out = self.named_outputs.get(
            'evaluated model', 'model_task_{}'.format(self.order))

        self.evaluator_out = self.named_outputs.get(
            'evaluator', 'evaluator_task_{}'.format(self.order))
        if not self.has_code and self.named_outputs.get(
                'evaluated model') is not None:
            raise ValueError(
                _('Model is being used, but at least one input is missing'))

        self.supports_cache = False
        self.transpiler_utils.add_import("from sklearn.metrics import *")
        self.transpiler_utils.add_custom_function(
                'get_X_train_data', f=get_X_train_data)
        self.transpiler_utils.add_custom_function(
                'get_label_data', f=get_label_data)
        if self.has_code:
            self.transpiler_utils.add_import(
                'from pandas.api.types import is_numeric_dtype')
            self.transpiler_utils.add_import(
                'from sklearn.metrics import *')
            self.transpiler_utils.add_import(
                    "from sklearn.preprocessing import LabelEncoder")

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return ''

    def generate_code(self):

        if not self.has_code:
            return ''

        display_text = self.parameters['task']['forms'].get(
            'display_text', {'value': 1}).get('value', 1) in (1, '1')
        display_image = self.parameters['task']['forms'].get(
            'display_image', {'value': 1}).get('value', 1) in (1, '1')

        is_regression = self.METRICS_LIST[self.metric][1] == 2
        is_classification = self.METRICS_LIST[self.metric][1] == 0
        is_supervised_metric = self.METRICS_LIST[self.metric][0] == 0

        self.template = """
        {%- if is_regression %}
        final_y_true = {{input}}['{{second_attr}}'].to_numpy()
        final_y_pred = {{input}}['{{prediction_attr}}'].to_numpy()
        {%- elif is_supervised_metric %}
        y_true = get_label_data({{input}}, ['{{second_attr}}'])
        y_pred = get_label_data({{input}}, ['{{prediction_attr}}'])

        # When Label attribute is categorical/string
        if not is_numeric_dtype(y_true):
            le = LabelEncoder()
            le.fit(y_true)
            classes = le.classes_.tolist()
            final_y_true = le.transform(y_true)
            final_y_pred = le.transform(y_pred)
        else:
            classes = list(set(y_true))
            final_y_true = y_true
            final_y_pred = y_pred
        {%- else %}
        X = get_X_train_data({{input}}, {{second_attr}})
        y_pred = {{input}}['{{prediction_attr}}'].to_numpy().tolist()

        # When Prediction attribute is categorical/string
        if not is_numeric_dtype(y_pred):
            le = LabelEncoder()
            le.fit(y_pred)
            final_y_pred = le.transform(y_pred)
        else:
            final_y_pred = y_pred
        {% endif %}

        {% if display_text -%}
        headers = {{table_headers}}
        {%- if is_classification %}
        rows = [
            ['F1', f1_score(final_y_true, final_y_pred, average='weighted')],
            ['Weighted Precision', precision_score(final_y_true, 
                final_y_pred, average='weighted')],
            ['Weighted Recall', recall_score(final_y_true, final_y_pred,
                average='weighted')],
            ['Balanced Accurary', balanced_accuracy_score(final_y_true,
                final_y_pred)],
            ['Cohens kappa', cohen_kappa_score(final_y_true, final_y_pred)],
            ['Jaccard coefficient score', jaccard_score(final_y_true, 
                final_y_pred, average='weighted')],
            ['Matthews correlation coefficient (MCC)', 
                 matthews_corrcoef(final_y_true, final_y_pred)]
        ]

        if len(set(final_y_true)) == 2:
            if set(final_y_true) != set([0, 1]):
                le = LabelEncoder()
                le.fit(final_y_true)
                final_y_true = le.transform(final_y_true)
                final_y_pred = le.transform(final_y_pred)
            rows.append(['Area under ROC', 
                roc_auc_score(final_y_true, final_y_pred)])
            rows.append(['Area under Precision-Recall', 
                average_precision_score(final_y_true, final_y_pred)])
        {%- elif is_regression %}
        rows = [
            ['Maximum error', max_error(final_y_true, final_y_pred)],
            ['Explained variance score', 
                explained_variance_score(final_y_true, final_y_pred)],
            ['Mean squared error', 
                mean_squared_error(final_y_true, final_y_pred)],
            ['Root mean squared error', 
                np.sqrt(mean_squared_error(final_y_true, final_y_pred))],
            ['Mean absolute error', 
                mean_absolute_error(final_y_true, final_y_pred)],
            ['Median absolute error', 
                median_absolute_error(final_y_true, final_y_pred)],
            ['R^2 (coefficient of determination)', 
                r2_score(final_y_true, final_y_pred)],
        ]

        if (final_y_true < 0).any() or (final_y_pred < 0).any():
            if '{{metric}}' == 'mean_squared_log_error':
                raise ValueError('Mean Squared Logarithmic Error cannot' 
                    ' be used when targets contain negative values.')
        else:
            rows.append(['Mean squared log error', 
             mean_squared_log_error(final_y_true, final_y_pred)])
        {%- elif is_supervised_metric %}
        rows = [
            ['Homogeneity, completeness and V-Measure', 
                homogeneity_completeness_v_measure(final_y_true, 
                final_y_pred)],
            ['Fowlkes-Mallows index', 
                fowlkes_mallows_score(final_y_true, final_y_pred)],
            ['Adjusted Mutual Information', 
                adjusted_mutual_info_score(final_y_true, final_y_pred)],
        ]
        {%- else %}
        rows = [
                ['Silhouette Coefficient',  silhouette_score(X, final_y_pred)],
                ['Calinski and Harabaz score',  
                    calinski_harabasz_score(X, final_y_pred)],
                ['Davies-Bouldin score', davies_bouldin_score(X, final_y_pred)],
        ]  
        {% endif %}
        
        content = SimpleTableReport(
                'table table-striped table-bordered table-sm',
                headers, rows, title='{{title}}').generate()

        emit_event(
            'update task', status='COMPLETED',
            identifier='{{task_id}}',
            message=content,
            type='HTML', title='{{title}}',
            task=dict([('id', '{{task_id}}')]),
            operation=dict([('id', {{operation_id}})]),
            operation_id={{operation_id}}) 
            
        headers = {{params_table_headers}}
        params = {{model}}.get_params()
        rows = []
        for p in params:
            rows.append([p, params[p]])

        content = SimpleTableReport(
                'table table-striped table-bordered table-sm',
                headers, rows, title='{{params_title}}').generate()

        emit_event(
            'update task', status='COMPLETED',
            identifier='{{task_id}}',
            message=content,
            type='HTML', title='{{title}}',
            task=dict([('id', '{{task_id}}')]),
            operation=dict([('id', {{operation_id}})]),
            operation_id={{operation_id}}) 
        {% endif %}
       
        {% if display_image %}
        {% if is_supervised_metric %} 
        if len(final_y_true) < 2000:
            identity = range(int(max(max(final_y_true), max(final_y_pred))))
            report2 = MatplotlibChartReport().plot(
                    '{{plot_title}}',
                    '{{plot_x_title}}',
                    '{{plot_y_title}}',
                    identity, identity, 'r-',
                    final_y_true, final_y_pred,'b.', linewidth=1, 
                    submission_lock=submission_lock)

            emit_event(
                'update task', status='COMPLETED',
                identifier='{{task_id}}',
                message=report2,
                type='IMAGE', title='{{plot_title}}',
                task=dict([('id', '{{task_id}}')]),
                operation=dict([('id', {{operation_id}})]),
                operation_id={{operation_id}}) 
        {%- endif %}

        {% if is_classification %}
        content = ConfusionMatrixImageReport(
            cm=confusion_matrix(final_y_true, final_y_pred), 
            classes=classes,).generate(submission_lock)

        emit_event(
                'update task', status='COMPLETED',
                identifier='{{task_id}}',
                message=content,
                type='IMAGE', title='{{title}}',
                task=dict([('id', '{{task_id}}')]),
                operation=dict([('id', {{operation_id}})]),
                operation_id={{operation_id}}) 
     
        {% elif is_regression %}
        if len(final_y_true) < 2000:
            residuals = final_y_true - final_y_pred
            pandas_df = pd.DataFrame.from_records(
                [
                    dict(prediction=x[0], residual=x[1])
                        for x in zip(final_y_true, residuals)
                ]
            )
            pandas_df.rename(index=str, columns=dict(
                        prediction='{join_plot_x_title}',
                        residuals='{join_plot_y_title}'))

            content = SeabornChartReport().jointplot(pandas_df, 
                'prediction', 'residual', '{{join_plot_title}}',
                '{{join_plot_x_title}}', '{{join_plot_y_title}}', 
                submission_lock)
                    
            emit_event(
                'update task', status='COMPLETED',
                identifier='{{task_id}}',
                message=content,
                type='IMAGE', title='{{join_plot_title}}',
                task=dict([('id', '{{task_id}}')]),
                operation=dict([('id', {{operation_id}})]),
                operation_id={{operation_id}}) 
        {%- endif %}
        {%- endif %}
        """

        ctx = dict(is_regression=is_regression,
                   is_classification=is_classification,
                   is_supervised_metric=is_supervised_metric,
                   display_text=display_text,
                   display_image=display_image,
                   evaluator_out=self.evaluator_out,
                   input=self.named_inputs['input data'],
                   metric=self.metric,
                   prediction_attr=self.prediction_attribute,
                   second_attr=self.second_attribute,
                   model=self.model,
                   model_output=self.model_out,
                   task_id=self.parameters['task_id'],
                   operation_id=self.parameters['operation_id'],
                   join_plot_title=_('Prediction versus Residual'),
                   join_plot_y_title=_('Residual'),
                   join_plot_x_title=_('Prediction'),
                   params_title=_('Parameters for this estimator'),
                   params_table_headers=[_('Parameters'), _('Value')],
                   plot_title=_('Actual versus Prediction'),
                   plot_x_title=_('Actual'),
                   plot_y_title=_('Prediction'),
                   table_headers=[_('Metric'), _('Value')],
                   title=_('Evaluation result')
                   )

        return dedent(self.render_template(ctx))


class ModelsEvaluationResultList:
    """ Stores a list of ModelEvaluationResult """

    def __init__(self, models, best, metric_name, metric_value):
        self.models = models
        self.best = best
        self.metric_name = metric_name
        self.metric_value = metric_value


class CrossValidationOperation(Operation):
    """
    Cross validation operation used to evaluate classifier results using as many
    as folds provided in input.
    """
    NUM_FOLDS_PARAM = 'folds'
    EVALUATOR_PARAM = 'evaluator'
    SEED_PARAM = 'seed'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction_attribute'
    LABEL_ATTRIBUTE_PARAM = 'label_attribute'
    FEATURE_ATTRIBUTE_PARAM = 'features'

    METRIC_TO_EVALUATOR = {
        'areaUnderROC': (
            'roc_auc', 'rawPredictionCol'),
        'areaUnderPR': (
            'evaluation.BinaryClassificationEvaluator', 'rawPredictionCol'),
        'f1': ('f1', 'predictionCol'),
        'weightedPrecision': (
            'precision_weighted', 'predictionCol'),
        'weightedRecall': (
            'recall_weighted', 'predictionCol'),
        'accuracy': (
            'accuracy', 'predictionCol'),
        'rmse': ('r2', 'predictionCol'),
        'mse': ('neg_mean_squared_error', 'predictionCol'),
        'mae': ('neg_mean_absolute_error', 'predictionCol'),
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(self.named_inputs) == 2])

        if self.EVALUATOR_PARAM not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.EVALUATOR_PARAM, self.__class__))

        self.metric = parameters.get(self.EVALUATOR_PARAM)

        if self.metric in self.METRIC_TO_EVALUATOR:
            self.evaluator = self.METRIC_TO_EVALUATOR[self.metric][0]
            self.param_prediction_arg = \
                self.METRIC_TO_EVALUATOR[self.metric][1]
        else:
            raise ValueError(
                _('Invalid metric value {}').format(self.metric))

        self.prediction_attr = parameters.get(
            self.PREDICTION_ATTRIBUTE_PARAM,
            'prediction')

        for param in [self.FEATURE_ATTRIBUTE_PARAM, self.LABEL_ATTRIBUTE_PARAM]:
            if param not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.LABEL_ATTRIBUTE_PARAM, self.__class__))

        self.label_attr = parameters.get(self.LABEL_ATTRIBUTE_PARAM)
        self.feature_attr = parameters.get(self.FEATURE_ATTRIBUTE_PARAM)

        self.num_folds = parameters.get(self.NUM_FOLDS_PARAM, 3)
        self.seed = parameters.get(self.SEED_PARAM, 'None')

        self.output = self.named_outputs.get(
            'scored data', 'scored_data_task_{}'.format(self.order))
        self.has_models = 'models' in self.named_outputs
        self.models = self.named_outputs.get(
            'models', 'models_task_{}'.format(self.order))
        self.best_model = self.named_outputs.get(
            'best model', 'best_model_{}'.format(self.order))

        self.algorithm_port = self.named_inputs.get(
            'algorithm', 'algo_{}'.format(self.order))
        self.input_port = self.named_inputs.get(
            'input data', 'in_{}'.format(self.order))
        self.evaluator_port = self.named_inputs.get(
            'evaluator', 'eval_{}'.format(self.order))

        self.transpiler_utils.add_import(
                "from sklearn.model_selection import cross_val_score, KFold")

    @property
    def get_inputs_names(self):
        return ', '.join(
            [self.algorithm_port, self.input_port, self.evaluator_port])

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.best_model, self.models])

    def get_data_out_names(self, sep=','):
        return sep.join([self.output])

    def generate_code(self):
        code = dedent("""
                  kf = KFold(n_splits={folds}, random_state={seed}, 
                  shuffle=True)
                  X_train = get_X_train_data({input_data}, {feature_attr})
                  y = get_label_data({input_data}, {label_attr})

                  scores = cross_val_score({algorithm}, X_train.tolist(), 
                                           y.tolist(), cv=kf, scoring='{metric}')

                  best_score = np.argmax(scores)
                  """.format(algorithm=self.algorithm_port,
                             input_data=self.input_port,
                             evaluator=self.evaluator,
                             output=self.output,
                             best_model=self.best_model,
                             models=self.models,
                             prediction_arg=self.param_prediction_arg,
                             prediction_attr=self.prediction_attr,
                             label_attr=self.label_attr,
                             feature_attr=self.feature_attr,
                             folds=self.num_folds,
                             metric=self.metric,
                             seed=self.seed))

        if self.has_models:

            code += dedent("""
                    models = []
                    for train_index, test_index in kf.split(X_train):
                        Xf_train, Xf_test = X_train[train_index], X_train[test_index]
                        yf_train, yf_test = y[train_index],  y[test_index]
                        {algorithm}.fit(Xf_train.tolist(), yf_train.tolist())
                        models.append({algorithm})
                    
                    {best_model} = models[best_score]
                    """.format(algorithm=self.algorithm_port,
                               input_data=self.input_port,
                               evaluator=self.evaluator,
                               output=self.output,
                               best_model=self.best_model,
                               models=self.models,
                               prediction_arg=self.param_prediction_arg,
                               prediction_attr=self.prediction_attr,
                               label_attr=self.label_attr[0],
                               feature_attr=self.feature_attr[0],
                               folds=self.num_folds,
                               metric=self.metric,
                               seed=self.seed))
        else:
            code += dedent("""
                    models = None
                    train_index, test_index = list(kf.split(X_train))[best_score]
                    Xf_train, Xf_test = X_train[train_index], X_train[test_index]
                    yf_train, yf_test = y[train_index],  y[test_index]
                    {best_model} = {algorithm}.fit(Xf_train.tolist(), yf_train.tolist())
                    """.format(algorithm=self.algorithm_port,
                               input_data=self.input_port,
                               evaluator=self.evaluator,
                               output=self.output,
                               best_model=self.best_model,
                               models=self.models,
                               prediction_arg=self.param_prediction_arg,
                               prediction_attr=self.prediction_attr,
                               label_attr=self.label_attr[0],
                               feature_attr=self.feature_attr[0],
                               folds=self.num_folds,
                               metric=self.metric,
                               seed=self.seed))
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['input data'] > 1 else ""

        code += dedent("""
                metric_result = scores[best_score]
                {output} = {input_data}{copy_code}
                {output}['{prediction_attr}'] = {best_model}.predict(X_train.tolist())
                {models} = models
                """.format(copy_code=copy_code, algorithm=self.algorithm_port,
                           input_data=self.input_port,
                           evaluator=self.evaluator,
                           output=self.output,
                           best_model=self.best_model,
                           models=self.models,
                           prediction_arg=self.param_prediction_arg,
                           prediction_attr=self.prediction_attr,
                           label_attr=self.label_attr[0],
                           feature_attr=self.feature_attr[0],
                           folds=self.num_folds,
                           metric=self.metric,
                           seed=self.seed))

        # # If there is an output needing the evaluation result, it must be
        # # processed here (summarization of data results)
        # needs_evaluation = 'evaluation' in self.named_outputs and False
        # if needs_evaluation:
        #     eval_code = """
        #         grouped_result = fit_data.select(
        #                 evaluator.getLabelCol(), evaluator.getPredictionCol())\\
        #                 .groupBy(evaluator.getLabelCol(),
        #                          evaluator.getPredictionCol()).count().collect()
        #         eval_{output} = {{
        #             'metric': {{
        #                 'name': evaluator.getMetricName(),
        #                 'value': metric_result
        #             }},
        #             'estimator': {{
        #                 'name': estimator.__class__.__name__,
        #                 'predictionCol': evaluator.getPredictionCol(),
        #                 'labelCol': evaluator.getLabelCol()
        #             }},
        #             'confusion_matrix': {{
        #                 'data': json.dumps(grouped_result)
        #             }},
        #             'evaluator': evaluator
        #         }}
        #
        #         emit_event('task result', status='COMPLETED',
        #             identifier='{task_id}', message='Result generated',
        #             type='TEXT', title='{title}',
        #             task={{'id': '{task_id}' }},
        #             operation={{'id': {operation_id} }},
        #             operation_id={operation_id},
        #             content=json.dumps(eval_{output}))
        #
        #         """.format(output=self.output,
        #                    title='Evaluation result',
        #                    task_id=self.parameters['task_id'],
        #                    operation_id=self.parameters['operation_id'])
        # else:
        #     eval_code = ''
        # code = '\n'.join([code, dedent(eval_code)])

        return code