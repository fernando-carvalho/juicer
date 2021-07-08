# coding=utf-8
import json
import os
import uuid
from textwrap import dedent, indent

import datetime

from juicer.operation import Operation
from juicer.service import limonero_service

from urllib.request import urlopen
from urllib.parse import urlparse, parse_qs


class DataReaderOperation(Operation):
    HEADER_PARAM = 'header'
    SEPARATOR_PARAM = 'separator'
    QUOTE_PARAM = 'quote'
    INFER_SCHEMA_PARAM = 'infer_schema'
    NULL_VALUES_PARAM = 'null_values'  # Must be compatible with other platforms
    MODE_PARAM = 'mode'
    DATA_SOURCE_ID_PARAM = 'data_source'

    INFER_FROM_LIMONERO = 'FROM_LIMONERO'
    INFER_FROM_DATA = 'FROM_VALUES'
    DO_NOT_INFER = 'NO'

    OPT_MODE_FAILFAST = 'FAILFAST'

    SEPARATORS = {
        '{tab}': '\\t',
        '{new_line}': '\\n',
    }

    SUPPORTED_DRIVERS = {
        'mysql': dedent("""
            import pymysql
            query = '{query}'
            connection = pymysql.connect(
                host='{host}',
                port={port} or 3306,
                user='{user}',
                password='{password}',
                db='{db}',
                charset='utf8')
            {out} = pd.read_sql(query, con=connection)
            connection.close()
        """)
    }
    LIMONERO_TO_PANDAS_DATA_TYPES = {
        "CHARACTER": 'object',
        "DATETIME": 'object',
        "DATE": 'datetime.date',
        "DOUBLE": 'np.float64',
        "DECIMAL": 'np.float64',
        "FLOAT": 'np.float64',
        "LONG": 'np.int64',
        "INTEGER": 'pd.Int64Dtype()',
        "TEXT": 'object',
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

        self.header = False
        if self.has_code:
            if self.DATA_SOURCE_ID_PARAM in parameters:
                self._set_data_source_parameters(parameters)
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.DATA_SOURCE_ID_PARAM, self.__class__))

            # Test if data source was changed since last execution and
            # invalidate cache if so.
            self._set_data_source_parameters(parameters)
            data_source_updated = self.metadata.get('updated')

            if data_source_updated:
                data_source_updated = datetime.datetime.strptime(
                    data_source_updated[0:19], '%Y-%m-%dT%H:%M:%S')
            self.supports_cache = (
                parameters.get('execution_date') is not None and
                data_source_updated < parameters['execution_date'])

            self.output = named_outputs.get('output data',
                                            'out_task_{}'.format(self.order))

    def _set_data_source_parameters(self, parameters):

        self.data_source_id = int(parameters[self.DATA_SOURCE_ID_PARAM])
        # Retrieve metadata from Limonero.
        limonero_config = self.parameters['configuration']['juicer'][
            'services']['limonero']
        url = limonero_config['url']
        token = str(limonero_config['auth_token'])

        # Is data source information cached?
        self.metadata = self.parameters.get('workflow', {}).get(
            'data_source_cache', {}).get(self.data_source_id)
        if self.metadata is None:
            self.metadata = limonero_service.get_data_source_info(
                url, token, self.data_source_id)
            self.parameters['workflow']['data_source_cache'][
                self.data_source_id] = self.metadata
        if not self.metadata.get('url'):
            raise ValueError(
                _('Incorrect data source configuration (empty url)'))

        self.header = parameters.get(
            self.HEADER_PARAM, False) not in ('0', 0, 'false', False)
        self.null_values = [v.strip() for v in parameters.get(
            self.NULL_VALUES_PARAM, '').split(",") if v.strip()]

        self.sep = parameters.get(
            self.SEPARATOR_PARAM,
            self.metadata.get('attribute_delimiter', ',')) or ','
        if self.metadata['format'] == 'TEXT':
            self.sep = '{new_line}'
        self.quote = parameters.get(self.QUOTE_PARAM,
                                    self.metadata.get('text_delimiter'))
        if self.quote == '\'':
            self.quote = '\\\''
        if self.sep in self.SEPARATORS:
            self.sep = self.SEPARATORS[self.sep]
        self.infer_schema = parameters.get(self.INFER_SCHEMA_PARAM,
                                           self.INFER_FROM_LIMONERO)
        self.mode = parameters.get(self.MODE_PARAM, 'FAILFAST')

    def analyse_attributes(self, attrs):
        attributes = None
        converters = None
        parse_dates = None
        names = None

        if attrs:
            attributes = []
            converters = {}  # Not supported
            parse_dates = []
            names = []
            for attr in attrs:
                names.append(attr['name'])
                if attr['type'] == 'DATETIME':
                    parse_dates.append(attr['name'])
                # elif attr['type'] == 'DECIMAL':
                    #    converters[attr['name']] = 'decimal.Decimal'
        
                data_type = self.LIMONERO_TO_PANDAS_DATA_TYPES[attr['type']]
                attributes.append([attr['name'], data_type])

                # Metadata is not supported in scikit-learn
                # metadata = {'sources': [
                #     '{}/{}'.format(self.data_source_id, attr['name'])
                # ]}

        return attributes, converters, parse_dates, names

    def generate_code(self):
        """
        """
        if not self.has_code:
            return ''

        infer_from_data = self.infer_schema == self.INFER_FROM_DATA
        infer_from_limonero = self.infer_schema == self.INFER_FROM_LIMONERO
        do_not_infer = self.infer_schema == self.DO_NOT_INFER
        mode_failfast = self.mode == self.OPT_MODE_FAILFAST
        
        protect = (self.parameters.get('export_notebook', False) or
                   self.parameters.get('plain', False)) or self.plain
        data_format = self.metadata.get('format')

        parsed = urlparse(self.metadata['url'])

        extra_params = {}
        if 'extra_params' in self.metadata['storage']:
            if self.metadata['storage']['extra_params']:
                extra_params = json.loads(self.metadata['storage'][
                    'extra_params'])

        if self.metadata.get('privacy_aware', False):
            raise ValueError(_('Not supported'))

        if parsed.scheme not in ('hdfs', 'file', 'mysql'):
            raise ValueError(_('Not supported'))

        if data_format not in ('CSV', 'TEXT', 'PARQUET', 'JDBC', 'JSON'):
            raise ValueError(_('Not supported'))

        if data_format == 'JDBC':
            qs_parsed = parse_qs(parsed.query)
            if parsed.scheme not in self.SUPPORTED_DRIVERS:
                raise ValueError(
                    _('Database {} not supported').format(parsed.scheme))
            if not self.metadata.get('command'):
                raise ValueError(
                    _('No command nor table specified for data source.'))

            jdbc_code = indent(dedent(self.SUPPORTED_DRIVERS[parsed.scheme].format(
                scheme=parsed.scheme, host=parsed.hostname,
                db=parsed.path[1:],
                port=parsed.port,
                query=self.metadata.get('command'),
                user=qs_parsed.get('user', [''])[0],
                password=qs_parsed.get('password', [''])[0],
                out=self.output)), '    ')
        else:
            jdbc_code = None

        self.header = self.metadata.get('is_first_line_header')

        attributes, converters, parse_dates, names = self.analyse_attributes(
             self.metadata.get('attributes'))

        self.template = """
        {%- if infer_from_limonero %}
        {%-   if attributes and format in ('TEXT', 'CSV') %}
        columns = {
        {%-     for attr in attributes %}
            '{{attr[0]}}': {{attr[1]}},
        {%-     endfor %}
        }
        {%-   elif format in ('TEXT', 'CSV') %}
        columns = {'value': object}
        {%-   endif %}
        {%- elif infer_from_data and format in ('TEXT', 'CSV') %}
        columns = None
        header = 'infer'
        {%- elif do_not_infer and format in ('TEXT', 'CSV') %}
        header = 'infer'
        {%- endif %}

        # Open data source
        {%- if protect %}
        f = open('{{parsed.path.split('/')[-1]}}', 'rb')
        {%- elif parsed.scheme == 'hdfs'  %}
        fs = pa.hdfs.connect(host='{{parsed.hostname}}', 
            port={{parsed.port}},
            user='{{extra_params.get('user', parsed.username) or 'hadoop'}}')
        f = fs.open('{{parsed.path}}', 'rb')
        {%- elif parsed.scheme == 'file' %}
        f = open('{{parsed.path}}', 'rb')
        {%- endif %}

        {%- if format == 'CSV' %}
        {{output}} = pd.read_csv(f, sep='{{sep}}',
                                 encoding='{{encoding}}',
                                 header={{header}},
                                 {%- if infer_from_limonero %}
                                 names={{names}},
                                 dtype=columns,
                                 parse_dates={{parse_dates}},
                                 converters={{converters}},
                                 {%- elif do_not_infer %}
                                 parse_dates = None,
                                 converters = None,
                                 dtype='str',
                                 {%-   endif %}
                                 na_values={{na_values}},
                                 error_bad_lines={{mode_failfast}})
        f.close()
        {%-   if header == 'infer' %}
        {{output}}.columns = ['attr{{i}}'.format(i=i) 
                        for i, _ in enumerate({output}.columns)]
        {%-   endif %}
        {%- elif format == 'TEXT' %}
        {{output}} = pd.read_csv(f, sep='{{sep}}',
                                 encoding='{{encoding}}',
                                 names = ['value'],
                                 error_bad_lines={{mode_failfast}})
        f.close()
        {%- elif format == 'PARQUET' %}
        {{output}} = pd.read_parquet(f, engine='pyarrow')
        f.close()
        {%- elif format == 'JSON' %}
        {{output}} = pd.read_json(f, orient='records')
        f.close()
        {%- elif format == 'JDBC' %}
        {{jdbc_code}}
        {%- endif %}

        {%- if infer_from_data %}
        {{output}} = {{output}}.infer_objects()
        {%- endif %}

        """
        ctx = {
            'attributes': attributes,
            'parse_dates': parse_dates,
            'names': names,
            'converters': converters,

            'infer_from_limonero': infer_from_limonero,
            'infer_from_data': infer_from_data,
            'do_not_infer': do_not_infer,
            'is_first_line_header': self.header,

            'protect': protect,  # Hide information about path
            'parsed': parsed,
            'extra_params': extra_params,
            'format': data_format,
            'encoding': self.metadata.get('encoding', 'utf-8') or 'utf-8',
            'header': 0 if self.header else 'None',
            'sep': self.sep,
            'na_values': self.null_values if len(self.null_values) else 'None',
            'output': self.output,
            'mode_failfast': mode_failfast,

            'jdbc_code': jdbc_code,
        }
        return dedent(self.render_template(ctx))


class SaveOperation(Operation):
    """
    Saves the content of the DataFrame at the specified path.
    """
    NAME_PARAM = 'name'
    PATH_PARAM = 'path'
    STORAGE_ID_PARAM = 'storage'
    FORMAT_PARAM = 'format'
    TAGS_PARAM = 'tags'
    OVERWRITE_MODE_PARAM = 'mode'
    HEADER_PARAM = 'header'

    MODE_ERROR = 'error'
    MODE_APPEND = 'append'
    MODE_OVERWRITE = 'overwrite'
    MODE_IGNORE = 'ignore'

    FORMAT_PICKLE = 'PICKLE'
    FORMAT_CSV = 'CSV'
    FORMAT_JSON = 'JSON'
    FORMAT_PARQUET = 'PARQUET'

    USER_PARAM = 'user'
    WORKFLOW_ID_PARAM = 'workflow_id'

    PANDAS_TO_LIMONERO_DATA_TYPES = {
        'object': "CHARACTER",
        'datetime64[ns]': "DATETIME",
        'float64': "DOUBLE",
        'float32': "FLOAT",
        'int64': "LONG",
        'Int64': "INTEGER",
        'int32': "INTEGER",
        'int16': "INTEGER",
        'int8': "INTEGER",
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        for att in [self.NAME_PARAM, self.FORMAT_PARAM, self.PATH_PARAM]:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        att, self.__class__))

        self.name = parameters.get(self.NAME_PARAM)
        self.tags = parameters.get(self.TAGS_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM, self.FORMAT_CSV)
        self.path = parameters.get(self.PATH_PARAM, '.')
        self.mode = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)
        self.storage_id = parameters.get(self.STORAGE_ID_PARAM)
        self.user = parameters.get(self.USER_PARAM)
        self.workflow_id = parameters.get(self.WORKFLOW_ID_PARAM)

        self.header = parameters.get(self.HEADER_PARAM, False) in (1, '1', True)

        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                 self.order))

        self.filename = self.name
        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):

        limonero_config = \
            self.parameters['configuration']['juicer']['services']['limonero']
        url = '{}'.format(limonero_config['url'], self.mode)
        token = str(limonero_config['auth_token'])
        storage = limonero_service.get_storage_info(url, token, self.storage_id)

        if storage['type'] != 'HDFS':
            raise ValueError(_('Storage type not supported: {}').format(
                storage['type']))

        protect = (self.parameters.get('export_notebook', False) or 
             self.parameters.get('plain', False)) or self.plain

        if storage['url'].endswith('/'):
            storage['url'] = storage['url'][:-1]
        if self.path.endswith('/'):
            self.path = self.path[:-1]

        if self.path.startswith('/'):
            self.path = self.path[1:]

        final_url = '{}/limonero/user_data/{}/{}/{}'.format(
            storage['url'],
            self.user['id'],
            self.path,
            self.name.replace(' ', '_'))

        if self.format == self.FORMAT_CSV and not final_url.endswith('.csv'):
            final_url += '.csv'
        elif self.format == self.FORMAT_JSON and not final_url.endswith(
                '.json'):
            final_url += '.json'
        elif self.format == self.FORMAT_PARQUET and not final_url.endswith(
                '.parquet'):
            final_url += '.parquet'

        parsed = urlparse(final_url)

        df_input = self.named_inputs['input data']
        extra_params = storage.get('extra_params') \
            if storage.get('extra_params') is not None else "{}"
        extra_params = json.loads(extra_params)

        hdfs_user = extra_params.get('user', parsed.username) or 'hadoop'
        self.template = """
            path = '{{path}}'
            {%- if scheme == 'hdfs' and not protect %}
            fs = pa.hdfs.connect(host='{{hdfs_server}}', 
                                 port={{hdfs_port}},
                                 user='{{hdfs_user}}')
            exists = fs.exists(path)
            {%- elif scheme == 'file' or protect %}
            exists = os.path.exists(path)
            {%- endif %}

            mode = '{{mode}}'
            if mode not in ('error', 'ignore', 'overwrite'):
                raise ValueError('{{error_invalid_mode}}')
            if exists:
                if mode == 'error':
                    raise ValueError('{{error_file_exists}}')
                elif mode == 'ignore':
                    emit_event(name='update task',
                        message='{{warn_ignored}}',
                        status='COMPLETED',
                        identifier='{{task_id}}')
                else:
                    {%- if scheme == 'hdfs' and not protect %}
                        fs.delete(path, False)
                    {%- elif scheme == 'file' or protect %}
                        os.remove(path)
                        parent_dir = os.path.dirname(path)
                        if not os.path.exists(parent_dir):
                            os.makedirs(parent_dir)
                    {%- endif %}
            else:
                {%-if scheme == 'hdfs' and not protect %}    
                fs.mkdir(os.path.dirname(path))
                {%- elif scheme == 'file' %}
                parent_dir = os.path.dirname(path)
                os.makedirs(parent_dir)
                {%- else %}
                pass
                {%- endif%}
            
            {%- if format == FORMAT_CSV %}
            {%- if scheme == 'hdfs' and not protect %}
            from io import StringIO
            with fs.open(path, 'wb') as f:
                s = StringIO()
                {{input}}.to_csv(s, sep=str(','), mode='w',
                header={{header}}, index=False, encoding='utf-8')
                f.write(s.getvalue().encode())               
            {%- elif scheme == 'file' or protect %}
            {{input}}.to_csv(path, sep=str(','), mode='w',
            header={{header}}, index=False, encoding='utf-8')
            {%- endif %}
            
            {%- elif format == FORMAT_PARQUET %}
            {%- if scheme == 'hdfs' and not protect %}
            from io import BytesIO
            with fs.open(path, 'wb') as f:
                s = BytesIO()
                {{input}}.to_parquet(s, engine='pyarrow')
                f.write(s.getvalue())               
            {%- elif scheme == 'file' or protect %}
            {{input}}.to_parquet(path, engine='pyarrow')
            {%- endif %}
            
            {%- elif format == FORMAT_JSON %}
            {%- if scheme == 'hdfs' and not protect %}
            from io import StringIO
            with fs.open(path, 'wb') as f:
                s = StringIO()
                {{input}}.to_json(s, orient='records')
                f.write(s.getvalue().encode())             
            {%- elif scheme == 'file' or protect %}
            {{input}}.to_json(path, orient='records')
            {%- endif %}
            {%- endif %}
            
            {%-if not protect %}
            # Code to update Limonero metadata information
            from juicer.service.limonero_service import register_datasource
            types_names = {{data_types}}

            write_header = {{header}}
            attributes = []
            for attr in {{input}}.columns:
                type_name = {{input}}.dtypes[attr]
                precision = None
                scale = None
                attributes.append({
                  'enumeration': 0,
                  'feature': 0,
                  'label': 0,
                  'name': attr,
                  'type': types_names[str(type_name)],
                  'nullable': True,
                  'metadata': None,
                  'precision': precision,
                  'scale': scale
                })
            parameters = {
                'name': "{{name}}",
                'is_first_line_header': write_header,
                'enabled': 1,
                'is_public': 0,
                'format': "{{format}}",
                'storage_id': {{storage}},
                'description': "{{description}}",
                'user_id': "{{user_id}}",
                'user_login': "{{user_login}}",
                'user_name': "{{user_name}}",
                'workflow_id': "{{workflow_id}}",
                'task_id': '{{task_id}}',
                'url': "{{final_url}}",
                'attributes': attributes
            }
            register_datasource('{{url}}', parameters, '{{token}}', 'overwrite')
            {%- endif %}
        """
        path = parsed.path

        ctx = dict(protect=protect,
                   path=path if not protect else os.path.basename(path),
                   hdfs_server=parsed.hostname,
                   hdfs_port=parsed.port,
                   hdfs_user=hdfs_user,
                   scheme=parsed.scheme,
                   name=self.name,
                   mode=self.mode,
                   storage=self.storage_id,
                   description=_('Data source generated by workflow {}').format(
                       self.workflow_id),
                   workflow_id=self.workflow_id,
                   format=self.format,
                   header=self.header,
                   user_name=self.user['name'],
                   user_id=self.user['id'],
                   user_login=self.user['login'],
                   tags=repr(self.tags),
                   FORMAT_CSV=self.FORMAT_CSV,
                   FORMAT_PICKLE=self.FORMAT_PICKLE,
                   FORMAT_JSON=self.FORMAT_JSON,
                   FORMAT_PARQUET=self.FORMAT_PARQUET,
                   data_types=json.dumps(self.PANDAS_TO_LIMONERO_DATA_TYPES),
                   final_url=final_url,
                   input=df_input,
                   token=token,
                   url=url,
                   error_file_exists=_('File already exists'),
                   warn_ignored=_('File not written (already exists)'),
                   error_invalid_mode=_('Invalid mode {}').format(self.mode),
                   uuid=uuid.uuid4().hex,
                   storage_url=storage['url'],
                   task_id=self.parameters['task_id'],
                   )

        return dedent(self.render_template(ctx))
