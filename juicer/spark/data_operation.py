# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import ast
import itertools
import json
import pprint
import uuid
from textwrap import dedent

import datetime

from future.backports.urllib.parse import urlparse, parse_qs
from juicer.operation import Operation
from juicer.privaaas import PrivacyPreservingDecorator
from juicer.service import limonero_service


class DataReaderOperation(Operation):
    """
    Reads a database.
    Parameters:
    - Limonero database ID
    """
    DATA_SOURCE_ID_PARAM = 'data_source'
    HEADER_PARAM = 'header'
    SEPARATOR_PARAM = 'separator'
    QUOTE_PARAM = 'quote'
    INFER_SCHEMA_PARAM = 'infer_schema'
    NULL_VALUES_PARAM = 'null_values'
    MODE_PARAM = 'mode'

    INFER_FROM_LIMONERO = 'FROM_LIMONERO'
    INFER_FROM_DATA = 'FROM_VALUES'
    DO_NOT_INFER = 'NO'

    LIMONERO_TO_SPARK_DATA_TYPES = {
        "CHARACTER": 'types.StringType',
        "DATETIME": 'types.TimestampType',
        "DATE": 'types.DateType',
        "DOUBLE": 'types.DoubleType',
        "DECIMAL": 'types.DecimalType',
        "FLOAT": 'types.FloatType',
        "LONG": 'types.LongType',
        "INTEGER": 'types.IntegerType',
        "TEXT": 'types.StringType',
    }

    SUPPORTED_DRIVERS = {
        'mysql': 'com.mysql.jdbc.Driver'
    }
    DATA_TYPES_WITH_PRECISION = {'DECIMAL'}

    SEPARATORS = {
        '{tab}': '\\t',
        '{new_line}': '\\n',
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

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
        limonero_config = \
            self.parameters['configuration']['juicer']['services'][
                'limonero']
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
            self.NULL_VALUES_PARAM, '').split(",")]

        self.sep = parameters.get(
            self.SEPARATOR_PARAM,
            self.metadata.get('attribute_delimiter', ',')) or ','
        self.quote = parameters.get(self.QUOTE_PARAM,
                                    self.metadata.get('text_delimiter'))
        if self.quote == '\'':
            self.quote = '\\\''
        if self.sep in self.SEPARATORS:
            self.sep = self.SEPARATORS[self.sep]
        self.infer_schema = parameters.get(self.INFER_SCHEMA_PARAM,
                                           self.INFER_FROM_LIMONERO)
        self.mode = parameters.get(self.MODE_PARAM, 'FAILFAST')

    def generate_code(self):

        # For now, just accept CSV files.
        # Should we create a dict with the CSV info at Limonero?
        # such as header and sep.
        # print "\n\n",self.metadata,"\n\n"
        code = []
        infer_from_data = self.infer_schema == self.INFER_FROM_DATA
        infer_from_limonero = self.infer_schema == self.INFER_FROM_LIMONERO
        if self.has_code:
            if infer_from_limonero:
                if 'attributes' in self.metadata:
                    code.append(
                        'schema_{0} = types.StructType()'.format(self.output))
                    attrs = self.metadata.get('attributes', [])
                    if attrs:
                        for attr in attrs:
                            self._add_attribute_to_schema(attr, code)
                    else:
                        code.append(
                            "schema_{0}.add('value', "
                            "types.StringType(), 1, None)".format(self.output))

                    code.append("")
                else:
                    raise ValueError(
                        _("Metadata do not include attributes information"))
            else:
                code.append('schema_{0} = None'.format(self.output))

            if self.metadata['format'] in ['CSV', 'TEXT']:
                # Multiple values not supported yet! See SPARK-17878
                code.append("url = '{url}'".format(url=self.metadata['url']))
                null_values = self.null_values
                if self.metadata.get('treat_as_missing'):
                    null_values.extend([x.strip() for x in self.metadata.get(
                        'treat_as_missing').split(',')])
                null_option = ''.join(
                    [".option('nullValue', '{}')".format(n) for n in
                     set(null_values)]) if null_values else ""

                if self.metadata['format'] == 'CSV':
                    encoding = self.metadata.get('encoding', 'UTF-8') or 'UTF-8'
                    # Spark does not works with encoding + multiline options
                    # See https://github.com/databricks/spark-csv/issues/448
                    code_csv = dedent("""
                        {output} = spark_session.read{null_option}.option(
                            'treatEmptyValuesAsNulls', 'true').option(
                            'wholeFile', True).option(
                                'multiLine', {multiline}).option('escape',
                                    '"').csv(
                                url, schema=schema_{output},
                                quote={quote},
                                encoding='{encoding}',
                                header={header}, sep='{sep}',
                                inferSchema={infer_schema},
                                mode='{mode}')""".format(
                        output=self.output,
                        header=self.header or self.metadata.get(
                            'is_first_line_header', False),
                        sep=self.sep,
                        quote='None' if self.quote is None else "'{}'".format(
                            self.quote),
                        infer_schema=infer_from_data,
                        null_option=null_option,
                        mode=self.mode,
                        encoding=encoding,
                        multiline=encoding in ('UTF-8', 'UTF8', '')
                    ))
                    code.append(code_csv)
                else:
                    code_csv = dedent("""
                    schema_{output} = types.StructType()
                    schema_{output}.add('value', types.StringType(), True)
                    {output} = spark_session.read{null_option}.schema(
                        schema_{output}).option(
                        'treatEmptyValuesAsNulls', 'true').text(
                            url)""".format(output=self.output,
                                           null_option=null_option))
                    code.append(code_csv)
            elif self.metadata['format'] == 'PARQUET_FILE':
                # TO DO
                pass
            elif self.metadata['format'] == 'JSON':
                code_json = dedent("""
                    schema_{output} = types.StructType()
                    schema_{output}.add('value', types.StringType(), True)
                    {output} = spark_session.read.option(
                        'treatEmptyValuesAsNulls', 'true').json(
                        '{url}')""".format(output=self.output,
                                           url=self.metadata['url']))
                code.append(code_json)
                # FIXME: Evaluate if it is good idea to always use cache
                code.append('{}.cache()'.format(self.output))
                pass
            elif self.metadata['format'] == 'LIB_SVM':
                self._generate_code_for_lib_svm(code, infer_from_data)
            elif self.metadata['format'] == 'JDBC':
                self._generate_code_for_jdbc(code)

        if self.metadata.get('privacy_aware', False):
            restrictions = self.parameters['workflow'].get(
                'privacy_restrictions', {}).get(self.data_source_id)
            code.extend(self._apply_privacy_constraints(restrictions))

        code.append('{}.cache()'.format(self.output))
        return '\n'.join(code)

    def _generate_code_for_jdbc(self, code):

        parsed = urlparse(self.metadata['url'])
        qs_parsed = parse_qs(parsed.query)
        driver = self.SUPPORTED_DRIVERS.get(parsed.scheme)
        if driver is None:
            raise ValueError(
                _('Database {} not supported').format(parsed.scheme))
        if not self.metadata.get('command'):
            raise ValueError(
                _('No command nor table specified for data source.'))
        code_jdbc = dedent("""
            query = '{table}'
            if query.strip()[:6].upper() == 'SELECT':
                # Add parentheses required by Spark
                query = '(' + query + ') AS tb'
            {out} = spark_session.read.format('jdbc').load(
                driver='{driver}',
                url='jdbc:{scheme}://{server}:{port}{db}',
                user='{user}', password='{password}',
                dbtable=query)
                """.format(scheme=parsed.scheme, server=parsed.hostname,
                           db=parsed.path,
                           port=parsed.port,
                           driver=driver, user=qs_parsed.get('user', [''])[0],
                           password=qs_parsed.get('password', [''])[0],
                           table=self.metadata.get('command'),
                           out=self.output))
        code.append(code_jdbc)

    def _generate_code_for_lib_svm(self, code, infer_from_data):
        """"""
        # # FIXME: Evaluate if it is good idea to always use cache
        code.append(
            "url_{0} = '{1}'".format(self.output, self.metadata['url']))
        code_csv = """
            {0} = spark_session.read.format('{2}').load(
                url_{0}, mode='DROPMALFORMED')""".format(
            self.output, infer_from_data, 'libsvm')
        code.append(dedent(code_csv))

    def _apply_privacy_constraints(self, restrictions):
        result = [
            'import juicer.privaaas',
            'original_schema = {out}.schema'.format(out=self.output)
        ]
        try:
            if restrictions.get('attributes'):
                attrs = sorted(restrictions['attributes'],
                               key=lambda x: x['anonymization_technique'])
                grouped_by_type = itertools.groupby(
                    attrs, key=lambda x: x['anonymization_technique'])
                privacy_decorator = PrivacyPreservingDecorator(self.output)
                for k, group in grouped_by_type:
                    if k is not None and k != 'NO_TECHNIQUE':
                        if hasattr(privacy_decorator, k.lower()):
                            action = getattr(privacy_decorator, k.lower())
                            action_result = action(group)
                            if action_result:
                                result.append(action_result)
                        else:
                            raise ValueError(
                                _('Invalid anonymization type ({})').format(k))
                result.append(dedent("""
                    for attr in {out}.schema:
                        found = next(o for o in original_schema
                            if attr.name == o.name)
                        version = dataframe_util.spark_version(spark_session)
                        if found is not None and version >= (2, 2, 0):
                            {out} = {out}.withColumn(attr.name,
                                functions.col(attr.name).alias(attr.name,
                                    metadata=found.metadata))
                """.format(out=self.output)))
        except Exception as e:
            print e
            raise

        return result

    def _add_attribute_to_schema(self, attr, code):
        data_type = self.LIMONERO_TO_SPARK_DATA_TYPES[attr['type']]
        if attr['type'] in self.DATA_TYPES_WITH_PRECISION:
            data_type = '{}({}, {})'.format(
                data_type,
                attr['precision'] + 3,  # extra precision to be safe
                attr.get('scale', 0) or 0)
        else:
            data_type = '{}()'.format(data_type)

        # Notice: According to Spark documentation, nullable
        # option of StructField is just a hint and when
        # loading CSV file, it won't work. So, we are adding
        # this information in metadata.
        # Not used.
        # metadata = {k: attr[k] for k in
        #             ['nullable', 'type', 'size', 'precision', 'enumeration',
        #              'missing_representation'] if attr[k]}

        metadata = {'sources': [
            '{}/{}'.format(self.data_source_id, attr['name'])
        ]}

        code.append("schema_{0}.add('{1}', {2}, {3},\n{5}{4})".format(
            self.output, attr['name'], data_type, attr['nullable'],
            pprint.pformat(metadata, indent=0), ' ' * 20
        ))

    def get_output_names(self, sep=", "):
        return self.output

    def get_data_out_names(self, sep=','):
        return self.output


class SaveOperation(Operation):
    """
    Saves the content of the DataFrame at the specified path
    and generate the code to call the Limonero API.
    Parameters:
        - Database name
        - Path for storage
        - Storage ID
        - Database tags
        - Workflow that generated the database
    """
    SPARK_TO_LIMONERO_DATA_TYPES = {
        'types.StringType': "CHARACTER",
        'types.TimestampType': "DATETIME",
        'types.DoubleType': "DOUBLE",
        'types.DecimalType': "DECIMAL",
        'types.FloatType': "FLOAT",
        'types.LongType': "LONG",
        'types.IntegerType': "INTEGER",

        'StringType': "CHARACTER",
        'TimestampType': "DATETIME",
        'DoubleType': "DOUBLE",
        'DecimalType': "DECIMAL",
        'FloatType': "FLOAT",
        'LongType': "LONG",
        'IntegerType': "INTEGER",
    }

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

    FORMAT_PARQUET = 'PARQUET'
    FORMAT_CSV = 'CSV'
    FORMAT_JSON = 'JSON'
    WORKFLOW_JSON_PARAM = 'workflow_json'
    USER_PARAM = 'user'
    WORKFLOW_ID_PARAM = 'workflow_id'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = parameters.get(self.NAME_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM, '') or ''
        valid_formats = (self.FORMAT_PARQUET, self.FORMAT_CSV, self.FORMAT_JSON)
        if not self.format.strip() or self.format not in valid_formats:
            raise ValueError(_('You must specify a valid format.'))

        self.url = parameters.get(self.PATH_PARAM)
        self.storage_id = parameters.get(self.STORAGE_ID_PARAM)
        if not self.storage_id:
            raise ValueError(_('You must specify a storage for saving data.'))

        self.tags = ast.literal_eval(parameters.get(self.TAGS_PARAM, '[]'))
        self.path = parameters.get(self.PATH_PARAM)
        if self.path is None or not self.path.strip():
            raise ValueError(_('You must specify a path for saving data.'))

        self.workflow_json = parameters.get(self.WORKFLOW_JSON_PARAM, '')

        self.mode = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)
        self.header = parameters.get(self.HEADER_PARAM, True) in (1, '1', True)

        self.user = parameters.get(self.USER_PARAM)
        self.workflow_id = parameters.get(self.WORKFLOW_ID_PARAM)
        self.has_code = len(self.named_inputs) == 1

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        # Retrieve Storage URL

        limonero_config = \
            self.parameters['configuration']['juicer']['services']['limonero']
        url = '{}'.format(limonero_config['url'], self.mode)
        token = str(limonero_config['auth_token'])
        storage = limonero_service.get_storage_info(url, token, self.storage_id)

        final_url = '{}/limonero/user_data/{}/{}/{}'.format(
            storage['url'], self.user['id'], self.path,
            self.name.replace(' ', '_'))
        code_save = ''
        if self.format == self.FORMAT_CSV:
            code_save = dedent(u"""
            cols = []
            for attr in {input}.schema:
                if attr.dataType.typeName() in ['array']:
                    cols.append(functions.concat_ws(
                        ', ', {input}[attr.name]).alias(attr.name))
                else:
                    cols.append({input}[attr.name])

            {input} = {input}.select(*cols)
            mode = '{mode}'
            # Write in a temporary directory
            # Header configuration will be handled by LemondadeFileUtil class
            {input}.write.csv('{url}{uuid}',
                         header=False, mode=mode)
            # Merge files using Hadoop HDFS API
            conf = spark_session._jsc.hadoopConfiguration()
            jvm = spark_session._jvm
            fs = jvm.org.apache.hadoop.fs.FileSystem.get(
                jvm.java.net.URI('{storage_url}'), conf)

            path = jvm.org.apache.hadoop.fs.Path('{url}')
            tmp_path = jvm.org.apache.hadoop.fs.Path(
                '{url}{uuid}')
            write_header = {header}
            # org.apache.hadoop.fs.FileUtil do not handle files with header
            header = None
            if write_header:
                header = ','.join([attr.name for attr in {input}.schema])

            fs_util = jvm.br.ufmg.dcc.lemonade.ext.io.LemonadeFileUtil
            if fs.exists(path):
                if mode == 'error':
                    raise ValueError('{error_file_exists}')
                elif mode == 'ignore':
                    emit_event(name='update task',
                        message='{warn_ignored}',
                        status='COMPLETED',
                        identifier='{task_id}')
                elif mode == 'overwrite':
                    fs.delete(path, False)
                    fs_util.copyMergeWithHeader(fs, tmp_path, fs, path, True,
                        conf, header)
                else:
                    raise ValueError('{error_invalid_mode}')
            else:
                fs_util.copyMergeWithHeader(fs, tmp_path, fs, path, True, conf,
                    header)
            """.format(
                input=self.named_inputs['input data'],
                url=final_url, header=self.header, mode=self.mode,
                uuid=uuid.uuid4().get_hex(),
                storage_url=storage['url'],
                task_id=self.parameters['task_id'],
                error_file_exists=_('File already exists'),
                warn_ignored=_('File not written (already exists)'),
                error_invalid_mode=_('Invalid mode {}').format(self.mode)
            ))
            # Need to generate an output, even though it is not used.
        elif self.format == self.FORMAT_PARQUET:
            code_save = dedent("""
            {}.write.parquet('{}', mode='{}')""".format(
                self.named_inputs['input data'],
                final_url, self.mode))
            # Need to generate an output, even though it is not used.
            code_save += '\n{0}_tmp = {0}'.format(
                self.named_inputs['input data'])
        elif self.format == self.FORMAT_JSON:
            code_save = dedent("""
            {}.write.json('{}', mode='{}')""".format(
                self.named_inputs['input data'],
                final_url, self.mode))

        code = dedent(code_save)

        code_api = u"""
            # Code to update Limonero metadata information
            from juicer.service.limonero_service import register_datasource
            types_names = {data_types}

            # nullable information is also stored in metadata
            # because Spark ignores this information when loading CSV files
            attributes = []
            decimal_regex = re.compile(r'DecimalType\((\d+),\s*(\d+)\)')
            for att in {input}.schema:
                type_name = unicode(att.dataType)
                precision = None
                scale = None
                found = decimal_regex.findall(type_name)
                if found:
                    type_name = 'DecimalType'
                    precision = found[0][0]
                    scale = found[0][1]
                attributes.append({{
                  'enumeration': 0,
                  'feature': 0,
                  'label': 0,
                  'name': att.name,
                  'type': types_names[str(type_name)],
                  'nullable': att.nullable,
                  'metadata': att.metadata,
                  'precision': precision,
                  'scale': scale
                }})
            parameters = {{
                'name': "{name}",
                'enabled': 1,
                'is_public': 0,
                'format': "{format}",
                'storage_id': {storage},
                'description': "{description}",
                'user_id': "{user_id}",
                'user_login': "{user_login}",
                'user_name': "{user_name}",
                'workflow_id': "{workflow_id}",
                'url': "{final_url}",
                'attributes': attributes
            }}
            register_datasource('{url}', parameters, '{token}', '{mode}')
            """.format(
            input=self.named_inputs['input data'],
            name=self.name,
            format=self.format,
            storage=self.storage_id,
            description=_('Data source generated by workflow {}').format(
                self.workflow_id),
            user_name=self.user['name'],
            user_id=self.user['id'],
            user_login=self.user['login'],
            workflow_id=self.workflow_id,
            final_url=final_url,
            token=token,
            url=url,
            mode=self.mode,
            data_types=json.dumps(self.SPARK_TO_LIMONERO_DATA_TYPES))
        code += dedent(code_api)
        # No return
        code += '{} = None'.format(self.output)

        return code


class ReadCSVOperation(Operation):
    """
    Reads a CSV file without HDFS.
    The purpose of this operation is to read files in
    HDFS without using the Limonero API.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.url = parameters['url']
        try:
            self.header = parameters['header']
        except KeyError:
            self.header = "True"
        try:
            self.separator = parameters['separator']
        except KeyError:
            self.separator = ";"

    def generate_code(self):
        code = """{} = spark_session.read.csv('{}',
            header={}, sep='{}',inferSchema=True)""".format(
            self.named_outputs['output data'], self.url, self.header,
            self.separator)
        return dedent(code)


class ChangeAttributeOperation(Operation):
    ATTRIBUTES_PARAM = 'attributes'
    IS_FEATURE_PARAM = 'is_feature'
    IS_LABEL_PARAM = 'is_label'
    NULLABLE_PARAM = 'nullable'
    NEW_NAME_PARAM = 'new_name'
    NEW_DATA_TYPE_PARAM = 'new_data_type'
    KEEP_VALUE = 'keep'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(_(
                "Parameter '{}' must be informed for task {}").format(
                self.ATTRIBUTES_PARAM, self.__class__))
        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):
        del self.parameters['workflow_json']
        code = []
        if self.parameters.get(self.NEW_DATA_TYPE_PARAM,
                               'keep') == self.KEEP_VALUE:
            # Do not require processing data frame, change only meta data
            code.append('{0} = {1}'.format(self.output,
                                           self.named_inputs['input data']))

            for attr in self.attributes:
                code.append(
                    "\ninx_{0} = [i for i, _ in enumerate({0}.schema) "
                    "if _.name.lower() == '{1}']".format(self.output,
                                                         attr.lower()))

                nullable = self.parameters.get(self.NULLABLE_PARAM,
                                               self.KEEP_VALUE)
                if nullable != self.KEEP_VALUE:
                    code.append(
                        ChangeAttributeOperation.change_meta(
                            self.output, attr, 'nullable', nullable == 'true'))

                feature = self.parameters.get(self.IS_FEATURE_PARAM,
                                              self.KEEP_VALUE)
                if feature != self.KEEP_VALUE:
                    code.append(
                        ChangeAttributeOperation.change_meta(
                            self.output, attr, 'feature', feature == 'true'))

                label = self.parameters.get(self.IS_LABEL_PARAM,
                                            self.KEEP_VALUE)
                if label != self.KEEP_VALUE:
                    code.append(
                        ChangeAttributeOperation.change_meta(
                            self.output, attr, 'label', label == 'true'))

            format_name = self.parameters[self.NEW_NAME_PARAM]
            if format_name:
                rename = [
                    "withColumnRenamed('{}', '{}')".format(
                        attr,
                        ChangeAttributeOperation.new_name(format_name, attr))
                    for
                    attr in self.attributes]

                code.append('{0} = {0}.{1}'.format(
                    self.output, '\\\n              .'.join(rename)))

        else:
            # Changing data type requires to rebuild data frame
            pass
        return '\n'.join(code)  # json.dumps(self.parameters)

    @staticmethod
    def new_name(format_new_name, name):
        return format_new_name.format(name)

    @staticmethod
    def change_meta(output, attr_name, meta_name, value):
        return dedent("if inx_{0}:\n"
                      "    {0}.schema.fields[inx_{0}[0]]"
                      ".metadata['{2}'] = {3}".format(output, attr_name,
                                                      meta_name, value))


class ExternalInputOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.output) > 0

    def generate_code(self):
        code = """{out} = None""".format(out=self.output)
        return code


class StreamConsumerOperation(DataReaderOperation):
    SOURCE_TYPE_PARAM = 'source_type'
    WINDOW_TYPE_PARAM = 'window_type'
    WINDOW_SIZE_PARAM = 'window_size'
    BROKER_URL_PARAM = 'broker_url'
    TOPIC_PARAM = 'topic'
    GROUP_PARAM = 'group'

    SOURCE_TYPES = ['kafka', 'hdfs']
    WINDOW_TYPES = ['seconds', 'size']

    def __init__(self, parameters, named_inputs, named_outputs):
        DataReaderOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.source_type = parameters.get(self.SOURCE_TYPE_PARAM, 'hdfs')
        if self.source_type not in self.SOURCE_TYPES:
            raise ValueError(_("Invalid value '{}' for parameter '{}'".format(
                self.source_type, self.SOURCE_TYPE_PARAM)))

        self.window_type = parameters.get(self.WINDOW_TYPE_PARAM, 'seconds')
        if self.window_type not in self.WINDOW_TYPES:
            raise ValueError(_("Invalid value '{}' for parameter '{}'".format(
                self.window_type, self.WINDOW_TYPE_PARAM)))

        self.broker_url = parameters.get(self.BROKER_URL_PARAM)
        if not self.broker_url:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.BROKER_URL_PARAM, self.__class__))

        self.window_size = parameters.get(self.WINDOW_SIZE_PARAM, 5)
        self.topic = parameters.get(self.TOPIC_PARAM)
        self.group = parameters.get(self.GROUP_PARAM)

        self.stream_context_ref = parameters.get('stream_context')
        self.has_code = len(self.output) > 0

    def generate_code(self):

        code = []
        if 'attributes' in self.metadata:
            code.append('schema_{0} = types.StructType()'.format(self.output))
            attrs = self.metadata.get('attributes')
            for attr in attrs:
                self._add_attribute_to_schema(attr, code)
            else:
                v = "schema_{0}.add('value', types.StringType(), 1, None)"
                code.append(v.format(self.output))
            code.append("")
        else:
            raise ValueError(
                _("Metadata do not include attributes information"))

        code.append(dedent("""
            from pyspark.streaming.kafka import KafkaUtils

            source_type = '{source_type}'
            window_type = '{window_type}'
            scc = {stream_ctx}

            if source_type == 'kafka':
                kvs = KafkaUtils.createStream(
                    ssc, broker, '{group}', {{'{topic}': 1}})
            lines = kvs.map(lambda x: x[1])

            counts = lines.flatMap(
                lambda line: [w.lower() for w in line.split() if w.strip()]).map(
                lambda word: (word, 1)).reduceByKey(
                lambda a, b: a + b).transform(
                lambda rdd: rdd.sortBy(lambda x: x[1], False))
        """.format(source_type=self.source_type, window_type=self.window_type,
                   stream_ctx=self.stream_context_ref, topic=self.topic,
                   group=self.group)))
        return code

    @property
    def is_stream_consumer(self):
        return True
