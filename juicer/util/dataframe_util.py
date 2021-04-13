# coding=utf-8


import decimal
import json

import datetime

import re
import simplejson
from six import text_type
from collections.abc import Sequence
import collections


def is_numeric(schema, col):
    import pyspark.sql.types as spark_types
    from pyspark.ml.linalg import VectorUDT
    return isinstance(schema[str(col)].dataType, spark_types.NumericType) or \
        isinstance(schema[str(col)].dataType, VectorUDT) 


def default_encoder_sklearn(obj):
    if isinstance(obj, float):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    else:
        return str(obj)


def default_encoder(obj):
    from pyspark.ml.linalg import DenseVector
    if isinstance(obj, decimal.Decimal):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, DenseVector):
        return list(obj)
    else:
        return str(obj)


class SimpleJsonEncoder(simplejson.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        return default_encoder(obj)


class SimpleJsonEncoderSklearn(simplejson.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        return default_encoder_sklearn(obj)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        return default_encoder(obj)


def get_csv_schema(df, only_name=False):
    return ','.join(get_schema_fmt(df, only_name))


def get_csv_schema_sklearn(df, only_name=False):
    return ','.join(get_schema_fmt_sklearn(df, only_name))


def get_schema_fmt(df, only_name=False):
    if only_name:
        return [f.name for f in df.schema.fields]
    else:
        return ['{}:{}'.format(f.dataType, f.name) for f in df.schema.fields]


def get_schema_fmt_sklearn(df, only_name=False):
    if only_name:
        return list(df.columns)
    else:
        return ['{}:{}'.format(i, str(f))
                for i, f in zip(df.columns, df.dtypes)]


def get_dict_schema(df):
    return [dict(type=f.dataType, name=f.name) for f in df.schema.fields]


def with_column_index(sdf, name):
    import pyspark.sql.types as spark_types
    new_schema = spark_types.StructType(sdf.schema.fields + [
        spark_types.StructField(name, spark_types.LongType(), False), ])
    return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(
        schema=new_schema)


def convert_to_csv(row):
    result = []
    for v in row:
        if isinstance(v, datetime.datetime):
            result.append(v.isoformat())
        elif isinstance(v, (str, text_type)):
            result.append('"{}"'.format(v))
        else:
            result.append(str(v))
    return ','.join(result)


def convert_to_python(row):
    result = []
    for v in row:
        t = type(v)
        if t in [datetime.datetime, datetime.date]:
            result.append(v.isoformat())
        else:
            result.append(v)
    return result


def format_row_for_visualization(row):
    date_types = [datetime.datetime, datetime.date]
    if len(row) == 2:
        # Use first column as id and name
        value = row[1] if type(row[1]) not in date_types else row[1].isoformat()
        _id = row[0]
        name = row[0]
    elif len(row) == 3:
        # Use first column as id and name
        value = row[2] if type(row[2]) not in date_types else row[2].isoformat()
        _id = row[0]
        name = row[0]
    else:
        raise ValueError(_('Invalid input data for visualization. '
                           'It should contains 2 (name, value) or '
                           '3 columns (id, name, value).'))
    return dict(id=_id, name=name, value=value)


def format_row_for_bar_chart_visualization(row):
    date_types = [datetime.datetime, datetime.date]
    if len(row) == 2:
        # Use first column as id and name
        value = row[1] if type(row[1]) not in date_types else row[1].isoformat()
        _id = row[0]
        name = row[0]
    elif len(row) == 3:
        # Use first column as id and name
        value = row[2] if type(row[2]) not in date_types else row[2].isoformat()
        _id = row[0]
        name = row[0]
    else:
        raise ValueError(_('Invalid input data for visualization. '
                           'It should contains 2 (name, value) or '
                           '3 columns (id, name, value).'))
    return dict(id=_id, name=name, value=value)


def emit_schema(task_id, df, emit_event, name, notebook=False):
    from juicer.spark.reports import SimpleTableReport
    headers = [_('Attribute'), _('Type'), _('Metadata (Spark)')]
    rows = [[f.name, str(f.dataType), json.dumps(f.metadata) if f else ''] for f
            in df.schema.fields]
    css_class = 'table table-striped table-bordered w-auto' \
        if not notebook else ''
    content = SimpleTableReport(
        css_class, headers, rows, _('Schema for {}').format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=_('Schema for {}').format(name),
               task={'id': task_id})


def emit_schema_sklearn(task_id, df, emit_event, name, notebook=False):
    from juicer.spark.reports import SimpleTableReport
    headers = [_('Attribute'), _('Type')]
    rows = [[i, str(f)] for i, f in zip(df.columns, df.dtypes)]
    css_class = 'table table-striped table-bordered w-auto' \
        if not notebook else ''
    content = SimpleTableReport(
        css_class, headers, rows, _('Schema for {}').format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=_('Schema for {}').format(name),
               task={'id': task_id})


def emit_sample(task_id, df, emit_event, name, size=50, notebook=False,
                title=None):
    from juicer.spark.reports import SimpleTableReport
    headers = [f.name for f in df.schema.fields]

    number_types = (int, float, decimal.Decimal)

    rows = []
    for row in df.take(size):
        new_row = []
        rows.append(new_row)
        for col in row:
            if isinstance(col, (str, text_type)):
                value = col
            elif isinstance(col, (datetime.datetime, datetime.date)):
                value = col.isoformat()
            elif isinstance(col, number_types):
                value = str(col)
            else:
                value = json.dumps(col, cls=CustomEncoder)
            # truncate column if size is bigger than 200 chars.
            if len(value) > 200:
                value = value[:150] + ' ... ' + value[-50:]
            new_row.append(value)

    css_class = 'table table-striped table-bordered w-auto' \
        if not notebook else ''
    if title is None:
        title = _('Sample data for {}').format(name)
    content = SimpleTableReport(css_class, headers, rows, title, numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=title,
               task={'id': task_id})

def is_float_or_null(v):
    try:
        if v is None:
            return True
        float(v)
        return True
    except:
        return False

def emit_sample_sklearn(task_id, df, emit_event, name, size=50, notebook=False, 
        describe=False, infer=False, use_types=None):

    import io
    import pandas as pd
    from collections import defaultdict
    from pandas.api import types 
        
    result = {}
    type_mappings = {'Int64': 'Integer', 'Float64': 'Decimal', 'object': 'Text',
            'datetime64[ns]': 'Datetime', 'float64': 'Decimal', 'int64': 'Integer',
            'bool': 'Boolean', 'array': 'Array'}

    df2 = df
    # Decide which data frame to use.
    # UI may require original data, represented as string and 
    # without casting. But in order to describe the data set,
    # a inferred data frame (or a user provided types) must be 
    # used.
    if describe and False:
        converters = {}
        if use_types:
            pandas_converters = {
                'Integer': lambda v: pd.to_numeric(v, errors='coerce'),
                'Decimal': lambda v: pd.to_numeric(v, errors='coerce'),
                'DateTime': lambda v: pd.to_datetime(v, errors='coerce'),
                'Boolean': bool,
            }
            # User provided types for attributes
            for attr, dtype in use_types.items():
                f = pandas_converters.get(dtype)
                if f:
                    converters[attr] = f
        if infer:
            # df.infer_objects() is not working.
            # save as CSV to infer data types
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            df2 = pd.read_csv(buf, converters=converters) 

    rows = []

    number_types = (int, float, decimal.Decimal)

    truncated = set()
    missing = defaultdict(list)
    invalid = defaultdict(list)

    dtypes = df2.dtypes[:]
    for y, (label, row) in enumerate(df.head(size).iterrows()):
        new_row = []
        for x, col in enumerate(df.columns):

            col_value = row[col]
            col_py_type = type(col_value)
            if (col_py_type != list and (
                    pd.isnull(row[col]) or (not row[col] and row[col] != 0))):
                missing[y].append(x)
                new_row.append('')
                continue

            if types.is_datetime64_dtype(df[col].dtypes):
                value = row[col].isoformat()
            elif types.is_numeric_dtype(col_py_type):
                if not types.is_integer_dtype(col_py_type):
                    value = round(row[col], 8)
                else:
                    value = row[col]
            elif types.is_datetime64_any_dtype(col_py_type): # list of dates
                value = '[' + ','.join(['"{}'.format(d.isoformat()) 
                    for d in row[col]]) + ']'
            elif isinstance(col_value, Sequence) and not isinstance(col_value, 
                    (str, bytes, bytearray)):
                value = '[' + ', '.join([str(x) if isinstance(x, number_types)
                                   else "'{}'".format(x) for x in row[col]]) + ']'
                dtypes[x] = 'array'
            elif types.is_string_dtype(col_py_type):
                # truncate column if size is bigger than 200 chars.
                value = row[col]
                if len(value) > 60:
                    value = value[:60] + ' (trunc.)'
                    truncated.add(col)
            else:
                value = json.dumps(row[col], cls=CustomEncoder)

            new_row.append(value)
        rows.append(new_row)
    
    result['attributes'] = [{'label': i, 'key': i, 
        'type': type_mappings.get(str(f), str(f))} 
            for i, f in zip(df2.columns, dtypes)]

    result['rows'] = rows
    if describe:
        missing_counters = df2.isna().sum().to_dict()    
        result['total'] = len(df)
        for attr in result['attributes']:
            attr['missing_count'] = missing_counters.get(attr['key'], 0)
            attr['count'] = result['total']
            attr['invalid_count'] = 0
            


        pandas_converters = {
            'Integer': lambda v: v.isdigit(),
            'Decimal': lambda v: is_float_or_null(v),
            'DateTime': lambda v: True,
            'Boolean': lambda v: True,
        }
        for i, attr in enumerate(df.columns):
            f = None
            if pd.api.types.is_integer_dtype(df[attr].dtype):
                f = pandas_converters.get('Integer')
            elif pd.api.types.is_numeric_dtype(df[attr].dtype):
                f = pandas_converters.get('Decimal')
        
            if f:
                df_invalid = df[~df[attr].apply(is_float_or_null) & df[attr].notnull()]
                result['attributes'][i]['invalid_count'] = len(df_invalid)
                invalid[attr].extend(df_invalid[:size].index.to_numpy())

        result['missing'] = missing 
        result['invalid'] = invalid
        result['truncated'] = list(truncated)

    print('*' * 20)
    print(dtypes)
    print('*' * 20)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=result,
               type='OBJECT', title=_('Sample data for {}').format(name),
               task={'id': task_id})


def old_emit_sample_sklearn(task_id, df, emit_event, name, size=50, notebook=False):
    from juicer.spark.reports import SimpleTableReport
    headers = list(df.columns)

    number_types = (int, float, decimal.Decimal)
    rows = []

    for row in df.head(size).values:
        new_row = []
        rows.append(new_row)
        for col in row:
            if isinstance(col, str):
                value = col
            elif isinstance(col, (datetime.datetime, datetime.date)):
                value = col.isoformat()
            elif isinstance(col, number_types):
                value = str(col)
            elif isinstance(col, list):
                value = '[' + ', '.join([str(x) if isinstance(x, number_types)
                                   else "'{}'".format(x) for x in col]) + ']'
            else:
                value = json.dumps(col, cls=CustomEncoder)
            # truncate column if size is bigger than 200 chars.
            if len(value) > 200:
                value = value[:150] + ' ... ' + value[-50:]
            new_row.append(value)

    css_class = 'table table-striped table-bordered w-auto' \
        if not notebook else ''
    content = SimpleTableReport(
        css_class, headers, rows, _('Sample data for {}').format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=_('Sample data for {}').format(name),
               task={'id': task_id})


class LazySparkTransformationDataframe(object):
    """
     Wraps a Spark Model in order to perform lazy transformation
    """

    def __init__(self, model, df, load_op):
        self.model = model
        self.df = df
        self.transformed_df = None
        self.load_op = load_op

    def __getitem__(self, item):
        if self.transformed_df is None:
            self.transformed_df = self.load_op(self.df)
        return self.transformed_df[item]

    def __getattr__(self, name):
        if self.transformed_df is None:
            self.transformed_df = self.load_op(self.df)
        return getattr(self.transformed_df, name)


class SparkObjectProxy(object):
    """
     Used to wrap Spark objects's methods calls and execute custom code before
     and/or after original method execution
    """

    def __init__(self, data_frame, **kwargs):
        """
        Wraps PySpark objects's methods with an implementation.
          :param data_frame Object being wrapped
          :param kwargs Contains method's name and a tuple with new
                        implementation (before/after, if None, ignored).
        """
        self.wrapped_obj = data_frame
        self.methods = kwargs

    def __getattr__(self, name):
        member_to_call = getattr(self.wrapped_obj, name)

        def wrapper(*args, **kwargs):
            if name in self.methods or '__any__' in self.methods:
                target = self.wrapped_obj

                # before method call
                if all(['__any__' in self.methods,
                        self.methods['__any__'][0] is not None]):
                    r = self.methods['__any__'][0](self.wrapped_obj)
                    if r is not None:  # side-effect, new target
                        target = r

                if name in self.methods and self.methods[name][0] is not None:
                    r = self.methods[name][0](self.wrapped_obj)
                    if r is not None:  # side-effect, new target
                        target = r

                method_to_call = getattr(target, name)
                if isinstance(method_to_call, collections.Callable):
                    result = method_to_call(*args, **kwargs)
                else:
                    result = method_to_call

                # after method call
                if all(['__any__' in self.methods,
                        self.methods['__any__'][1] is not None]):
                    r = self.methods['__any__'][1](self.wrapped_obj)
                    if r is not None:
                        result = r

                if name in self.methods and self.methods[name][1] is not None:
                    r = self.methods[name][1](self.wrapped_obj)
                    if r is not None:
                        result = r
                return result
            else:
                method_to_call = getattr(self.wrapped_obj, name)
                if isinstance(method_to_call, collections.Callable):
                    return method_to_call(*args, **kwargs)
                else:
                    return method_to_call

        return wrapper if isinstance(member_to_call, collections.Callable) else member_to_call


def spark_version(spark_session):
    return tuple(map(int, spark_session.version.split('.')))


def merge_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def handle_spark_exception(e):
    from pyspark.sql.utils import AnalysisException, IllegalArgumentException
    result = False

    if isinstance(e, AnalysisException):
        value_expr = re.compile(r'[`"](.+)[`"].+columns:\s(.+)$')
        found = value_expr.findall(e.desc.split('\n')[0])
        if found:
            field, fields = found[0]
            raise ValueError(
                _('Attribute {} not found. Valid attributes: {}').format(
                    field, fields.replace(';', '')))
        else:
            value_expr = re.compile(r'The data type of the expression in the '
                                    r'ORDER BY clause should be a numeric type')
            found = value_expr.findall(e.desc.split('\n')[0])
            if found:
                raise ValueError(
                    _('When using Window Operation with range type, the order '
                      'by attribute must be numeric.'))
            found = 'This Range Window Frame only accepts ' \
                    'at most one ORDER BY' in e.desc
            if found:
                raise ValueError(
                    _('When using Window Operation with range type, the order '
                      'option must include only one attribute.'))
            found = 'Path does not exist' in e.desc
            if found:
                raise ValueError(
                    _('Data source does not exist. It may have been deleted.'))
            value_expr = re.compile(r'Table or view not found: (.+?);')
            found = value_expr.findall(e.desc.split('\n')[0])
            if found:
                raise ValueError(
                        _('Table or view not found: {}').format(found[0]))
    elif isinstance(e, KeyError):
        value_expr = re.compile(r'No StructField named (.+)\'$')
        found = value_expr.findall(str(e))
        if found:
            raise ValueError(
                _('Attribute {} not found.').format(found[0]))
    elif isinstance(e, IllegalArgumentException):
        # Invalid column type
        if 'must be of type equal' in str(e):
            value_expr = re.compile(
                "requirement failed: Column (.+?) must be"
                ".+following types: \[(.+?)\] but was actually of type (.+).")
            found = value_expr.findall(e.desc)
            if found:
                attr, correct, used = found[0]
                raise ValueError(_('Attribute {attr} must be one of these types'
                                   ' [{correct}], but it is {used}').format(
                    attr=attr, used=used, correct=correct
                ))
        elif 'Available fields' in str(e):
            value_expr = re.compile(
                r'Field "(.+?)" does not exist.\nAvailable fields: (.+)',
                re.MULTILINE)
            found = value_expr.findall(e.desc)
            if found:
                used, correct = found[0]
                raise ValueError(
                    _('Attribute {} not found. Valid attributes: {}').format(
                        used, correct))
        elif 'Binomial family only supports' in str(e):
            value_expr = re.compile(
                r'outcome classes but found (\d+)',
                re.MULTILINE)
            found = value_expr.findall(e.desc)
            if found:
                total = found[0]
                raise ValueError(
                    _('Binomial family only supports 1 or 2 outcome '
                      'classes but found {}').format(total))
        else:
            raise ValueError(e.desc)
    elif hasattr(e, 'java_exception'):
        se = 'org.apache.spark.SparkException'
        cause = e.java_exception.getCause()
        if cause and cause.getClass().getName() != se:
            if 'unwrapRemoteException' in dir(cause):
                cause = cause.unwrapRemoteException()
            else:
                while cause is not None and cause.getCause() is not None:
                    cause = cause.getCause()
        if cause is not None:
            nfe = 'java.lang.NumberFormatException'
            uoe = 'java.lang.UnsupportedOperationException'
            npe = 'java.lang.NullPointerException'
            bme = 'org.apache.hadoop.hdfs.BlockMissingException'
            ace = 'org.apache.hadoop.security.AccessControlException'
            iae = 'java.lang.IllegalArgumentException'

            cause_msg = cause.getMessage()
            inner_cause = cause.getCause()
            if cause.getClass().getName() == nfe and cause_msg:
                value_expr = re.compile(r'.+"(.+)"')
                value = value_expr.findall(cause_msg)[0]
                raise ValueError(_('Invalid numeric data in at least one '
                                   'data source (value: {})').format(
                    value).encode('utf8'))
            elif 'Malformed' in cause_msg:
                raise ValueError(_('At least one input data source is not in '
                                   'the correct format.'))
            elif inner_cause and inner_cause.getClass().getName() == npe:
                if cause_msg and 'createTransformFunc' in cause_msg:
                    raise ValueError(_('There is null values in your data set '
                                       'and Spark cannot handle them. '
                                       'Please, remove them before applying '
                                       'a data transformation.'))
                pass
            elif cause.getClass().getName() == bme:
                raise ValueError(
                    _('Cannot read data from the data source. In this case, '
                      'it may be a configuration problem with HDFS. '
                      'Please, check if HDFS namenode is up and you '
                      'correctly configured the option '
                      'dfs.client.use.datanode.hostname in Juicer\' config.'))
            elif cause.getClass().getName() == ace:
                raise ValueError(
                    _('You do not have permissions to read or write in the '
                      'storage. Probably, it is a configuration problem. '
                      'Please, contact the support.')
                )
            elif cause.getClass().getName() == iae:
                gbt_error = 'dataset with invalid label'
                if cause_msg is not None and gbt_error in cause_msg:
                    raise ValueError(_('GBT classifier requires labels '
                                       'to be in [0, 1] range.'))
                else:
                    raise ValueError(cause_msg)
        elif e.java_exception.getMessage():
            value_expr = re.compile(r'CSV data source does not support '
                                    r'(.+?) data type')
            value = value_expr.findall(e.java_exception.getMessage())
            if value:
                raise ValueError(
                    _('CSV format does not support the data type {}. '
                      'Try to convert the attribute to string (see to_json()) '
                      'before saving.'.format(value[0])))
    return result


def df_zip_with_index(df, offset=1, name="row_id"):
    import pyspark.sql.types as spark_types
    new_schema = spark_types.StructType(
        [spark_types.StructField(name, spark_types.LongType(),
                                 True)] + df.schema.fields
    )

    zipped_rdd = df.rdd.zipWithIndex()

    return zipped_rdd.map(
        lambda row_row_id: ([row_row_id[1] + offset] + list(row_row_id[0]))).toDF(new_schema)
