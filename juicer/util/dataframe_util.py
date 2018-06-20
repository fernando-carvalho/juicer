# coding=utf-8
import decimal
import json

import datetime
import pyspark.sql.types as spark_types
from pyspark.ml.linalg import DenseVector

import re
import simplejson
import types


# noinspection PyProtectedMember
# noinspection PyProtectedMember
def add_metadata(column, alias, meta):
    from pyspark import SparkContext
    sc = SparkContext._active_spark_context
    from pyspark.sql import Column
    jmeta = sc._gateway.jvm.org.apache.spark.sql.types.Metadata
    return Column(
        getattr(column._jc, "as")(alias, jmeta.fromJson(json.dumps(meta))))


def is_numeric(schema, col):
    return isinstance(schema[str(col)].dataType, spark_types.NumericType)


def default_encoder(obj):
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
        return default_encoder(obj)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        return default_encoder(obj)


def get_csv_schema(df, only_name=False):
    return ','.join(get_schema_fmt(df, only_name))


def get_schema_fmt(df, only_name=False):
    if only_name:
        return [f.name for f in df.schema.fields]
    else:
        return ['{}:{}'.format(f.dataType, f.name) for f in df.schema.fields]


def get_dict_schema(df):
    return [dict(type=f.dataType, name=f.name) for f in df.schema.fields]


def with_column_index(sdf, name):
    new_schema = spark_types.StructType(sdf.schema.fields + [
        spark_types.StructField(name, spark_types.LongType(), False), ])
    return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(
        schema=new_schema)


def convert_to_csv(row):
    result = []
    for v in row:
        t = type(v)
        if t in [datetime.datetime]:
            result.append(v.isoformat())
        elif t in [unicode, str]:
            result.append(u'"{}"'.format(v))
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


def emit_schema(task_id, df, emit_event, name):
    from juicer.spark.reports import SimpleTableReport
    headers = [_('Attribute'), _('Type'), _('Metadata')]
    rows = [[f.name, str(f.dataType), json.dumps(f.metadata) if f else ''] for f
            in df.schema.fields]
    content = SimpleTableReport(
        'table table-striped table-bordered', headers, rows,
        _('Schema for {}').format(name),
        numbered=True)

    emit_event('update task', status='COMPLETED',
               identifier=task_id,
               message=content.generate(),
               type='HTML', title=_('Schema for {}').format(name),
               task={'id': task_id})


def emit_sample(task_id, df, emit_event, name, size=50):
    from juicer.spark.reports import SimpleTableReport
    headers = [f.name for f in df.schema.fields]

    number_types = (types.IntType, types.LongType,
                    types.FloatType, types.ComplexType, decimal.Decimal)

    rows = []
    for row in df.take(size):
        new_row = []
        rows.append(new_row)
        for col in row:
            if isinstance(col, str):
                value = col
            elif isinstance(col, unicode):
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

    content = SimpleTableReport(
        'table table-striped table-bordered', headers, rows,
        _('Sample data for {}').format(name),
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
                if callable(method_to_call):
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
                if callable(method_to_call):
                    return method_to_call(*args, **kwargs)
                else:
                    return method_to_call

        return wrapper if callable(member_to_call) else member_to_call


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
        found = value_expr.findall(unicode(e.desc.split('\n')[0]))
        if found:
            field, fields = found[0]
            raise ValueError(
                _('Attribute {} not found. Valid attributes: {}').format(
                    field, fields.replace(';', '')))
        else:
            value_expr = re.compile(r'The data type of the expression in the '
                                    r'ORDER BY clause should be a numeric type')
            found = value_expr.findall(unicode(e.desc.split('\n')[0]))
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
    elif isinstance(e, IllegalArgumentException):
        # Invalid column type
        if 'must be of type equal' in unicode(e.message):
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
    elif hasattr(e, 'java_exception'):
        cause = e.java_exception.getCause()
        while cause is not None and cause.getCause() is not None:
            cause = cause.getCause()

        if cause is not None:
            nfe = 'java.lang.NumberFormatException'
            uoe = 'java.lang.UnsupportedOperationException'
            npe = 'java.lang.NullPointerException'
            bme = 'org.apache.hadoop.hdfs.BlockMissingException'

            cause_msg = cause.getMessage()
            inner_cause = cause.getCause()
            if cause.getClass().getName() == nfe and cause_msg:
                value_expr = re.compile(r'.+"(.+)"')
                value = value_expr.findall(cause_msg)[0]
                raise ValueError(_('Invalid numeric data in at least one '
                                   'data source (value: {})').format(
                    value).encode('utf8'))
            elif cause_msg == u'Malformed CSV record':
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
