# coding=utf-8
from __future__ import absolute_import

import ast
from textwrap import dedent

from juicer.spark.data_operation import DataReaderOperation
from juicer.spark.geo_operation import ReadShapefile, GeoWithin
from mock import patch
from tests import compare_ast, format_code_comparison


def set_ds_param(self, params):
    self.metadata = {'updated': None, 'url': 'http://limonero/shape1.shp'}


@patch.object(DataReaderOperation, '_set_data_source_parameters',
              side_effect=set_ds_param, autospec=True)
def test_read_shapefile_success(mock_set_ds_param):
    params = {
        ReadShapefile.DATA_SOURCE_ID_PARAM: 2017,
    }
    n_out = {'geodata': 'output_1'}
    instance = ReadShapefile(params, named_inputs={}, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
         import shapefile
         from io import BytesIO
         shp_file = '{url}'
         dbf_file = re.sub('.shp$', '.dbf', shp_file)
         shp_content = spark_session.sparkContext.binaryFiles(
             shp_file).collect()
         dbf_content = spark_session.sparkContext.binaryFiles(
             dbf_file).collect()
         shp_io = BytesIO(shp_content[0][1])
         dbf_io = BytesIO(dbf_content[0][1])

         shp_object = shapefile.Reader(shp=shp_io, dbf=dbf_io)
         records = shp_object.records()
         records = shp_object.shapeRecords()
         header = []
         header = types.StructType(
             [types.StructField(h, types.StringType(), False)
                          for h in header])
         header.add(types.StructField('points', types.ArrayType(
             types.ArrayType(types.DoubleType()))))
         data = []
         for shape_record in records:
             data.append(shape_record.record +
                 [shape_record.shape.points])
         {output} = spark_session.createDataFrame(data, header)
            """.format(url=instance.metadata['url'],
                       output=n_out['geodata'], ))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_geo_within_success():
    params = {
        GeoWithin.POLYGON_POINTS_COLUMN_PARAM: 'polygon',
        GeoWithin.POLYGON_ATTRIBUTES_COLUMN_PARAM: ['attribute'],
        GeoWithin.POLYGON_ALIAS_COLUMN_PARAM: 'alias',
        GeoWithin.TARGET_LAT_COLUMN_PARAM: 'latitude',
        GeoWithin.TARGET_LON_COLUMN_PARAM: 'longitude'
    }
    n_out = {'output data': 'output_1'}
    n_in = {'input data': 'input_1', 'geo data': '{geo_data}'}
    instance = GeoWithin(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = dedent("""
         from matplotlib.path import Path
         import pyqtree

         schema = [s.name for s in {geo}.schema]
         shp_object = {geo}.collect()
         bcast_shapefile = spark_session.sparkContext.broadcast(
             shp_object)

         x_min = float('+inf')
         y_min = float('+inf')
         x_max = float('-inf')
         y_max = float('-inf')
         for i, polygon in enumerate(shp_object):
             for point in polygon['points']:
                 x_min = min(x_min, point[1])
                 y_min = min(y_min, point[0])
                 x_max = max(x_max, point[1])
                 y_max = max(y_max, point[0])
         #
         sp_index = pyqtree.Index(bbox=[x_min, y_min, x_max, y_max])
         for inx, polygon in enumerate(shp_object):
             points = []
             x_min = float('+inf')
             y_min = float('+inf')
             x_max = float('-inf')
             y_max = float('-inf')
             for point in polygon['points']:
                 points.append((point[0], point[1]))
                 x_min = min(x_min, point[0])
                 y_min = min(y_min, point[1])
                 x_max = max(x_max, point[0])
                 y_max = max(y_max, point[1])
             sp_index.insert(item=inx, bbox=[x_min, y_min, x_max, y_max])

         broad_casted_sp_index = spark_session.sparkContext.broadcast(
             sp_index)

         def get_first_polygon(lat, lng):
             x = float(lat)
             y = float(lng)
             bcast_index = broad_casted_sp_index.value
             # Here it uses longitude, latitude
             matches = bcast_index.intersect([y, x, y, x])

             for shp_inx in matches:
                 row = bcast_shapefile.value[shp_inx]
                 polygon = Path(row['points'])
                 # Here it uses longitude, latitude
                 if polygon.contains_point([y, x]):
                     return [col for col in row]
             return [None] * len(bcast_shapefile.value[0])
         shapefile_features_count= len(bcast_shapefile.value[0])
         udf_get_first_polygon = functions.udf(
             get_first_polygon, types.ArrayType(types.StringType()))
         within = input_1.withColumn(
             "polygon_position", udf_get_first_polygon(functions.col('l'),
                                                     functions.col('l')))
         {output} = within.select(within.columns +
             [within.polygon_position[i].alias(schema[i])
                 for i in xrange(shapefile_features_count)])
            """.format(output=n_out['output data'], geo=n_in['geo data']))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)
