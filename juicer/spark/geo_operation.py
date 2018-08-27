# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
from itertools import izip_longest
from textwrap import dedent

from juicer.operation import Operation
from juicer.spark.data_operation import DataReaderOperation


class ReadShapefile(DataReaderOperation):
    """
    Reads a shapefile.
    Parameters:
        - File location
        - List of target columns with position and alias
        - Row filter expression
    """
    DATA_SOURCE_ID_PARAM = 'shapefile'

    def __init__(self, parameters, named_inputs, named_outputs):
        DataReaderOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        if self.DATA_SOURCE_ID_PARAM in parameters:
            self._set_data_source_parameters(parameters)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}".format(
                    self.DATA_SOURCE_ID_PARAM, self.__class__)))
        self.output = self.named_outputs.get('geodata',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        """ We still have to add a parameter to define whether the points are
        expressed as (LAT,LON) or (LON,LAT). This will change the way the points
        are read.

        LON,LAT:
            points.append([point[0], point[1]])
        LAT,LON:
            points.append([point[1], point[0]])
        """
        code = """
            import shapefile
            from io import BytesIO
            # reload(sys)
            # sys.setdefaultencoding('utf-8')
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
            header = {attrs}
            header = types.StructType(
                [types.StructField(h, types.StringType(), False)
                             for h in header])
            header.add(types.StructField('points', types.ArrayType(
                types.ArrayType(types.DoubleType()))))
            data = []
            for shape_record in records:
                data.append(shape_record.record +
                    [shape_record.shape.points])
            {out} = spark_session.createDataFrame(data, header)
        """.format(url=self.metadata['url'],
                   attrs=json.dumps([a['name'] for a in
                                     self.metadata.get('attributes', [])]),
                   out=self.output)

        return dedent(code)


class GeoWithin(Operation):
    POLYGON_POINTS_COLUMN_PARAM = 'polygon'
    POLYGON_ATTRIBUTES_COLUMN_PARAM = 'polygon_attributes'
    POLYGON_ALIAS_COLUMN_PARAM = 'alias'
    TARGET_LAT_COLUMN_PARAM = 'latitude'
    TARGET_LON_COLUMN_PARAM = 'longitude'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.polygon_column = parameters[self.POLYGON_POINTS_COLUMN_PARAM]
        self.attributes = parameters[self.POLYGON_ATTRIBUTES_COLUMN_PARAM]

        self.alias = [
            alias.strip() for alias in
            parameters.get(self.POLYGON_ALIAS_COLUMN_PARAM, '').split(',')]

        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_alias'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.lat_column = parameters[self.TARGET_LAT_COLUMN_PARAM]
        self.lon_column = parameters[self.TARGET_LON_COLUMN_PARAM]

        if len(self.lat_column) == 0 or len(self.lon_column) == 0 or len(
                self.polygon_column) == 0:
            raise ValueError(
                _('Values for latitude and longitude columns must be informed'))

    def generate_code(self):
        code = """
            from matplotlib.path import Path
            import pyqtree

            schema = [s.name for s in {0}.schema]
            shp_object = {0}.collect()
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
            within = {2}.withColumn(
                "polygon_position", udf_get_first_polygon(functions.col('{3}'),
                                                        functions.col('{4}')))
            {5} = within.select(within.columns +
                [within.polygon_position[i].alias(schema[i])
                    for i in xrange(shapefile_features_count)])
        """.format(self.named_inputs['geo data'], self.polygon_column[0],
                   self.named_inputs['input data'], self.lat_column[0],
                   self.lon_column[0], self.named_outputs['output data'])
        return dedent(code)
