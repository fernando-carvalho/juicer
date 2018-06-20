# -*- coding: utf-8 -*-
from textwrap import dedent
from itertools import izip_longest
import re
from juicer.operation import Operation


class ReadShapefileOperation(Operation):
    """Reads a shapefile.

    Parameters:
        - File location
        - Alias
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = 'shapefile' in parameters
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('shapefile', self.__class__))

        self.shp_file = parameters['shapefile']
        if '.shp' not in self.shp_file:
            self.shp_file = self.shp_file+'.shp'

        self.dbf_file = re.sub('.shp$', '.dbf', self.shp_file)
        self.has_import = "from functions.geo.read_shapefile "\
                          "import ReadShapeFileOperation\n"
        self.alias = parameters.get('polygon', 'points')

        self.port = 9000
        self.host = 'localhost'
        self.lat_long = True
        self.attributes = []
        self.output = self.named_outputs.get(
                'geo data', 'output_data_{}'.format(self.order))

    def get_optimization_information(self):
        flags = {'one_stage': True,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['polygon'] = '{alias}'
        settings['shp_path'] = '{shp}'
        settings['dbf_path'] = '{dbf}'
        settings['lat_long'] = {lat_long}
        settings['attributes'] = {att}
        settings['host'] = '{host}'
        settings['port'] = {port}
        shapefile_data = ReadShapeFileOperation().transform(settings, numFrag)
        
        conf.append([])  # needed to optimization
        """.format(alias=self.alias, shp=self.shp_file, dbf=self.dbf_file,
                   lat_long=self.lat_long, host=self.host, port=self.port,
                   att=self.attributes, out=self.output)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {out} = ReadShapeFileOperation().transform_serial(input_data)
        """.format(out=self.output)
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        settings = dict()
        settings['polygon'] = '{alias}'
        settings['shp_path'] = '{shp}'
        settings['dbf_path'] = '{dbf}'
        settings['lat_long'] = {lat_long}
        settings['attributes'] = {att}
        settings['host'] = '{host}'
        settings['port'] = {port}
        {out} = ReadShapeFileOperation().transform(settings, numFrag)
        """.format(alias=self.alias, shp=self.shp_file, dbf=self.dbf_file,
                   lat_long=self.lat_long, host=self.host, port=self.port,
                   att=self.attributes, out=self.output)
        return dedent(code)


class GeoWithinOperation(Operation):
    """GeoWithinOperation.
    """

    POLYGON_POINTS_COLUMN_PARAM = 'polygon'
    POLYGON_ATTR_COLUMN_PARAM = 'polygon_attributes'
    TARGET_LAT_COLUMN_PARAM = 'latitude'
    TARGET_LON_COLUMN_PARAM = 'longitude'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        attributes = [self.TARGET_LAT_COLUMN_PARAM,
                      self.TARGET_LON_COLUMN_PARAM,
                      self.POLYGON_POINTS_COLUMN_PARAM]

        for att in attributes:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(att, self.__class__))

        self.lat_column = parameters[self.TARGET_LAT_COLUMN_PARAM]
        self.lon_column = parameters[self.TARGET_LON_COLUMN_PARAM]
        self.polygon_column = parameters[self.POLYGON_POINTS_COLUMN_PARAM]
        self.attributes = parameters.get(self.POLYGON_ATTR_COLUMN_PARAM, [])
        self.alias = parameters.get('alias', 'sector_position')

        if len(named_inputs) == 2:
            self.has_code = True
            self.has_import = \
                "from functions.geo.geo_within import GeoWithinOperation\n"
        else:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input data',  'geo data', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def get_optimization_information(self):
        flags = {'one_stage': True,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': True,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['lat_col'] = "{LAT}"
        settings['lon_col'] = "{LON}"
        settings['attributes'] = {ids}
        settings['alias'] = '{alias}'
        settings['polygon'] = '{polygon}'
        conf.append(GeoWithinOperation().preprocessing(settings, {shape}))
        """.format(shape=self.named_inputs['geo data'],
                   polygon=self.polygon_column[0],
                   ids=self.attributes,  alias=self.alias,
                   LAT=self.lat_column[0], LON=self.lon_column[0])
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {output} = GeoWithinOperation().transform_serial({input}, conf_X)
        """.format(output=self.output,  input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
            settings = dict()
            settings['lat_col'] = "{LAT}"
            settings['lon_col'] = "{LON}"
            settings['attributes'] = {ids}
            settings['alias'] = '{alias}'
            settings['polygon'] = '{polygon}'
            {out} = GeoWithinOperation().transform({input},
                                                   {shape}, settings, numFrag)
            """.format(shape=self.named_inputs['geo data'],
                       polygon=self.polygon_column[0],
                       ids=self.attributes,  alias=self.alias,
                       input=self.named_inputs['input data'],
                       LAT=self.lat_column[0], LON=self.lon_column[0],
                       out=self.output)
        return dedent(code)


class STDBSCANOperation(Operation):
    """STDBSCANOperation.

    Perform a ST-DBSCAN with the geospatial data
    REVIEW: 2017-10-20
    OK - Juicer / Tahiti / implementation
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.latCol = parameters.get('LAT', 0)
        self.lonCol = parameters.get('LON', 0)
        self.datetimeCol = parameters.get('DATETIME', 0)

        if any([self.latCol == 0, self.lonCol == 0, self.datetimeCol == 0]):
            raise ValueError(
                _('Parameters {}, {} and {} must be informed for task {}.')
                .format('Latitude', 'Longitude', 'Datetime', self.__class__))

        if any([len(self.latCol) > 1,
                len(self.lonCol) > 1,
                len(self.datetimeCol) > 1]):
            raise ValueError(
                _('Parameters {}, {} and {} must contain only '
                  'one field (in each one) for task {}.')
                .format('Latitude', 'Longitude', 'Datetime', self.__class__))

        self.predCol = parameters.get('alias', 'Cluster')
        self.minPts = parameters.get('min_sample', 5)
        self.spatialThr = parameters.get('spatial_threshold', 500)
        self.temporalThr = parameters.get('thresold_temporal', 60)

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.has_code = len(named_inputs) == 1
        if self.has_code:
            self.has_import = \
                'from functions.geo.stdbscan.stdbscan import STDBSCAN\n'


    def get_optimization_information(self):
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        """
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
            settings = dict()

            settings['spatial_threshold'] = {spatial}  # meters
            settings['temporal_threshold'] = {temporal} # minutes
            settings['minPts'] = {minPts}

            # columns
            settings['lat_col'] = '{LAT}'
            settings['lon_col'] = '{LON}'
            settings['datetime'] = '{datetime}'
            settings['predCol'] = '{predCol}'

            {output} = STDBSCAN().fit_predict({data_input}, settings, numFrag)
            """.format(data_input=self.named_inputs['input data'],
                       output=self.output, minPts=self.minPts,
                       spatial=self.spatialThr, temporal=self.temporalThr,
                       LAT=self.latCol[0], LON=self.lonCol[0],
                       datetime=self.datetimeCol[0], predCol=self.predCol)

        return dedent(code)
