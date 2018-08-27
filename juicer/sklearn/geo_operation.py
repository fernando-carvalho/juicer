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
        self.alias = parameters.get('polygon', 'points')
        self.lat_long = True
        self.attributes = []
        self.output = self.named_outputs.get(
                'geo data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        """Generate code."""

        code = """
        import shapefile
        from io import BytesIO, StringIO

        polygon = '{polygon}'
        lat_long = {lat_long}
        header = {header}

        shp_content = open('{shp}', "rb")
        dbf_content = open('{dbf}', "rb")
    
        # shp_io = BytesIO(shp_path)
        # dbf_io = BytesIO(dbf_path)

        shp_object = shapefile.Reader(shp=shp_content, dbf=dbf_content)
        records = shp_object.records()
        sectors = shp_object.shapeRecords()

        fields = dict()  # column: position
        for i, f in enumerate(shp_object.fields):
            fields[f[0]] = i
        del fields['DeletionFlag']

        if len(header) == 0:
            header = [f for f in fields]

        # position of each selected field
        num_fields = [fields[f] for f in header]

        header.append(polygon)
        data = []
        for i, sector in enumerate(sectors):
            attributes = []
            r = records[i]
            for t in num_fields:
                attributes.append(r[t-1])

            points = []
            for point in sector.shape.points:
                if lat_long:
                    points.append([point[1], point[0]])
                else:
                    points.append([point[0], point[1]])
            attributes.append(points)
            data.append(attributes)

        {out} = pd.DataFrame(data, columns=header)

        """.format(shp=self.shp_file,
                   dbf=self.dbf_file,
                   lat_long=self.lat_long,
                   polygon=self.alias,
                   header=self.attributes,
                   out=self.output)
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
        self.polygon_column = parameters.get(self.POLYGON_POINTS_COLUMN_PARAM, 'points')
        self.attributes = parameters.get(self.POLYGON_ATTR_COLUMN_PARAM, [])
        if len(self.attributes) == 0:
            self.attributes = []

        self.alias = parameters.get('alias', 'sector_position')

        if len(named_inputs) == 2:
            self.has_code = True
        else:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input data',  'geo data', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        """Generate code."""
        code = """
                                                         
        def GeoWithinOperation(data_input, shp_object):                                         
            import pyqtree
    
            lat_long = {lat_long}
            attributes = {att}
            if len(attributes) == 0:
                attributes = shp_object.columns
            alias = '{alias}'
            polygon_col = '{polygon}'
            col_lat = "{LAT}"
            col_long = "{LON}"
                              
            if lat_long:
                LAT_pos = 0
                LON_pos = 1
            else:
                LAT_pos = 1
                LON_pos = 0
    
            xmin = float('+inf')
            ymin = float('+inf')
            xmax = float('-inf')
            ymax = float('-inf')
    
            for i, sector in shp_object.iterrows():
                for point in sector[polygon_col]:
                    xmin = min(xmin, point[LON_pos])
                    ymin = min(ymin, point[LAT_pos])
                    xmax = max(xmax, point[LON_pos])
                    ymax = max(ymax, point[LAT_pos])
    
            # create the main bound box
            spindex = pyqtree.Index(bbox=[xmin, ymin, xmax, ymax])
    
            # than, insert all sectors bbox
            for inx, sector in shp_object.iterrows():
                points = []
                xmin = float('+inf')
                ymin = float('+inf')
                xmax = float('-inf')
                ymax = float('-inf')
                for point in sector[polygon_col]:
                    points.append((point[LON_pos], point[LAT_pos]))
                    xmin = min(xmin, point[LON_pos])
                    ymin = min(ymin, point[LAT_pos])
                    xmax = max(xmax, point[LON_pos])
                    ymax = max(ymax, point[LAT_pos])
                spindex.insert(item=inx, bbox=[xmin, ymin, xmax, ymax])
            
            from matplotlib.path import Path
            
            sector_position = []        
            if len(data_input) > 0:
                for i, point in data_input.iterrows():
                    y = float(point[col_lat])
                    x = float(point[col_long])
        
                    # (xmin,ymin,xmax,ymax)
                    matches = spindex.intersect([x, y, x, y])
        
                    for shp_inx in matches:
                        row = shp_object.loc[shp_inx]
                        polygon = Path(row[polygon_col])
                        if polygon.contains_point([y, x]):
                            content = [i] + row[attributes].tolist()
                            sector_position.append(content)
        
            if len(sector_position) > 0:
                attributes = ['index_geoWithin'] + list(attributes)
                cols = ["%s%s" % (a, alias) for a in attributes]
                tmp = pd.DataFrame(sector_position, columns=cols)
        
                key = 'index_geoWithin'+alias
                data_input = pd.merge(data_input, tmp,
                                      left_index=True, right_on=key)
                data_input = data_input.drop([key], axis=1)
            else:
                for a in [a + alias for a in attributes]:
                    data_input[a] = np.nan
        
            data_input = data_input.reset_index(drop=True)  
            return data_input

        {out} = GeoWithinOperation({input}, {shape})                                   
        """.format(shape=self.named_inputs['geo data'],
                   polygon=self.polygon_column[0],
                   att=self.attributes,  alias=self.alias,
                   input=self.named_inputs['input data'],
                   LAT=self.lat_column[0], LON=self.lon_column[0],
                   out=self.output, lat_long=True)
        return dedent(code)
