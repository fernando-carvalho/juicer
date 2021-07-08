# coding=utf-8


import collections
import decimal
import itertools
import json
from collections import Iterable
from textwrap import dedent

import pandas as pd
import numpy as np
import datetime

from juicer import auditing
from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util import chunks
from juicer.util import dataframe_util
from juicer.util.dataframe_util import get_csv_schema_sklearn

COLORS_PALETTE = list(reversed([
    '#590400', '#720601', '#8F0701', '#9C241E', '#AD443F',  # red
    '#934400', '#BD5700', '#ED6E00', '#FF9030', '#FFA75C',  # orange
    '#936300', '#BD7F00', '#ED9F00', '#FFBB30', '#FFCA5C',  # yellow
    '#285900', '#347201', '#428F01', '#579C1E', '#71AD3F',  # green
    '#005559', '#016D72', '#01898F', '#1E969C', '#3FA8AD',  # blue
    '#072163', '#0C2D7F', '#113A9F', '#3054AD', '#506FBB', ]))  # purple
SHAPES = ['diamond', 'point', 'circle']

TRUE_VALS = [True, 1, '1']


def _get_color_palette(idx):
    return COLORS_PALETTE[(idx % 6) * 5 + ((idx // 6) % 5)]


def get_caipirinha_config(config, indentation=0):
    limonero_conf = config['juicer']['services']['limonero']
    caipirinha_conf = config['juicer']['services']['caipirinha']
    result = dedent("""
    # Basic information to connect to other services
    config = {{
        'juicer': {{
            'services': {{
                'limonero': {{
                    'url': '{limonero_url}',
                    'auth_token': '{limonero_token}'
                }},
                'caipirinha': {{
                    'url': '{caipirinha_url}',
                    'auth_token': '{caipirinha_token}',
                    'storage_id': {storage_id}
                }},
            }}
        }}
    }}""".format(
        limonero_url=limonero_conf['url'],
        limonero_token=limonero_conf['auth_token'],
        caipirinha_url=caipirinha_conf['url'],
        caipirinha_token=caipirinha_conf['auth_token'],
        storage_id=caipirinha_conf['storage_id'], )
    )
    if indentation:
        return '\n'.join(
            ['{}{}'.format(' ' * indentation, r) for r in result.split('\n')])
    else:
        return result


class PublishVisualizationOperation(Operation):
    """
    This operation receives one dataframe as input and one or many
    VisualizationMethodOperation and persists the transformed data
    (currently HBase) for forthcoming visualizations
    """
    TITLE_PARAM = 'title'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.title = parameters.get(self.TITLE_PARAM, '')
        self.has_code = len(self.named_inputs) == 1
        self.supports_cache = False
        self.icon = 'fa-question'

    """
    This operation represents a strategy for visualization and is used together
    with 'PublishVisOperation' to create a visualization dashboard
    """

    def get_generated_results(self):
        return []
        # return [
        #     {'type': ResultType.VISUALIZATION,
        #      'id': self.parameters['task']['id'],
        #      'icon': self.icon,
        #      'title': self.title,
        #      }
        # ]

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return ''

    @property
    def get_inputs_names(self):
        if isinstance(self.named_inputs['visualizations'], (list, tuple)):
            return ', '.join(self.named_inputs['visualizations'])
        else:
            return ', '.join([self.named_inputs['visualizations']])

    def generate_code(self):
        # Create connection with storage, get visualization table and initialize
        # list for visualizations metadata
        code_lines = [
            "from juicer.service import caipirinha_service",
            "from juicer.util.dataframe_util import SimpleJsonEncoderSklearn as enc",
            "visualizations = []"
        ]
        if isinstance(self.named_inputs['visualizations'], (list, tuple)):
            visualizations = self.named_inputs['visualizations']
        else:
            visualizations = [self.named_inputs['visualizations']]

        for vis_model in visualizations:
            code_lines.append(dedent("""
            visualizations.append({{
                'job_id': '{job_id}',
                'task_id': {vis_model}.task_id,
                'title': {vis_model}.title ,
                'type': {{
                    'id': {vis_model}.type_id,
                    'name': {vis_model}.type_name
                }},
                'data': simplejson.dumps(
                    {vis_model}.get_data(), cls=enc, ignore_nan=True),
                'model': {vis_model}
            }})
            """).format(job_id=self.parameters['job_id'], vis_model=vis_model))

        # Register this new dashboard with Caipirinha
        code_lines.append(get_caipirinha_config(self.config))
        code_lines.append(dedent("""
            caipirinha_service.new_dashboard(config, '{title}', {user},
                {workflow_id}, u'{workflow_name}',
                {job_id}, '{task_id}', visualizations, emit_event)
            """.format(
            title=self.title or 'Result for job ' + str(
                self.parameters.get('job_id', '0')),
            user=self.parameters['user'],
            workflow_id=self.parameters['workflow_id'],
            workflow_name=self.parameters['workflow_name'],
            job_id=self.parameters['job_id'],
            task_id=self.parameters['task']['id']
        )))

        code = '\n'.join(code_lines)
        return dedent(code)


####################################################
# Visualization operations used to generate models #
####################################################

class VisualizationMethodOperation(Operation):
    TITLE_PARAM = 'title'
    COLUMN_NAMES_PARAM = 'column_names'
    ORIENTATION_PARAM = 'orientation'
    ID_ATTR_PARAM = 'id_attribute'
    VALUE_ATTR_PARAM = 'value_attribute'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        # TODO: validate parameters
        self.title = parameters.get(
            self.TITLE_PARAM, 'Result for job ' + str(
                self.parameters.get('job_id', '0')))
        self.column_names = [c.strip() for c in
                             parameters.get(self.COLUMN_NAMES_PARAM, [])]
        self.orientation = parameters.get(self.ORIENTATION_PARAM, '')
        self.id_attribute = parameters.get(self.ID_ATTR_PARAM, [])
        self.value_attribute = parameters.get(self.VALUE_ATTR_PARAM, [])

        # Visualizations are not cached!
        self.supports_cache = False
        self.output = self.named_outputs.get('visualization',
                                             'vis_task_{}'.format(self.order))

    def get_model_parameters(self):
        result = {}
        valid = ['x_axis_attribute', "y_title", "y_prefix", 'legend',
                 "y_suffix", "y_format", "x_title", "x_prefix", "x_suffix",
                 "x_format", "x_format", 'type',
                 'z_axis_attribute', 'z_title', 'z_prefix', 'z_suffix',
                 'z_format',
                 't_axis_attribute', 't_title', 't_prefix', 't_suffix',
                 't_format',
                 'latitude', 'longitude', 'value', 'label',
                 'y_axis_attribute', 'z_axis_attribute', 't_axis_attribute',
                 'series_attribute', 'extra_data', 'polygon', 'geojson_id',
                 'polygon_url']
        for k, v in list(self.parameters.items()):
            if k in valid:
                result[k] = v
        return result

    def get_output_names(self, sep=','):
        return self.output

    def get_model_name(self):
        NotImplementedError(_("Method get_model_name should be implemented "
                              "in {} subclass").format(self.__class__))

    def generate_code(self):
        if self.plain:
            return self._generate_plain_code()
        code_lines = [dedent(
            """
            from juicer.scikit_learn.vis_operation import {model}
            from juicer.util.dataframe_util import SimpleJsonEncoderSklearn as enc

            params = '{params}'
            {out} = {model}(
                {input}, '{task}', '{op}',
                '{op_slug}', '{title}',
                {columns},
                '{orientation}', {id_attr}, {value_attr},
                params=json.loads(params))
            """.format(out=self.output,
                       model=self.get_model_name(),
                       input=self.named_inputs['input data'],
                       task=self.parameters['task']['id'],
                       op=self.parameters['operation_id'],
                       op_slug=self.parameters['operation_slug'],
                       title=self.title,
                       columns=json.dumps(self.column_names),
                       orientation=self.orientation,
                       id_attr=self.id_attribute,
                       value_attr=self.value_attribute,
                       params=json.dumps(self.get_model_parameters() or {}),
                       ))]
        if len(self.named_outputs) == 0:
            # Standalone visualization, without a dashboard
            code_lines.append("from juicer.service import caipirinha_service")
            code_lines.append(get_caipirinha_config(self.config))
            code_lines.append(dedent("""
            visualization = {{
                'job_id': '{job_id}',
                'task_id': {out}.task_id,
                'title': {out}.title ,
                'type': {{
                    'id': {out}.type_id,
                    'name': {out}.type_name
                }},
                'model': {out},
                'data': json.dumps({out}.get_data(), cls=enc, ignore_nan=True),
            }}""").format(job_id=self.parameters['job_id'],
                          out=self.output))

            code_lines.append(dedent("""
            caipirinha_service.new_visualization(
                config,
                {user},
                {workflow_id}, {job_id}, '{task_id}',
                visualization, emit_event)
            """.format(
                user=self.parameters['user'],
                workflow_id=self.parameters['workflow_id'],
                job_id=self.parameters['job_id'],
                task_id=self.parameters['task']['id']
            )))
        return '\n'.join(code_lines)

    def _generate_plain_code(self):
        # model = klass = globals()[self.get_model_name()]
        # instance = klass(data=self.named_inputs['input data'],
        #                task_id=self.parameters['task']['id'],
        #                type_id=self.parameters['operation_id'],
        #                type_name=self.parameters['operation_slug'],
        #                title=self.title,
        #                column_names=json.dumps(self.column_names),
        #                orientation=self.orientation,
        #                id_attribute=self.id_attribute,
        #                value_attribute=self.value_attribute,
        #                params=json.dumps(self.get_model_parameters() or {})

        return "# TODO: Visualization code generation not implemented!"


class BarChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return BarChartModel.__name__


class PieChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return PieChartModel.__name__


class DonutChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return DonutChartModel.__name__


class LineChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return LineChartModel.__name__


class AreaChartOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return AreaChartModel.__name__


class TableVisualizationOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return TableVisualizationModel.__name__


class ScatterPlotOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return ScatterPlotModel.__name__


class MapOperation(VisualizationMethodOperation):
    def __init__(self, parameters, named_inputs, named_outputs):

        if parameters.get('type') in ['polygon', 'geojson']:
            limonero_config = parameters['configuration']['juicer']['services'][
                'limonero']
            url = limonero_config['url']
            token = str(limonero_config['auth_token'])

            metadata = limonero_service.get_data_source_info(
                url, token, parameters.get('polygon'))
            if not metadata.get('url'):
                raise ValueError(
                    _('Incorrect data source configuration (empty url or '
                      'not GEOJSON)'))
            else:
                parameters['polygon_url'] = metadata['url']
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)

    def get_model_name(self):
        return MapModel.__name__


class SummaryStatisticsOperation(VisualizationMethodOperation):
    ATTRIBUTES_PARAM = 'attributes'
    CORRELATION_PARAM = 'correlation'
    COMPLETE_ANALYSIS_PARAM = 'complete'

    def __init__(self, parameters, named_inputs, named_outputs):
        if not parameters.get(self.TITLE_PARAM):
            parameters[self.TITLE_PARAM] = 'Summary'
        VisualizationMethodOperation.__init__(self, parameters, named_inputs,
                                              named_outputs)
        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, None)

        self.correlation = parameters.get(self.CORRELATION_PARAM) in TRUE_VALS
        self.complete = parameters.get(self.COMPLETE_ANALYSIS_PARAM) \
            in TRUE_VALS

    def get_model_parameters(self):
        return {self.ATTRIBUTES_PARAM: self.attributes or [],
                self.CORRELATION_PARAM: self.correlation,
                self.COMPLETE_ANALYSIS_PARAM: self.complete}

    def get_model_name(self):
        return SummaryStatisticsModel.__name__


#######################################################
# Visualization Models used inside the code generated #
#######################################################

class VisualizationModel(object):
    def __init__(self, data, task_id, type_id, type_name, title, column_names,
                 orientation,
                 id_attribute, value_attribute, params):
        self.data = data
        self.task_id = task_id
        self.type_id = type_id
        self.type_name = type_name
        self.title = title
        self.column_names = column_names
        self.orientation = orientation
        self.params = params
        self.default_time_format = '%Y-%m-%d'

        if len(id_attribute) > 0 and isinstance(id_attribute, list):
            self.id_attribute = id_attribute[0]
        else:
            self.id_attribute = id_attribute

        self.value_attribute = value_attribute

    def get_data(self):
        raise NotImplementedError(_('Should be implemented in derived classes'))

    def get_schema(self):
        return self.data.schema.json()

    def get_icon(self):
        return 'fa-question-o'

    def get_column_names(self):
        return ""


class ChartVisualization(VisualizationModel):
    def get_data(self):
        raise NotImplementedError(_('Should be implemented in derived classes'))

    @staticmethod
    def _get_attr_type(attr):
        if attr == 'object':
            attr_type = 'text'
        elif attr == 'str':
            attr_type = 'text'
        elif attr.startswith('datetime'):
            attr_type = 'date'
        else:
            attr_type = 'number'

        return attr_type

    def _get_title_legend_tooltip(self):
        """ Common title and legend """
        return {
            "title": self.title,
            "legend": {
                "isVisible": True,
                "text": "{{name}}"
            },
            "tooltip": {
                "title": [
                    "{{name}}"
                ],
                "body": [
                    "<span class='metric'>{{x}}</span><span class='number'>{{y}}</span>"
                ]
            },
        }

    def _get_axis_info(self):
        schema = self.data.columns
        if not self.params.get('x_axis_attribute'):
            raise ValueError(_('X-axis attribute not specified'))
        x = self.params.get('x_axis_attribute')[0]
        x_attr = [c for c in schema if c == x]
        y_attrs = [c for c in schema if c in self.column_names]
        if len(x_attr):
            x_attr = x_attr[0]
        else:
            raise ValueError(
                _('Attribute {} for X-axis does not exist in ({})').format(
                    x, ', '.join([c for c in schema])))
        if len(y_attrs) == 0:
            raise ValueError(_(
                'At least one attribute for Y-axis does not exist: {}').format(
                ', '.join(self.params.get('column_names', []))))

        x_type = str(self.data[x_attr].dtype)
        x_type = ChartVisualization._get_attr_type(x_type)
        return x_attr, x_type, y_attrs

    @staticmethod
    def _format(value):
        if value is None:
            return None
        elif any([isinstance(value, datetime.datetime),
                  isinstance(value, datetime.date)]):
            return value.isoformat()
        elif isinstance(value, decimal.Decimal):
            return float(value)
        else:
            return value


class BarChartModel(ChartVisualization):
    """ Bar chart model for visualization of data """

    def get_icon(self):
        return 'fa-bar-chart'

    def get_data(self):
        x_attr, x_type, y_attrs = self._get_axis_info()

        colors = {}
        color_counter = 0
        for i, attr in enumerate(y_attrs):
            color = _get_color_palette(i)
            colors[attr] = {
                'fill': color,
                'gradient': color,
                'stroke': color,
            }
            color_counter = i

        result = {}
        result.update(self._get_title_legend_tooltip())

        # For barcharts this is right option
        result['legend']['text'] = '{{x}}'

        result.update({
            "x": {
                "title": self.params.get("x_title"),
                "type": x_type,
                "prefix": self.params.get("x_prefix"),
                "suffix": self.params.get("x_suffix"),
            },
            "y": {
                "equal": True,
                "title": self.params.get("y_title"),
                "prefix": self.params.get("y_prefix"),
                "suffix": self.params.get("y_suffix"),
                "format": self.params.get("y_format", {}).get('key'),
            },
            "data": []
        })

        if x_type in ['number']:
            result['x']['format'] = self.params.get("x_format", {}).get('key')
        elif x_type in ['timestamp', 'date', 'time']:
            # lets have this hardcoded for now
            result['x']["inFormat"] = self.default_time_format
            result['x']["outFormat"] = self.default_time_format

        rows_x = self.data[x_attr].to_numpy().tolist()
        rows_y = self.data[y_attrs].to_numpy().tolist()

        for inx_row, (row_x, row_y) in enumerate(zip(rows_x, rows_y)):
            x_value = row_x
            if x_value not in colors:
                inx_row += 1
                color = _get_color_palette(color_counter)
                colors[x_value] = {
                    'fill': color,
                    'gradient': color,
                    'stroke': color,
                }

            data = {
                'x': LineChartModel._format(x_value),
                'name': row_x,
                'key': row_x,
                'color': _get_color_palette(inx_row),
                'values': []
            }
            result['data'].append(data)
            for i, (attr, row) in enumerate(zip(y_attrs, row_y)):
                data['values'].append(
                    {
                        'x': attr,
                        'name': LineChartModel._format(x_value),
                        'y': LineChartModel._format(row),
                    }
                )
                if i >= 100:
                    raise ValueError(
                        _('The maximum number of values for x-axis is 100.'))

        result['colors'] = colors
        return result


class PieChartModel(ChartVisualization):
    """
    In PieChartModel, x_attr contains the label and y_attrs[0] contém os valores
    """

    def __init__(self, data, task_id, type_id, type_name, title, column_names,
                 orientation, id_attribute, value_attribute, params):
        ChartVisualization.__init__(self, data, task_id, type_id,
                                    params.get('type', 'pie-chart'),
                                    title, column_names, orientation,
                                    id_attribute, value_attribute, params)

    def get_icon(self):
        return 'fa-pie-chart'

    def _get_axis_info(self):
        schema = self.data.columns

        if self.id_attribute:
            label = self.id_attribute
        else:
            # Important to use only first item!
            label = self.value_attribute[0]

        value_attr = [c for c in schema if c == self.value_attribute[0]]
        if len(value_attr):
            value_attr = value_attr[0]
        else:
            raise ValueError(
                _('Attribute {} does not exist in ({})').format(
                    label, ', '.join([c for c in schema])))

        label_attr = [c for c in schema if c == label]
        if len(label_attr):
            label_attr = label_attr[0]
        else:
            raise ValueError(
                _('Attribute {} for label does not exist in ({})').format(
                    label, ', '.join([c for c in schema])))

        return label_attr, None, value_attr

    def get_data(self):
        label_attr, _, value_attr = self._get_axis_info()

        rows_x = self.data[value_attr].to_numpy()
        rows_y = self.data[label_attr].to_numpy()
        result = self._get_title_legend_tooltip()
        result['legend']['isVisible'] = self.params.get('legend') in ('1', 1)

        x_format = self.params.get("x_format", {})
        if not isinstance(x_format, dict):
            x_format = {'key': x_format}

        result.update({
            "x": {
                "title": self.params.get("x_title"),
                "value": "sum",
                "color": "#222",
                "prefix": self.params.get("x_prefix"),
                "suffix": self.params.get("x_suffix"),
                "format": x_format.get('key'),
            },
            "data": []

        })
        for i, (row_x, row_y) in enumerate(zip(rows_x, rows_y)):
            data = {
                'x': float(row_x),
                'value': float(row_x),
                'id': '{}_{}'.format(label_attr, i),
                'name': row_y,
                'label': row_y,
                'color': _get_color_palette(i),
            }
            result['data'].append(data)
            if i >= 100:
                raise ValueError(
                    _('The maximum number of values for this chart is 100.'))
        return result


class DonutChartModel(PieChartModel):
    def get_data(self):
        data = super(DonutChartModel, self).get_data()
        return data


class LineChartModel(ChartVisualization):
    def get_icon(self):
        return 'fa-line-chart'

    def get_data(self):
        x_attr, x_type, y_attrs = self._get_axis_info()

        data = []
        for i, attr in enumerate(y_attrs):
            data.append({
                "id": attr,
                "name": attr,
                "color": _get_color_palette(i),
                "pointColor": _get_color_palette(i),
                "pointShape": SHAPES[i % len(SHAPES)],
                "pointSize": 3,
                "values": []
            })

        result = {}
        result.update(self._get_title_legend_tooltip())

        result.update({
            "y": {
                "title": self.params.get("y_title"),
                "prefix": self.params.get("y_prefix"),
                "suffix": self.params.get("y_suffix"),
                "format": self.params.get("y_format", {}).get('key'),
            },
            "x": {
                "title": self.params.get("x_title"),
                "type": x_type,
                "prefix": self.params.get("x_prefix"),
                "suffix": self.params.get("x_suffix"),
            },
            "data": data
        })

        if x_type in ['number']:
            result['x']['format'] = self.params.get("x_format", {}).get('key')
        elif x_type == 'time':
            # FIXME: gViz does not handles datetime correctly
            result['x']['inFormat'] = '%Y-%m-%dT%H:%M:%S'
            result['x']['outFormat'] = '%Y-%m-%d'
        elif x_type in ['date']:
            result['x']["inFormat"] = self.default_time_format
            result['x']["outFormat"] = self.default_time_format
            result['x']["type"] = 'time'  # FIXME

        if self.data[x_attr].dtype.name.startswith('datetime'):
            rows_x = self.data[x_attr].to_numpy()\
                .astype('datetime64[s]').tolist()
        else:
            rows_x = self.data[x_attr].to_numpy().tolist()

        rows_y = self.data[y_attrs].to_numpy().tolist()

        for row_x, row_y in zip(rows_x, rows_y):
            for i, y in enumerate(row_y):
                data[i]['values'].append(
                    {
                        "x": LineChartModel._format(row_x),
                        "y": LineChartModel._format(y),
                    }
                )

        return result


class MapModel(ChartVisualization):
    def get_icon(self):
        return 'fa-map-marker'

    def get_data(self):
        result = {}
        result.update(self._get_title_legend_tooltip())
        rows = self.data.to_numpy().tolist()

        if self.params.get('value'):
            value_attr = next((c for c in self.data.columns if
                               c == self.params['value'][0]), None)
            value_type = ChartVisualization._get_attr_type(value_attr)
        else:
            value_type = 'number'

        param_map_type = self.params.get('type', 'heatmap')

        map_type = {
            'heatmap': 'heatmap',
            'points': 'points',
            'polygon': 'polygon'
        }[param_map_type]

        result['mode'] = {
            map_type: True
        }
        if param_map_type == 'polygon':
            result['geojson'] = {
                'url': self.params.get('polygon_url'),
                'idProperty': self.params.get('geojson_id', 'id') or 'id'
            }

        data = []
        result['data'] = data

        lat = self.params.get('latitude', [None])[0]
        lng = self.params.get('longitude', [None])[0]
        label = self.params.get('label', [None])[0]

        for i, row in self.data.iterrows():
            if self.params.get('value'):
                value = row[self.params.get('value')[0]]
            else:
                value = 0
            if param_map_type == 'polygon':

                info = {"id": row[label], "value": value}
                extra = self.params.get('extra_data', [])
                for f in extra:
                    if f in row:
                        info[f] = row[f]

            else:
                info = {
                    "id": str(i), "value": value,
                    "name": row[label] if label else None,
                }
                if lat and lng:
                    info["lat"] = row[lat]
                    info["lon"] = row[lng]

            data.append(info)

        return result


class AreaChartModel(LineChartModel):
    def get_icon(self):
        return 'fa-area-chart'


class ScatterPlotModel(ChartVisualization):
    """
    Scatter plot chart model
    """

    # noinspection PyArgumentEqualDefault
    def get_data(self):

        result = {}
        attrs = {}
        columns = self.data.columns
        for axis in ['x', 'y', 'z', 't']:
            name = self.params.get('{}_axis_attribute'.format(axis), [None])
            if isinstance(name, list) and len(name):
                name = name[0]
            else:
                name = None

            attrs[axis] = next((c for c in columns if c == name), None)
            if attrs[axis]:
                axis_type = str(self.data[name].dtype)
                axis_type = ChartVisualization._get_attr_type(axis_type)

                # this way we don't bind x_axis and y_axis types. Y is only
                # going to be number for now
                if axis == 'y':
                    axis_type = 'number'

                result[axis] = {
                    "title": self.params.get("{}_title".format(axis)),
                    "prefix": self.params.get("{}_prefix".format(axis)),
                    "suffix": self.params.get("{}_suffix".format(axis)),
                    "type": axis_type
                }
                axis_format = self.params.get('{}_format'.format(axis), {})

                if axis_type in ['number']:
                    result[axis]['format'] = axis_format.get('key')

                elif axis_type in ['timestamp', 'date', 'time']:
                    result[axis]["inFormat"] = self.default_time_format
                    result[axis]["outFormat"] = self.default_time_format

                    # result[axis]["outFormat"] = axis_format.get('key')
                    # result[axis]["inFormat"] = axis_format.get('key')

        result.update(self._get_title_legend_tooltip())

        series_attr_name = self.params.get('series_attribute', [None])[0]
        if series_attr_name:
            series_attr = next((c for c in columns if c == series_attr_name),
                               None)
        else:
            series_attr = None

        series = {}
        series_key = '@_ \\UNIQUE KEY/ :P_ @'
        if not series_attr:
            series[series_key] = {
                "id": result['title'],
                "name": result['title'],
                "image": None,
                "color": COLORS_PALETTE[0],
                "values": []
            }

        current_color = 0
        rows = self.data.to_numpy()
        if series_attr:
            series_attr_idx = self.data.columns.get_loc(series_attr)
        for row in rows:
            if series_attr:
                series_value = row[series_attr_idx]
                if series_value not in series:
                    color = _get_color_palette(current_color)
                    series[series_value] = {
                        "id": series_value,
                        "name": series_value,
                        "image": None,
                        "color": color,
                        "values": []
                    }
                    current_color += 1
                data = series[series_value]['values']
            else:
                data = series[series_key]['values']

            item = {}
            for axis in ['x', 'y', 'z', 't']:
                col = attrs[axis]
                idx = None
                if col:
                    idx = self.data.columns.get_loc(col)
                item[axis] = ScatterPlotModel._get_value(row, idx)
            data.append(item)

        result['data'] = list(series.values())
        return result

    @staticmethod
    def _get_value(row, attr, default_value=None):
        if attr is not None:
            return ChartVisualization._format(row[attr])
        else:
            return default_value


class HtmlVisualizationModel(VisualizationModel):
    # noinspection PyUnusedLocal
    def __init__(self, data=None, task_id=None, type_id=1, type_name=None,
                 title=None,
                 column_names=None,
                 orientation=None, id_attribute=None,
                 value_attribute=None, params=None):
        type_id = 1
        type_name = 'html'
        if id_attribute is None:
            id_attribute = []
        if value_attribute is None:
            value_attribute = []
        if column_names is None:
            column_names = []
        VisualizationModel.__init__(self, data, task_id, type_id, type_name,
                                    title, column_names, orientation,
                                    id_attribute, value_attribute, params)

    def get_icon(self):
        return "fa-html5"

    def get_data(self):
        return self.data

    def get_schema(self):
        return ''


class TableVisualizationModel(VisualizationModel):
    def __init__(self, data, task_id, type_id, type_name, title,
                 column_names,
                 orientation, id_attribute, value_attribute, params):
        type_id = 35
        type_name = 'table-visualization'
        if not title:
            title = 'Results'
        VisualizationModel.__init__(self, data, task_id, type_id, type_name,
                                    title, column_names, orientation,
                                    id_attribute, value_attribute, params)

    def get_icon(self):
        return 'fa-table'

    def get_data(self):
        """
        Returns data as tabular (list of lists in Python).
        """
        if self.column_names:
            rows = self.data.head(50)[self.column_names].to_numpy().tolist()
        else:
            rows = self.data.head(50).to_numpy().tolist()

        return {"rows": rows,
                "attributes": self.get_column_names().split(',')}

    # def get_schema(self):
    #
    #     if self.column_names:
    #         return self.data[self.column_names].schema.json()
    #     else:
    #         return self.data.schema.json()

    def get_column_names(self):
        if self.column_names:
            return ','.join(self.column_names)
        else:
            return get_csv_schema_sklearn(self.data, only_name=True)


class SummaryStatisticsModel(TableVisualizationModel):
    # noinspection PyUnusedLocal
    def __init__(self, data, task_id, type_id, type_name, title,
                 column_names,
                 orientation, id_attribute, value_attribute, params):
        TableVisualizationModel.__init__(self, data, task_id, type_id,
                                         type_name,
                                         title, column_names, orientation,
                                         id_attribute, value_attribute,
                                         params)

        from pandas.api.types import is_numeric_dtype

        all_attr = list(self.data.columns)
        if len(self.params['attributes']) == 0:
            self.attrs = all_attr
        else:
            self.attrs = [attr for attr in all_attr if
                          attr in self.params['attributes']]

        self.numeric_attrs = [
            t for t in self.attrs if is_numeric_dtype(self.data[t])]

        self.names = collections.OrderedDict([
            ('attribute', _('attribute')), ('count', _('count')),
            ('unique', _('unique')), ('mean', _('mean')),
            ('min', _('min')), ('max', _('max')),
            ('std. dev.', _('std. dev.')), ('sum', _('sum')),
            ('25%', _('25%')), ('50%', _('50%')), ('75%', _('75%')),
            ('mode', _('mode')), ('iqr', _('iqr')),
            ('skewness', _('skewness')), ('kurtosis', _('kurtosis'))
        ])

        complete = self.params[
            SummaryStatisticsOperation.COMPLETE_ANALYSIS_PARAM]
        correlation = self.params[SummaryStatisticsOperation.CORRELATION_PARAM]

        if complete:
            self.names['outliers'] = _('outliers')
            self.names['histogram'] = _('histogram')
            self.names['top'] = _('top')

        if correlation or complete:
            for attr in self.numeric_attrs:
                col = 'correlation to {} (Pearson)'.format(attr)
                self.names[col] = _(col)

    def get_icon(self):
        return 'fa-table'

    # noinspection PyUnresolvedReferences
    def get_data(self):
        """
        Returns statistics about attributes in a data frame
        """

        def find_outliers(df, summary, whisker_width=1.5):
            outliers = dict()
            for column in df.columns:
                iqr = summary.loc[column, 'iqr']
                q3 = summary.loc[column, '75%']
                q1 = summary.loc[column, '25%']

                lower_bound = q1 - (whisker_width * iqr)
                upper_bound = q3 + (whisker_width * iqr)

                query = (df[column] < lower_bound) | (df[column] > upper_bound)
                outliers[column] = [
                    list(df[column].loc[query].sort_values().unique())]

            return pd.DataFrame.from_dict(outliers, orient='index',
                                          columns=['outliers'])

        def gen_top_freq(df, count, n=10):
            top = {}
            mode = {}
            for col in df.columns:
                res_tmp = df[col].value_counts().iloc[:n]
                mode[col] = res_tmp.index[0]
                c = count.loc[col, 'count']
                res = {k: [v, (v / c) * 100] for k, v in
                       res_tmp.to_dict().items()}
                top[col] = [res]

            top = pd.DataFrame.from_dict(top, orient="index", columns=['top'])
            mode = pd.DataFrame.from_dict(mode, orient="index",
                                          columns=['mode'])
            return top, mode

        def gen_histogram(df, cols, bins=10):
            histograms = dict()
            for col in cols:
                hist, edges = np.histogram(df[col].dropna().to_numpy(),
                                           bins=bins, density=False)
                histograms[col] = json.dumps(
                        {"{}-{}".format(edges[i], edges[i + 1]): str(hist[i])
                         for i in range(bins)})
            return pd.DataFrame.from_dict(histograms, orient='index',
                                          columns=['histogram'])

        data = self.data[self.attrs]

        summary = data.describe(include=None, datetime_is_numeric=True) \
            .transpose() \
            .drop(['count'], axis=1)

        sum_metric = data.sum(skipna=True, numeric_only=True).to_frame('sum')
        skewness = data.skew(skipna=True).to_frame('skewness')
        kurtosis = data.kurtosis(skipna=True).to_frame('kurtosis')
        iqr = (summary['75%'] - summary['25%']).to_frame('iqr')
        unique = data.nunique(dropna=True).to_frame('unique')
        count = data.count().to_frame('count')
        top, mode = gen_top_freq(data, count, n=10)

        summary = summary \
            .rename(columns={"std": 'std. dev.'})\
            .merge(count, left_index=True, right_index=True, how='outer') \
            .merge(skewness, left_index=True, right_index=True, how='left') \
            .merge(kurtosis, left_index=True, right_index=True, how='left') \
            .merge(iqr, left_index=True, right_index=True, how='left') \
            .merge(sum_metric, left_index=True, right_index=True, how='left') \
            .merge(unique, left_index=True, right_index=True, how='left') \
            .merge(mode, left_index=True, right_index=True, how='left')

        correlation = self.params[SummaryStatisticsOperation.CORRELATION_PARAM]
        complete = self.params[
            SummaryStatisticsOperation.COMPLETE_ANALYSIS_PARAM]

        if complete:
            outliers = find_outliers(data, summary)
            numeric_cols = summary.index[~summary['mean'].isnull()].tolist()
            histogram = gen_histogram(data, numeric_cols, bins=10)

            summary = summary \
                .merge(outliers,
                       left_index=True, right_index=True, how='left') \
                .merge(histogram,
                       left_index=True, right_index=True, how='left') \
                .merge(top, left_index=True, right_index=True, how='left')

        if correlation or complete:
            corr = data.corr()
            corr.columns = ['correlation to {} (Pearson)'.format(col)
                            for col in corr.columns]
            summary = summary \
                .merge(corr, left_index=True, right_index=True, how='left')

        summary['attribute'] = summary.index
        # reordering and renaming
        summary = summary[self.names.keys()].rename(columns=self.names)
        columns = summary.columns.tolist()
        rows = summary[columns].to_numpy().tolist()

        return {"rows": rows, "attributes": columns}
