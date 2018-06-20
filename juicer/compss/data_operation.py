# -*- coding: utf-8 -*-

import ast
import pprint
from textwrap import dedent
from itertools import izip_longest

from juicer.operation import Operation


class DataReaderOperation(Operation):
    """DataReaderOperation.

    Reads a CSV file using HDFS.
    """

    SCHEMA = 'infer_schema'  # FROM_VALUES, FROM_LIMONEIRO or NO
    NULL_VALUES_PARAM = 'null_values'
    SEPARATOR = 'separator'
    MODE_PARAM = 'mode'
    FILE = 'data_source'

    def __init__(self, parameters, named_inputs,  named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.FILE not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.FILE, self.__class__))

        self.name_file = "/"+parameters[self.FILE]
        self.separator = parameters.get(self.SEPARATOR, ',')
        self.header = parameters.get('header', False) in (1, '1', True)
        self.schema = parameters.get(self.SCHEMA, "FROM_VALUES")
        self.mode = parameters.get(self.MODE_PARAM, 'FAILFAST')
        null_values = parameters.get(self.NULL_VALUES_PARAM, '')
        self.format = parameters.get('format', 'csv')

        if null_values == '':
            self.null_values = []
        else:
            self.null_values = \
                list(set(v.strip() for v in null_values.split(",")))

        self.has_code = len(named_outputs) > 0
        self.has_import = "from functions.data.read_data import "\
                          "ReadOperationHDFS\n"

    def get_optimization_information(self):
        flags = {'one_stage': True,  # if has only one stage
                 'keep_balance': True,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        if len(self.null_values) > 0:
            null = "settings['na_values'] = {}".format(self.null_values)
        else:
            null = ""

        code = """
        filename = '{input}'
        settings = dict()
        settings['port'] = 9000
        settings['host'] = 'localhost'
        settings['header'] = {header}
        settings['separator'] = '{separator}'
        settings['infer'] = '{schema}'
        settings['mode'] = '{mode}'
        settings['format'] = '{format}'
        {null}
        
        hdfs_blocks, settings = ReadOperationHDFS().preprocessing(filename, 
        settings, numFrag)
        conf.append(settings)
        """.format(input=self.name_file, separator=self.separator,
                   header=self.header, mode=self.mode, format=self.format,
                   schema=self.schema, null=null, n=self.order)
        return code

    def generate_optimization_code(self):
        """Generate code."""
        code = """
        {output} = ReadOperationHDFS().transform_serial(input_data, conf_X)
        """.format(output=self.output)
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        if len(self.null_values) > 0:
            null = "settings['na_values'] = {}".format(self.null_values)
        else:
            null = ""

        code = """
        filename = '{input}'
        settings = dict()
        settings['port'] = 9000
        settings['host'] = 'localhost'
        settings['header'] = {header}
        settings['separator'] = '{separator}'
        settings['infer'] = '{schema}'
        settings['mode'] = '{mode}'
        settings['format'] = '{format}'
        {null}

        {output} = ReadOperationHDFS().transform(filename, settings, numFrag)
        """.format(output=self.output, input=self.name_file,
                   separator=self.separator, header=self.header,
                   mode=self.mode, schema=self.schema, format=self.format,
                   null=null)
        return dedent(code)


class SaveOperation(Operation):
    """SaveHDFSOperation.

    Saves the content of the DataFrame at the specified path (HDFS or FS)
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

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        for att in [self.NAME_PARAM, self.FORMAT_PARAM, self.PATH_PARAM]:
            if att not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(att, self.__class__))

        self.name = parameters.get(self.NAME_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM, self.FORMAT_CSV)
        self.path = parameters.get(self.PATH_PARAM, '.')
        self.mode = parameters.get(self.OVERWRITE_MODE_PARAM,
                                   self.MODE_ERROR)
        self.storage = parameters.get(self.STORAGE_ID_PARAM, 'hdfs')
        self.header = parameters.get('header', False) in (1, '1', True)
        self.has_code = len(named_inputs) == 1
        if self.has_code:
            self.has_import = \
                "from functions.data.save_data import SaveOperation\n"

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)

        if len(self.path) > 0:
            self.filename = '/'+self.path+'/'+self.name
        else:
            self.filename = '/'+self.name

        self.has_code = len(self.named_inputs) == 1

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return self.output

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
        settings['filename'] = '{output}'
        settings['mode'] = '{mode}'
        settings['header'] = {header}
        settings['format'] = '{format}'
        settings['storage'] = '{fs}'
        conf.append(SaveOperation().preprocessing(settings, numFrag))
        """.format(output=self.filename,  mode=self.mode,
                   header=self.header, format=self.format,
                   fs=self.storage)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {tmp} = SaveOperation().transform_serial({input}, conf_X, idfrag)
        """.format(tmp=self.output, input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        # Need to generate an output, even though it is not used.
        code_save = ''
        # if self.format == self.FORMAT_CSV:
        code_save = """
            settings = dict()
            settings['filename'] = '{output}'
            settings['mode'] = '{mode}'
            settings['header'] = {header}
            settings['format'] = '{format}'
            settings['storage'] = '{fs}'
            {tmp} = SaveOperation().transform({input}, settings, numFrag)
            """.format(tmp=self.output, output=self.filename,
                       input=self.named_inputs['input data'],
                       mode=self.mode, header=self.header,
                       format=self.format, fs=self.storage)
        # elif self.format == self.FORMAT_JSON:
        #     code_save = """
        #         settings = dict()
        #         settings['filename'] = '{output}'
        #         settings['mode'] = '{mode}'
        #         settings['header'] = {header}
        #         settings['format'] = '{format}'
        #         settings['storage'] = '{fs}'
        #         {tmp} = SaveOperation().transform({input}, settings, numFrag)
        #         """.format(tmp=self.output_tmp,
        #                    output=self.filename,
        #                    input=self.named_inputs['input data'],
        #                    mode=self.mode, header=self.header,
        #                    format=self.format, fs=self.storage)

        return dedent(code_save)


class WorkloadBalancerOperation(Operation):

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if (len(self.named_inputs) == 1) and (len(self.named_outputs) > 0):
            self.has_code = True
            self.has_import = "from functions.data.balancer " \
                              "import WorkloadBalancerOperation\n"
        else:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input', 'output', self.__class__))

    def get_optimization_information(self):
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        pass

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        pass

    def generate_code(self):
        """Generate code."""
        code = """
        forced = True
        {out} = WorkloadBalancerOperation().transform({input}, forced, numFrag)
        """.format(out=self.named_outputs['output data'],
                   input=self.named_inputs['input data'])

        return dedent(code)


class ChangeAttributesOperation(Operation):

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        if 'attributes' not in self.parameters:
            self.has_code = False
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('attributes', self.__class__))

        self.output = named_outputs.get('output data',
                                        'output_data_{}'.format(self.order))

        self.attributes = parameters['attributes']
        self.name = [s.strip() for s in
                     self.parameters.get('new_name', '').split(',')]
        # Adjust alias in order to have the same number of aliases as
        # attributes by filling missing alias with the attribute name
        # sufixed by _indexed.
        if len(self.name) > 0:
            size = len(self.attributes)
            self.name = [x[1] or '{}_new'.format(x[0]) for x in
                         izip_longest(self.attributes, self.name[:size])]

        self.new_type = parameters.get('new_data_type', 'keep')
        self.has_code = len(named_inputs) == 1

        if self.has_code:
            self.has_import = "from functions.data.attributes_changer " \
                              "import AttributesChangerOperation\n"

    def get_optimization_information(self):
        flags = {'one_stage': True,  # if has only one stage
                 'keep_balance': True,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['new_data_type'] = '{newtype}'
        settings['attributes'] = {att}
        settings['new_name'] = {newname}
        conf.append(AttributesChangerOperation().preprocessing(settings))
        """.format(att=self.attributes, newname=self.name,
                   newtype=self.new_type)
        return code

    def generate_optimization_code(self):
        """Generate code."""
        code = """
        {output} = AttributesChangerOperation().transform_serial({input},
        conf_X)
        """.format(output=self.output,
                   input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        settings = dict()
        settings['new_data_type'] = '{newtype}'
        settings['attributes'] = {att}
        settings['new_name'] = {newname}
        {out} = AttributesChangerOperation().transform({inputData},
        settings, numFrag)
        """.format(out=self.output, att=self.attributes,
                   inputData=self.named_inputs['input data'],
                   newname=self.name, newtype=self.new_type)
        return dedent(code)
