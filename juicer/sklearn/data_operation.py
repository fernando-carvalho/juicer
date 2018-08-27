
import ast
import pprint
from textwrap import dedent
from itertools import izip_longest

from juicer.operation import Operation


class DataReaderOperation(Operation):
    # ADD MODE

    HEADER_PARAM = 'header'
    SCHEMA = 'infer_schema' # FROM_VALUES, FROM_LIMONEIRO or NO
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

        self.name_file = parameters[self.FILE]
        self.separator = parameters.get(self.SEPARATOR, ',')
        self.header = parameters.get(self.HEADER_PARAM, False) in (1, '1', True)
        self.schema = parameters.get(self.SCHEMA, "FROM_VALUES")
        self.mode = parameters.get(self.MODE_PARAM, 'FAILFAST')
        null_values = parameters.get(self.NULL_VALUES_PARAM, '')
        if null_values == '':
            self.null_values = []
        else:
            self.null_values = list(set(v.strip()
                                        for v in null_values.split(",")))

        self.has_code = True

    def generate_code(self):

        code = "{output} = pd.read_csv('{input}', sep='{sep}', " \
               "encoding='utf-8'".format(output=self.output,
                                         input= self.name_file,
                                         sep=self.separator)

        if not self.header:
            code += ", header=None"

        if len(self.null_values) > 0:
            code += ", na_values= {}".format(self.null_values)
        code += ")\n"

        if not self.header:
            code += \
                "{out}.columns = ['col_'+str(col) for col in {out}.columns]"\
                .format(out=self.output)
        return dedent(code)


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

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        for att in [self.NAME_PARAM, self.FORMAT_PARAM, self.PATH_PARAM]:
            if att not in parameters:
                raise ValueError(
                   _("Parameter '{}' must be informed for task {}")
                   .format(att, self.__class__))

        self.name = parameters.get(self.NAME_PARAM)
        self.format = parameters.get(self.FORMAT_PARAM, self.FORMAT_CSV)
        self.path = parameters.get(self.PATH_PARAM, '.')
        self.mode = parameters.get(self.OVERWRITE_MODE_PARAM, self.MODE_ERROR)

        if self.mode == "overwrite":
            self.mode = "w"
        elif self.mode == "append":
            self.mode = "a"
        else:
            # NEED CHECK
            self.mode = "w"

        self.header = parameters.get(self.HEADER_PARAM, False) \
            in (1, '1', True)

        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                     self.order))

        self.filename = self.name
        # if len(self.path) > 0:
        #     self.filename = '/'+self.name
        # else:
        #     self.filename = '/'+self.path+'/'+self.name

        self.has_code = len(self.named_inputs) == 1

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):

        code_save = ""
        if self.FORMAT_CSV == self.format:
            code_save = """
                {input}.to_csv('{output}', sep=',', mode='{mode}',
                    header={header}, index=False, encoding='utf-8')
                
                {tmp} = None
                """.format(tmp=self.output,
                           output=self.filename,
                           input=self.named_inputs['input data'],
                           mode=self.mode,
                           header=self.header)
        elif self.FORMAT_JSON == self.format:
            code_save = """
            {input}.to_json('{output}', orient='records', encoding='utf-8')
            {tmp} = None
            """.format(tmp=self.output, output=self.filename,
                       input=self.named_inputs['input data'])

        return dedent(code_save)
