# -*- coding: utf-8 -*-
import json
from textwrap import dedent
from juicer.operation import Operation
from juicer.compss.expression import Expression
from itertools import izip_longest


class AddColumnsOperation(Operation):
    """AddColumnsOperation.

    Merge two data frames, column-wise, similar to the command paste in Linux.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        self.has_code = len(named_inputs) == 2

        self.suffixes = parameters.get('aliases', '_l,_r')
        self.suffixes = [s for s in self.suffixes.split(',')]

        if self.has_code:
            self.has_import = \
                "from functions.etl.add_columns import AddColumnsOperation\n"
        else:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format('input data 1',  'input data  2', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def get_optimization_information(self):
        # optimization problemn: iteration over others fragments
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
        suffixes = {suffixes}
        balanced = False  # Currently, all data are considered unbalanced
        {out} = AddColumnsOperation().transform({input1},
        {input2}, balanced, numFrag)
        """.format(out=self.output, suffixes=self.suffixes,
                   input1=self.named_inputs['input data 1'],
                   input2=self.named_inputs['input data 2'])
        return dedent(code)


class AggregationOperation(Operation):
    """AggregationOperation.

    Computes aggregates and returns the result as a DataFrame.
    Parameters:
        - Expression: a single dict mapping from string to string, then the key
        is the column to perform aggregation on, and the value is the aggregate
        function. The available aggregate functions are avg, max, min, sum,
        count.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        self.input_columns = parameters.get('attributes', [])
        functions = parameters.get('function', [])
        self.input_aliases = {}
        self.input_operations = {}

        self.has_code = all([len(named_inputs) == 1,
                            len(functions) > 0,
                            len(self.input_columns) > 0])

        if self.has_code:
            self.has_import = \
                "from functions.etl.aggregation import AggregationOperation\n"
        else:
            raise ValueError(
                _("Parameter '{}', '{}' and '{}' must be informed for task {}")
                .format('input data',  'attributes', 'function',
                        self.__class__))

        for d in functions:
            att = d['attribute']
            f = d['f']
            a = d['alias']
            if (f is not None) and (a is not None):
                if att in self.input_operations:
                        self.input_operations[att].append(f)
                        self.input_aliases[att].append(a)
                else:
                        self.input_operations[att] = [f]
                        self.input_aliases[att] = [a]

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)

    def get_optimization_information(self):
        # optimization problemn: the last task is executed multiple times
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
        settings['columns'] = {columns}
        settings['operation'] = {operations}
        settings['aliases'] = {aliases}
        {output} = AggregationOperation().transform({input}, settings, numFrag)
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   columns=self.input_columns,
                   aliases=self.input_aliases,
                   operations=self.input_operations)
        return dedent(code)


class CleanMissingOperation(Operation):
    """CleanMissingOperation.

    Clean missing fields from data set
    Parameters:
        - attributes: list of attributes to evaluate
        - cleaning_mode: what to do with missing values.
          * "VALUE": replace by parameter "value",
          * "MEDIAN": replace by median value
          * "MODE": replace by mode value
          * "MEAN": replace by mean value
          * "REMOVE_ROW": remove entire row
          * "REMOVE_COLUMN": remove entire column
        - value: optional, used to replace missing values
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        if 'attributes' in parameters:
            self.attributes_CM = parameters['attributes']
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('attributes', self.__class__))

        self.mode_CM = self.parameters.get('cleaning_mode', "REMOVE_ROW")
        self.value_CM = self.parameters.get('value', None)

        self.has_code = all([
                any([self.value_CM is not None,  self.mode_CM != "VALUE"]),
                len(self.named_inputs) == 1])

        if self.has_code:
            self.has_import = \
                "from functions.etl.clean_missing "\
                "import CleanMissingOperation\n"

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def get_optimization_information(self):
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 }

        if self.mode_CM in ['REMOVE_ROW', 'VALUE']:

            flags['one_stage'] = True
        else:
            flags['if_first'] = True

        if self.mode_CM != 'REMOVE_ROW':
            flags['keep_balance'] = True

        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['attributes'] = {attributes}
        settings['cleaning_mode'] = '{mode}'
        """.format(attributes=self.attributes_CM, mode=self.mode_CM)

        if self.mode_CM == "VALUE":
            code += """
        settings['value'] = {value}
        """.format(value=self.value_CM)

        code += """
        conf.append(CleanMissingOperation().preprocessing(settings))
        """
        return code

    def generate_optimization_code(self):
        """Generate code."""
        code = """
        {output} = CleanMissingOperation().transform_serial({input}, conf_X)
        """.format(output=self.output,
                   input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        code = """
            settings = dict()
            settings['attributes'] = {attributes}
            settings['cleaning_mode'] = '{cleaning_mode}'
            """.format(attributes=self.attributes_CM,
                       cleaning_mode=self.mode_CM)
        if self.mode_CM == "VALUE":
            code += """
            settings['value'] = {value}
            """.format(value=self.value_CM)
        code += """
            {output} = CleanMissingOperation().transform({input},
            settings, numFrag)
            """.format(output=self.output,
                       input=self.named_inputs['input data'])

        return dedent(code)


class DifferenceOperation(Operation):
    """DifferenceOperation.

    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(named_inputs) == 2
        if self.has_code:
            self.has_import = \
                "from functions.etl.difference import DifferenceOperation\n"
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def get_optimization_information(self):
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': True,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        pass

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        pass

    def generate_code(self):
        code = """
        {} = DifferenceOperation().transform({}, {}, numFrag)
        """.format(self.output, self.named_inputs['input data 1'],
                   self.named_inputs['input data 2'])
        return dedent(code)


class DistinctOperation(Operation):
    """DistinctOperation.

    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)

    REVIEW: 2017-10-20
    OK - Juicer / Tahiti / implementation
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_inputs) == 1

        if self.has_code:
            self.has_import = \
                "from functions.etl.distinct import DistinctOperation\n"

        self.attributes = parameters.get('attributes', [])
        self.attributes = [] if len(self.attributes) == 0 else self.attributes
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def get_optimization_information(self):
        #! multiple executions of same task
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
        if self.has_code:
            code = """
            columns = {keys}
            {output} = DistinctOperation().transform({input}, columns, numFrag)
            """.format(output=self.output, keys=self.attributes,
                       input=self.named_inputs['input data'])
            return dedent(code)


class DropOperation(Operation):
    """DropOperation.

    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(named_inputs) == 1
        if self.has_code:
            self.has_import = "from functions.etl.drop import DropOperation\n"

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))
        self.output_pre = 'conf_{}'.format(parameters['task']['order'])

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
        columns = {columns}
        conf.append(DropOperation().preprocessing(columns))
        """.format(columns=self.parameters['attributes'])
        return code

    def generate_optimization_code(self):
        """Generate code."""
        code = """
        {output} = DropOperation().transform_serial({input}, conf_X)
        """.format(output=self.output,
                   input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        columns = {columns}
        {output} = DropOperation().transform({input}, columns, numFrag)
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   columns=self.parameters['attributes'])
        return dedent(code)


class FilterOperation(Operation):
    """FilterOperation.

    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """

    FILTER_PARAM = 'filter'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.FILTER_PARAM not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}").
                format(self.FILTER_PARAM, self.__class__))

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))
        self.output_pre = 'conf_{}'.format(parameters['task']['order'])
        self.has_code = (len(named_inputs) == 1)
        if self.has_code:
            self.has_import = \
                "from functions.etl.filter import FilterOperation\n"

        self.query = ""
        for dict in parameters.get(self.FILTER_PARAM):
            self.query += "({} {} '{}') and ".format(dict['attribute'],
                                                     dict['f'],
                                                     dict['alias'])
        self.query = self.query[:-4]


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
        settings['query'] = "{query}"
        conf.append(FilterOperation().preprocessing(settings))
        """.format(query=self.query)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {out} = FilterOperation().transform_serial({input}, conf_X)
        """.format(out=self.output,
                   input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        settings = dict()
        settings['query'] = "{query}"
        {out} = FilterOperation().transform({input}, settings, numFrag)
        """.format(out=self.output,
                   input=self.named_inputs['input data'],
                   query=self.query)
        return dedent(code)


class Intersection(Operation):
    """Intersection.

    Returns a new DataFrame containing rows only in both this
    frame and another frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(named_inputs) == 2

        if self.has_code:
            self.has_import = \
                "from functions.etl.intersect import IntersectionOperation\n"
        else:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input data 1',  'input data 2', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def get_optimization_information(self):
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': True,  # if need to be executed as a first task
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
        code = "{} = IntersectionOperation().transform({},{})"\
            .format(self.output, self.named_inputs['input data 1'],
                    self.named_inputs['input data 2'])
        return dedent(code)


class JoinOperation(Operation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    """
    KEEP_RIGHT_KEYS_PARAM = 'keep_right_keys'
    MATCH_CASE_PARAM = 'match_case'
    JOIN_TYPE_PARAM = 'join_type'
    LEFT_ATTRIBUTES_PARAM = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM = 'right_attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.keep_right_keys = parameters.get(self.KEEP_RIGHT_KEYS_PARAM,
                                              False) in (1, '1', True)
        self.match_case = parameters.get(self.MATCH_CASE_PARAM,
                                         False) in (1, '1', True)
        self.join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')

        if not all([self.LEFT_ATTRIBUTES_PARAM in parameters,
                    self.RIGHT_ATTRIBUTES_PARAM in parameters]):
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format(self.LEFT_ATTRIBUTES_PARAM,
                        self.RIGHT_ATTRIBUTES_PARAM,
                        self.__class__))

        self.has_code = len(named_inputs) == 2
        if self.has_code:
            self.has_import = "from functions.etl.join import JoinOperation\n"
        else:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input data 1',  'input data 2', self.__class__))

        self.left_attributes = parameters.get(self.LEFT_ATTRIBUTES_PARAM)
        self.right_attributes = parameters.get(self.RIGHT_ATTRIBUTES_PARAM)

        self.suffixes = parameters.get('aliases', '_l,_r')
        self.suffixes = [s for s in self.suffixes.split(',')]
        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)

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
        settings = dict()
        settings['option'] = '{type}'
        settings['key1'] = {id1}
        settings['key2'] = {id2}
        settings['case'] = {case}
        settings['suffixes'] = {suffixes}
        settings['keep_keys'] = {keep}
        {out} = JoinOperation().transform({in1}, {in2}, settings, numFrag)
        """.format(out=self.output, type=self.join_type,
                   in1=self.named_inputs['input data 1'],
                   in2=self.named_inputs['input data 2'],
                   id1=self.parameters['left_attributes'],
                   id2=self.parameters['right_attributes'],
                   case=self.match_case, keep=self.keep_right_keys,
                   suffixes=self.suffixes)

        return dedent(code)


class ReplaceValuesOperation(Operation):
    """ReplaceValuesOperation.

    Replace values in one or more attributes from a dataframe.
    Parameters:
    - The list of columns selected.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        self.mode = parameters.get('mode', 'value')
        self.input_regex = self.mode != 'value'
        self.replaces = {}

        i = 0
        if not self.input_regex:

            if any(['old_value' not in parameters,
                    'new_value' not in parameters]):
                raise ValueError(
                    _("Parameter {} and {} must be informed if is using "
                      "replace by value in task {}.")
                    .format('old_value',  'new_value', self.__class__))

            for att in parameters['attributes']:
                if att not in self.replaces:
                    self.replaces[att] = [[], []]
                self.replaces[att][0].append(self.parameters['old_value'])
                self.replaces[att][1].append(self.parameters['new_value'])
                i += 1
        else:
            for att in parameters['attributes']:
                if att not in self.replaces:
                    self.replaces[att] = [[], []]
                self.replaces[att][0].append(self.parameters['regex'])
                self.replaces[att][1].append(self.parameters['new_value'])
                i += 1

        self.has_code = len(named_inputs) == 1
        if self.has_code:
            self.has_import = \
               "from functions.etl.replace_values "\
               "import ReplaceValuesOperation\n"
        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)


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
        settings['replaces'] = {replaces}
        settings['regex'] = {regex}
        conf.append(ReplaceValuesOperation().preprocessing(settings))
        """.format(replaces=self.replaces, regex=self.input_regex)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {output} = ReplaceValuesOperation().transform_serial({input}, conf_X)
        """.format(output=self.output,
                   input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        settings = dict()
        settings['replaces'] = {replaces}
        settings['regex'] = {regex}
        {output} = ReplaceValuesOperation().transform({input},
        settings, numFrag)
        """.format(output=self.output, replaces=self.replaces,
                   input=self.named_inputs['input data'],
                   regex=self.input_regex)

        return dedent(code)


class SampleOrPartition(Operation):
    """SampleOrPartition.

    Returns a sampled subset of this DataFrame.
    Parameters:
    - withReplacement -> can elements be sampled multiple times
                        (replaced when sampled out)
    - fraction -> fraction of the data frame to be sampled.
        without replacement: probability that each element is chosen;
            fraction must be [0, 1]
        with replacement: expected number of times each element is chosen;
            fraction must be >= 0
    - seed -> seed for random operation.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.type = self.parameters.get('type', 'percent')
        self.value = self.parameters.get('value', -1)
        if (self.value < 0) and (self.type != 'percent'):
            raise ValueError(
                _("Parameter 'value' must be [x>=0] if is using "
                  "the current type of sampling in task {}.")
                .format(self.__class__))
        self.seed = self.parameters.get('seed', None)
        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('sampled data', tmp)
        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.has_import = "from functions.etl.sample "\
                              "import SampleOperation\n"

    def get_optimization_information(self):
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': True,  # if need to be executed as a first task
                 'need_keeped_data': True  # the group do not change the size ---> REMOVER
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['type'] = '{type}'
        settings['value'] = {value}
        settings['seed'] = {seed}
        conf.append(SampleOperation().preprocessing({input}, settings, numFrag)
        """.format(output=self.output, type=self.type, seed=self.seed,
                   input=self.named_inputs['input data'],
                   value=self.value)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {output} = SampleOperation().transform_serial(({input}, conf_X, idfrag)
        """.format(output=self.output, input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        settings = dict()
        settings['type'] = '{type}'
        settings['value'] = {value}
        settings['seed'] = {seed}
        {output} = SampleOperation().tranform({input}, settings, numFrag)
        """.format(output=self.output, type=self.type, seed=self.seed,
                   input=self.named_inputs['input data'], value=self.value)
        return dedent(code)


class SelectOperation(Operation):
    """SelectOperation.

    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.
    """

    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
            self.cols = ', '.join(['"{}"'.format(x) for x in self.attributes])
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get(
            'output projected data', 'projection_data_{}'.format(self.order))

        self.output_pre = 'conf_{}'.format(parameters['task']['order'])
        if self.has_code:
            self.has_import = \
                "from functions.etl.select import SelectOperation\n"

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
        columns = [{column}]
        conf.append(SelectOperation().preprocessing(columns))
        """.format(column=self.cols)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {output} = SelectOperation().transform_serial({input}, conf_X)
        """.format(output=self.output,
                   input=self.named_inputs['input data'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        columns = [{column}]
        {output} = SelectOperation().transform({input}, columns, numFrag)
        """.format(output=self.output, column=self.cols,
                   input=self.named_inputs['input data'])
        return dedent(code)


class SortOperation(Operation):
    """SortOperation.

    Returns a new DataFrame sorted by the specified column(s).
    Parameters:
    - The list of columns to be sorted.
    - A list indicating whether the sort order is ascending for the columns.
    Condition: the list of columns should have the same size of the list of
               boolean to indicating if it is ascending sorting.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        attributes = parameters.get('attributes', [])
        if len(attributes) == 0:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('attributes', self.__class__))

        self.input_columns = [dict['attribute'] for dict in attributes]
        self.AscDes = [True for _ in range(len(self.input_columns))]
        for i, v in enumerate([dict['f'] for dict in attributes]):
            if v != "asc":
                self.AscDes[i] = False
            else:
                self.AscDes[i] = True

        self.has_code = len(named_inputs) == 1
        if self.has_code:
            self.has_import = "from functions.etl.sort import SortOperation\n"

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

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
            settings = dict()
            settings['columns'] = {columns}
            settings['ascending'] = {asc}
            {output} = SortOperation().transform({input}, settings, numFrag)
            """.format(output=self.output, columns=self.input_columns,
                       input=self.named_inputs['input data'],
                       asc=self.AscDes)
        return dedent(code)


class SplitOperation(Operation):
    """SplitOperation.

    Randomly splits a Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation
        ('0' means no seed).
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_inputs) == 1
        if self.has_code:
            self.has_import = "from functions.etl.split "\
                              "import SplitOperation\n"

        self.percentage = self.parameters.get('weights', 0.5)
        if self.percentage == "":
            self.percentage = 0.5
        self.percentage = float(self.percentage)/100
        self.seed = self.parameters.get("seed", 0)
        self.seed = self.seed if self.seed != 0 else None

        self.out1 = self.named_outputs.get('splitted data 1',
                                           'splitted_1_{}'.format(self.order))
        self.out2 = self.named_outputs.get('splitted data 2',
                                           'splitted_2_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
            return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.out1, self.out2])

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
        settings = dict()
        settings['percentage'] = {percentage}
        settings['seed'] = {seed}
        {out1}, {out2} = SplitOperation().transform({input}, settings, numFrag)
                """.format(out1=self.out1, out2=self.out2,
                           input=self.named_inputs['input data'],
                           seed=self.seed, percentage=self.percentage)
        return dedent(code)


class TransformationOperation(Operation):
    """TransformationOperation.

    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing,
            replace it.
        - Expression: json describing the transformation expression

    !!! TODO: Need to review the list of functions
    """

    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])

        if self.has_code:
            if 'expression' in parameters:
                self.expressions = parameters['expression']
            else:
                msg = _("Parameter must be informed for task {}.")
                raise ValueError(
                    msg.format(self.EXPRESSION_PARAM, self.__class__))
            self.has_import = \
                "from functions.etl.transform import TransformOperation\n"
            self.output = self.named_outputs.get(
                'output data', 'sampled_data_{}'.format(self.order))

            self.input_data = self.named_inputs['input data']
            params = {'input': self.input_data}

            self.expr_alias = []
            for expr in self.expressions:
                # Builds the expression and identify the target column
                expression = Expression(expr['tree'], params)
                self.expr_alias.append([expr['alias'],
                                        expression.parsed_expression,
                                        expression.imports])

    def get_optimization_information(self):
        flags = {'one_stage': True,  # if has only one stage
                 'keep_balance': True,  # if number of rows doesnt change
                 'bifurcation': False,  # if has two inputs
                 'if_first': False,  # if need to be executed as a first task
                 }
        #!TODO: keep_balance not always
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        code = """
        settings = dict()
        settings['functions'] = {expr}
        conf.append(TransformOperation().preprocessing(settings))
        """.format(expr=self.expr_alias)
        return code

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {output} = TransformOperation().transform_serial({input}, conf_X)
        """.format(output=self.output,  input=self.input_data)
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        settings = dict()
        settings['functions'] = {expr}
        {out} = TransformOperation().transform({input}, settings, numFrag)
        """.format(out=self.output,
                   input=self.input_data,
                   expr=self.expr_alias)
        return dedent(code)


class UnionOperation(Operation):
    """UnionOperation.

    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 2
        if self.has_code:
            self.has_import = \
                "from functions.etl.union import UnionOperation\n"
        else:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input data 1',  'input data 2', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def get_optimization_information(self):
        flags = {'one_stage': False,  # if has only one stage
                 'keep_balance': False,  # if number of rows doesnt change
                 'bifurcation': True,  # if has two inputs
                 'if_first': True,  # if need to be executed as a first task
                 }
        return flags

    def generate_preoptimization_code(self):
        """Generate code for optimization task."""
        pass

    def generate_optimization_code(self):
        """Generate code for optimization task."""
        code = """
        {out} = UnionOperation().transform_serial({input1}, {input2})
        """.format(out=self.output,
                   input1=self.named_inputs['input data 1'],
                   input2=self.named_inputs['input data 2'])
        return dedent(code)

    def generate_code(self):
        """Generate code."""
        code = """
        {0} = UnionOperation().transform({1}, {2}, numFrag)
        """.format(self.output,
                   self.named_inputs['input data 1'],
                   self.named_inputs['input data 2'])
        return dedent(code)
