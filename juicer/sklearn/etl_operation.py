# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation
from juicer.compss.expression import Expression
from itertools import izip_longest


class AddColumnsOperation(Operation):
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        self.has_code = len(named_inputs) == 2

        self.suffixes = parameters.get('aliases', 'ds0_,ds1_')
        self.suffixes = [s for s in self.suffixes.replace(" ", "").split(',')]

        if not self.has_code:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format('input data 1',  'input data  2', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):

        code = """
        
        {out} = pd.merge({input1}, {input2}, left_index=True, 
            right_index=True, suffixes=('{s1}', '{s1}')) 
        """.format(out=self.output,
                   s1=self.suffixes[0],
                   s2=self.suffixes[1],
                   input1 = self.named_inputs['input data 1'],
                   input2 = self.named_inputs['input data 2'])
        return dedent(code)


class AggregationOperation(Operation):
    """
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

        if not self.has_code:
            raise ValueError(
                _("Parameter '{}', '{}' and '{}' must be informed for task {}")
                .format('input data', 'attributes', 'function', self.__class__))

        for dictionary in functions:
            att = dictionary['attribute']
            f = dictionary['f']
            a = dictionary['alias']
            if (f is not None) and (a is not None):
                if att in self.input_operations:
                        self.input_operations[att].append(f)
                        self.input_aliases[att].append(a)
                else:
                        self.input_operations[att] = [f]
                        self.input_aliases[att] = [a]

        self.input_operations = str(self.input_operations)\
            .replace("u'collect_set'", '_merge_set')\
            .replace("u'collect_list'", '_collect_list')

        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                     self.order))

    def generate_code(self):

        code = """
        def _collect_list(x):
            return x.tolist()
        
        def _merge_set(x):
            return set(x.tolist())
    
    
        columns = {columns}
        target = {aliases}
        operations = {operations}

        {output} = {input}.groupby(columns).agg(operations)
        new_idx = []
        i = 0
        old = None
        for (n1, n2) in {output}.columns.ravel():
            if old != n1:
                old = n1
                i = 0
            new_idx.append(target[n1][i])
            i += 1
    
        {output}.columns = new_idx
        {output} = {output}.reset_index()
        {output}.reset_index(drop=True, inplace=True)
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   columns=self.input_columns,
                   aliases=self.input_aliases,
                   operations=self.input_operations)
        return dedent(code)


class CleanMissingOperation(Operation):
    """
    Clean missing fields from data set
    Parameters:
        - attributes: list of attributes to evaluate
        - cleaning_mode: what to do with missing values. Possible values include
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

        self.output = self.named_outputs.get(
            'output result', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = ""
        if self.mode_CM == "REMOVE_ROW":
            code = """
            #ADD MIN e MAX
            {output} = {input}.dropna(subset={columns}, axis='index')
            """.format(columns=self.attributes_CM,
                       output=self.output,
                       input=self.named_inputs['input data'])
        elif self.mode_CM == "REMOVE_COLUMN":
            code = """
            columns = {columns}
            #ADD MIN e MAX
            for col in columns:
                {input}[col] = {input}[col].dropna(axis='columns')
            {output} = {input}
            """.format(columns=self.attributes_CM,
                       output=self.output,
                       input=self.named_inputs['input data'])

        elif self.mode_CM == "VALUE":
            code = """
            columns = {columns}
            for col in columns:
                {input}[col].fillna(value={value}, inplace=True)
            
            {output} = {input}
            """.format(columns=self.attributes_CM,
                       output=self.output, value=self.value_CM,
                       input=self.named_inputs['input data'])

        elif self.mode_CM == "MEAN":
            code = """
            columns = {columns}
            for col in columns:
                {input}[col].fillna(value={input}[col].mean(), inplace=True)
            {output} = {input}
            """.format(columns=self.attributes_CM,
                       output=self.output,
                       input=self.named_inputs['input data'])
        elif self.mode_CM == "MEDIAN":
            code = """
            columns = {columns}
            for col in columns:
                {input}[col].fillna(value={input}[col].median(), inplace=True)
            {output} = {input}
            """.format(columns=self.attributes_CM,
                       output=self.output,
                       input=self.named_inputs['input data'])

        elif self.mode_CM == "MODE":
            code = """
            columns = {columns}
            for col in columns:
                {input}[col].fillna(value={input}[col].mode(), inplace=True)
            {output} = {input}
            """.format(columns=self.attributes_CM,
                       output=self.output,
                       input=self.named_inputs['input data'])

        return dedent(code)


class DifferenceOperation(Operation):
    """
    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(named_inputs) == 2
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        names = {input1}.columns
        {output} = pd.merge({input1}, {input2},
            indicator=True, how='left', on=None)
        {output} = {output}.loc[{output}['_merge'] == 'left_only', names]
        """.format(output=self.output,
                   input1=self.named_inputs['input data 1'],
                   input2=self.named_inputs['input data 2'])
        return dedent(code)


class DistinctOperation(Operation):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.attributes = parameters.get('attributes', [])
        if len(self.attributes) == 0:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        'attributes', self.__class__))

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        columns = {keys}
        {output} = {input}.drop_duplicates(columns, keep='first')\
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   keys=self.attributes)
        return dedent(code)


class DropOperation(Operation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
            self.cols = ','.join(['"{}"'.format(x)
                                   for x in self.attributes])
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        columns = {columns}
        {output} = {input}.drop(columns=columns)
        """.format(output=self.output,
                   input=self.named_inputs['input data'],
                   columns=self.attributes)
        return dedent(code)


class FilterOperation(Operation):
    """
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """
    FILTER_PARAM = 'filter'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.FILTER_PARAM not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                    .format(self.FILTER_PARAM, self.__class__))

        self.has_code = (len(named_inputs) == 1)

        self.query = ""
        for dictionary in parameters.get(self.FILTER_PARAM):
            self.query += "({} {} {}) and ".format(dictionary['attribute'],
                                                   dictionary['f'],
                                                   dictionary['alias'] )
        self.query = self.query[:-4]
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        query = "{query}"
        {out} = {input}.query(query)
        """.format(out=self.output, input=self.named_inputs['input data'],
                   query=self.query)

        return dedent(code)


class Intersection(Operation):
    """
    Returns a new DataFrame containing rows only in both this
    frame and another frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(named_inputs) == 2

        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input data 1',  'input data 2', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        {input1} = {input1}.dropna(axis=0, how='any')
        {input2} = {input2}.dropna(axis=0, how='any')
        keys = {input1}.columns.tolist()
        {input1} = pd.merge({input1}, {input2}, how='left', on=keys,
                       indicator=True, copy=False)
        {output} = {input1}.loc[{input1}['_merge'] == 'both', keys]
        
        """.format(output=self.output,
                   input1=self.named_inputs['input data 1'],
                   input2=self.named_inputs['input data 2'])
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
        self.keep_right_keys = \
            parameters.get(self.KEEP_RIGHT_KEYS_PARAM, False) in (1, '1', True)
        self.match_case = parameters.get(self.MATCH_CASE_PARAM, False) \
                          in (1, '1', True)
        self.join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')

        self.join_type = self.join_type.replace("_outer", "")

        if not all([self.LEFT_ATTRIBUTES_PARAM in parameters,
                    self.RIGHT_ATTRIBUTES_PARAM in parameters]):
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")\
                    .format(self.LEFT_ATTRIBUTES_PARAM,
                            self.RIGHT_ATTRIBUTES_PARAM,
                            self.__class__))

        self.has_code = len(named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                    .format('input data 1',  'input data 2', self.__class__))

        self.left_attributes = parameters.get(self.LEFT_ATTRIBUTES_PARAM)
        self.right_attributes = parameters.get(self.RIGHT_ATTRIBUTES_PARAM)

        self.suffixes = parameters.get('aliases', '_l,_r')
        self.suffixes = [s for s in self.suffixes.split(',')]
        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                     self.order))

    def generate_code(self):

        code = ""
        if not self.keep_right_keys:
            code = """
            cols_to_remove = [c+'{sf}' for c in 
                {in2}.columns if c in {in1}.columns]
            """.format(in1=self.named_inputs['input data 1'],
                       in2=self.named_inputs['input data 2'],
                       sf=self.suffixes[1])

        if self.match_case:
            code += """
            
            data1_tmp = {in1}[{id1}].applymap(lambda col: str(col).lower())
            col1 = [c+"_lower" for c in data1_tmp.columns]
            data1_tmp.columns = col1
            data1_tmp = pd.concat([{in1}, data1_tmp], axis=1, sort=False)
            
            data2_tmp = {in2}[{id2}].applymap(lambda col: str(col).lower())
            col2 = [c+"_lower" for c in data2_tmp.columns]
            data2_tmp.columns = col2
            data2_tmp = pd.concat([{in2}, data2_tmp], axis=1, sort=False)

            {out} = pd.merge(data1_tmp, data2_tmp, left_on=col1, right_on=col2,
                copy=False, suffixes={suffixes}, how='{type}')
            {out}.drop(col1+col2, axis=1, inplace=True)
             """.format(out=self.output, type=self.join_type,
                        in1=self.named_inputs['input data 1'],
                        in2=self.named_inputs['input data 2'],
                        id1=self.left_attributes,
                        id2=self.right_attributes,
                        suffixes=self.suffixes)
        else:
            code += """
            {out} = pd.merge({in1}, {in2}, how='{type}', 
                suffixes={suffixes},
                left_on={id1}, right_on={id2})
             """.format(out=self.output, type=self.join_type,
                        in1=self.named_inputs['input data 1'],
                        in2=self.named_inputs['input data 2'],
                        id1=self.left_attributes,
                        id2=self.right_attributes,
                        suffixes=self.suffixes)

        if not self.keep_right_keys:
            code += """
            {out}.drop(cols_to_remove, axis=1, inplace=True)
            """.format(out=self.output)

        return dedent(code)


class ReplaceValuesOperation(Operation):
    """
    Replace values in one or more attributes from a dataframe.
    Parameters:
    - The list of columns selected.
    """
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        self.replaces = {}

        if any(['value' not in parameters,
                'replacement' not in parameters]):
            raise ValueError(
                _("Parameter {} and {} must be informed if is using "
                  "replace by value in task {}.")
                    .format('value',  'replacement', self.__class__))

        for att in parameters['attributes']:
            if att not in self.replaces:
                self.replaces[att] = [[], []]
            self.replaces[att][0].append(self.parameters['value'])
            self.replaces[att][1].append(self.parameters['replacement'])

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
            replacement = {replaces}
            for col in replacement:
                old_values = replacement[col][0]
                new_values = replacement[col][1]
                for o, n in zip(old_values, new_values):
                    {input}[col] = {input}[col].replace(o, n)
            
            {output} = {input}
            """.format(output=self.output,
                       input=self.named_inputs['input data'],
                       replaces=self.replaces)
        return dedent(code)


class SampleOrPartition(Operation):
    """
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
        self.fraction = float(self.parameters.get('fraction', 50)) / 100

        if (self.value < 0) and (self.type != 'percent'):
            raise ValueError(
                _("Parameter 'value' must be [x>=0] if is using "
                  "the current type of sampling in task {}.")
                .format(self.__class__))

        self.seed = self.parameters['seed'] \
            if self.parameters['seed'] != "" else 'None'
        self.output = self.named_outputs.get('sampled data',
                                             'output_data_{}'.format(
                                                     self.order))

        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):
        if self.type == 'percent':
            code = """
            {output} = {input}.sample(frac={value}, random_state={seed})
            """.format(output=self.output,
                       input=self.named_inputs['input data'],
                       seed=self.seed, value=self.fraction)
        elif self.type == 'head':
            code = """
            {output} = {input}.head({value})
            """.format(output=self.output,
                       input=self.named_inputs['input data'], value=self.value)
        else:
            code = """
            {output} = {input}.sample(n={value}, random_state={seed})
            """.format(output=self.output,
                       input=self.named_inputs['input data'],
                       seed=self.seed, value=self.value)

        return dedent(code)


class SelectOperation(Operation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.

    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
            self.cols = ','.join(['"{}"'.format(x)
                                   for x in self.attributes])
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get(
            'output projected data', 'projection_data_{}'.format(self.order))

    def generate_code(self):

        code = """
        columns = [{column}]
        {output} = {input}[columns]
        """.format(output=self.output, column=self.cols,
                   input=self.named_inputs['input data'])
        return dedent(code)


class SortOperation(Operation):
    """
    Returns a new DataFrame sorted by the specified column(s).
    Parameters:
    - The list of columns to be sorted.
    - A list indicating whether the sort order is ascending for the columns.
    Condition: the list of columns should have the same size of the list of
               boolean to indicating if it is ascending sorting.
    REVIEW: 2017-10-20
    OK - Juicer / Tahiti / implementation
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        attributes = parameters.get('attributes', [])
        if len(attributes) == 0:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    'attributes', self.__class__))

        self.input_columns = [dict['attribute'] for dict in attributes]
        self.AscDes = [True for _ in range(len(self.input_columns))]
        for i, v in enumerate([dict['f'] for dict in attributes]):
            if v != "asc":
                self.AscDes[i] = False
            else:
                self.AscDes[i] = True

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
            columns = {columns}
            ascending = {asc}
            {output} = {input}.sort_values(by=columns, ascending=ascending)
            """.format(output=self.output,
                       input=self.named_inputs['input data'],
                       columns=self.input_columns,
                       asc=self.AscDes)
        return dedent(code)


class SplitOperation(Operation):
    """
    Randomly splits a Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation
        ('0' means no seed).

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_inputs) == 1

        self.percentage = float(self.parameters.get('weights', 50))/100
        self.seed = self.parameters.get("seed", 0)
        self.seed = 'None' if self.seed == "" else self.seed

        self.out1 = self.named_outputs.get('splitted data 1',
                                           'splitted_1_{}'.format(self.order))
        self.out2 = self.named_outputs.get('splitted data 2',
                                           'splitted_2_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
            return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.out1, self.out2])

    def generate_code(self):
        code = """
        {out1}, {out2} = np.split({input}.sample(frac=1, random_state={seed}), 
        [int({percentage}*len({input}))])
                """.format(out1=self.out1, out2=self.out2,
                           input=self.named_inputs['input data'],
                           seed=self.seed, percentage=self.percentage)
        return dedent(code)


class TransformationOperation(Operation):
    """
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing,
            replace it.
        - Expression: json describing the transformation expression
    """
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        import json

        if any(['alias' not in self.parameters,
                'expression' not in self.parameters]):
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format(self.ALIAS_PARAM,
                        self.EXPRESSION_PARAM, self.__class__))

        self.alias = self.parameters['alias']
        self.json_expression = json.loads(self.parameters['expression'])['tree']
        self.has_code = len(self.named_inputs) == 1
        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                     self.order))

    def generate_code(self):

        # Builds the expression and identify the target column
        params = {'input': 'input_data'}
        expression = Expression(self.json_expression, params)
        #print self.json_expression
        functions = [ self.alias,
                      expression.parsed_expression,
                      expression.imports]

        code = """
        functions = [{expr}]
        {out} = {input}
        for action in functions:
            ncol, function, imp = action
            exec(imp)
            if len({out}) > 0:
                func = eval(function)
                v1s = []
                for _, row in {out}.iterrows():
                    v1s.append(func(row))
                {out}[ncol] = v1s
            else:
                {out}[ncol] = np.nan
        """.format(out=self.output,
                   input=self.named_inputs['input data'],
                   expr=functions)
        return dedent(code)


class UnionOperation(Operation):
    """
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                    .format('input data 1',  'input data 2', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        {0} = pd.concat([{1}, {2}], sort=False, axis=0, ignore_index=True)
        """.format( self.output,
                    self.named_inputs['input data 1'],
                    self.named_inputs['input data 2'])
        return dedent(code)

