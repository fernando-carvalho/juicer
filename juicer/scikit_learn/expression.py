#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import json
from six import text_type
from gettext import gettext


class Expression:
    def __init__(self, json_code, params):
        self.code = json_code
        self.functions = {}
        self.imports_functions = {}
        self.translate_functions = {}
        self.build_functions_dict()

        self.imports = ""
        self.parsed_expression = "lambda row: " + self.parse(json_code, params)

    def parse(self, tree, params):

        if tree['type'] == 'BinaryExpression':
            # Parenthesis are needed in some pandas/np expressions 
            result = "({} {} {})".format(  
                self.parse(tree['left'], params),
                tree['operator'],
                self.parse(tree['right'], params))

        # Literal parsing
        elif tree['type'] == 'Literal':
            value = tree['value']
            result = "'{}'".format(value) if isinstance(
                value, (str, text_type)) else str(value)

        # Expression parsing
        elif tree['type'] == 'CallExpression':
            if tree['callee']['name'] not in self.functions:
                raise ValueError(_('Function {f}() does not exists.').format(
                    f=tree['callee']['name']))
            result = self.functions[tree['callee']['name']](tree, params)

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            if 'input' in params:
                result = "{}['{}']".format('row', tree[
                    'name'])  # params['input'], tree['name'])
            else:
                result = "functions.row['{}']".format(tree['name'])

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            if tree['operator'] == '!':
                tree['operator'] = '~'
            result = "({} {})".format(tree['operator'],
                                      self.parse(tree['argument'], params))

        elif tree['type'] == 'LogicalExpression':
            operators = {"&&": "&", "||": "|", "!": "~"}
            operator = operators[tree['operator']]
            result = "{} {} {}".format(self.parse(tree['left'], params),
                                           operator,
                                           self.parse(tree['right'], params))

        elif tree['type'] == 'ConditionalExpression':
            spec = {'arguments': [tree['test'], tree['consequent'],
                                  tree['alternate']]}
            result = self.get_when_function(spec, params)

        else:
            raise ValueError("Unknown type: {}".format(tree['type']))

        return result

    def get_numpy_function_call(self, spec, params, alias=None):
        """
        Wrap column name with row() function call, if such call is not present.
        Convert the function to np.function.

        Example: sin(value) will be converted to np.sin(value)

        """
        function = spec['callee']['name']
        if alias is not None and alias != function:
            function = alias

        function = self.translate_functions[
            function] if function in self.translate_functions else function
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])
        # function_name = spec['callee']['name']
        result = " np.{}({})".format(function, arguments)
        return result

    def get_when_function(self, spec, params):
        """
        Map when() function call in Lemonade into np.select() call in Pandas.
        """
        arguments = [self.parse(x, params) for x in spec['arguments']]
        # print >> sys.stderr, group(arguments[:-1], 2)
        code = "np.select([{}], [{}], default={})".format(
            ', '.join(arguments[:-1:2]),
            ', '.join(arguments[1::2]), 
            arguments[-1])

        return code


    def get_function_call(self, spec, params, alias=None):
        """
        Wrap column name with row() function call, if such call is not present.
        Custom functions or like "len()" or "str()"

        Example: len(col) will be converted to len(col)
        """
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])
        function_name = spec['callee']['name']
        if alias is not None and alias != function_name:
            function_name = alias
        result = "{}({})".format(function_name, arguments)
        return result

    # TODO: remove this method
    def get_date_function_call(self, spec, params):
        """
        Wrap column name with row() function call, if such call is not present.

        """
        function = spec['callee']['name']
        function = self.translate_functions[
            function] if function in self.translate_functions else function
        # Evaluates if column name is wrapped in a col() function call
        args = [self.parse(x, params) for x in spec['arguments']]
        # origin = args[0]

        arguments = ', '.join(args)

        result = "{}({})".format(function, arguments)
        return result

    def get_built_function_call(self, spec, params):
        """
        Wrap column name with row() function call, if such call is not present.
        It will generate python functions like value.split()
        """
        function = spec['callee']['name']
        function = self.translate_functions[
            function] if function in self.translate_functions else function
        # Evaluates if column name is wrapped in a row() function call
        args = [self.parse(x, params) for x in spec['arguments']]
        col = args[0]
        arguments = ', '.join(args[1:])

        result = "{}.{}({})".format(col, function, arguments)
        return result

    def get_date_instance_function_call(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        # Evaluates if column name is wrapped in a col() function call
        args = [self.parse(x, params) for x in spec['arguments']]
        origin = args[0]

        arguments = ', '.join(args[1:])

        result = "{}.{}({})".format(origin, function, arguments)
        return result

    def get_date_instance_attribute_call(self, spec, params, alias=None):
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        # Evaluates if column name is wrapped in a col() function call
        args = [self.parse(x, params) for x in spec['arguments']]
        origin = args[0]

        if alias is not None and alias != function:
            function = alias

        result = "{}.{}".format(origin, function)
        return result

    def get_window_function(self, spec, params):
        """
            Window function: group a datetime in bins
        """
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        args = [self.parse(x, params) for x in spec['arguments']]
        bins_size = args[1]
        var_date = args[0]

        result = "group_datetime(row[{date}], {bins_size})" \
            .format(date=var_date, bins_size=bins_size)

        return result

    def get_strip_accents_function(self, spec, params):
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])
        result = " ''.join(c for c in unicodedata.normalize('NFD', {}) if unicodedata.category(c) != 'Mn')".format(
            arguments)

        return result

    def get_strip_punctuation_function(self, spec, params):
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])
        strip_punctuation = ".translate(None, string.punctuation)"
        result = '{}{}'.format(arguments, strip_punctuation)
        return result

    def inject_argument(self, spec, inx, arg):
        spec['arguments'].insert(inx, arg)
        return spec

    def get_to_timestamp_function(self, spec, params):
        mapping = {
            'yyyy': '%Y', 'yy': '%y',
            'MM': '%m', 'MMM': '%b', 'MMMM': '%B',
            'dd': '%d',
            'HH': '%H', 'h': '%I',
            'mm': '%M',
            'ss': '%S',
            'EEE': '%a', 'EEEE': '%A',
            'a': '%p',
            "'": ''
        }
        value = self.parse(spec['arguments'][0], params)
        fmt = spec['arguments'][1]['value']  # no parsing
        parts = re.split(r'(\W)', fmt)
        py_fmt = ''.join([mapping.get(x, x) for x in parts])

        return "datetime.datetime.strptime({}, '{}')".format(value, py_fmt)

    def get_substring_function_call(self, spec, params):
        args = spec['arguments']
        if len(args) == 3:
            return '{}[{}:{}]'.format(*[self.parse(x, params) for x in args])
        elif len(args) == 2:
            return '{}[{}:]'.format(*[self.parse(x, params) for x in args])
        else:
            raise ValueError(
                _('Incorrect number of arguments ({}) for function {}'.format
                  (len(args), 'substring')))

    def get_concat_call(self, spec, params):
        args = spec['arguments'][:]

        sep = ''
        if spec['callee']['name'] == 'concat_ws':
            if args[0]['type'] == 'Identifier':
                raise ValueError(gettext('Separator cannot be an attribute'))
            if len(args) < 3:
                raise ValueError(gettext(
                    'Invalid number of arguments for function concat_ws'))
            sep = args.pop(0)['value']
        elif len(args) < 2:
            raise ValueError(gettext(
                'Invalid number of arguments for function concat_ws'))
        attributes = [
                f'str({self.parse(arg, params)})' 
                if arg['type'] == 'Identifier' 
                else self.parse(arg, params)
                for arg in args]

        to_concat = ', '.join(attributes)
        return f'{repr(sep)}.join([{to_concat}])'

    def get_substring_index_function_call(self, spec, params):
        raise ValueError(_('Unsupported function: {}').format(
            'substring_index'))
        # args = spec['arguments']
        # if len(args) == 3:
        #     return ('{txt}[[i for i in range(0, len({txt})) '
        #             'if {txt}[i:].startswith({delim})][{count}]:]').format(
        #         txt=self.parse(args[0], params),
        #         delim=self.parse(args[1], params),
        #         count=self.parse(args[2], params))
        # else:
        #     raise ValueError(
        #         _('Incorrect number of arguments ({}) for function {}'.format
        #           (len(args), 'substring')))

    def get_sequence_function_call(self, spec, params):
        args = spec['arguments']
        if len(args) == 2:
            return 'list(range({}, {}+1))'.format(*[self.parse(x, params)
                                                    for x in args])
        elif len(args) == 3:
            return 'list(range({}, {}+1, {}))'.format(*[self.parse(x, params)
                                                        for x in args])
        else:
            raise ValueError(
                _('Incorrect number of arguments ({}) for function {}'.format
                  (len(args), 'sequence')))

    def get_next_day_function_call(self, spec, params):
        args = spec['arguments']
        if len(args) == 2:
            weekdays = {
                'monday': 0, 'mo': 0, 'mon': 0,
                'tuesday': 1, 'tu': 1, 'tue': 1,
                'wednesday': 2, 'we': 2, 'wed': 2,
                'thursday': 3, 'th': 3, 'thu': 3,
                'friday': 4, 'fr': 4, 'fri': 4,
                'saturday': 5, 'sa': 5, 'sat': 5,
                'sunday': 6, 'su': 6, 'sun': 6,
            }
            weekday = weekdays.get(args[1].get('value', '').lower(), 0)
            return ('{date} + datetime.timedelta('
                    'days=({wd}-{date}.weekday() + 7) % 7)').format(
                date=self.parse(args[0], params),
                wd=weekday,
            )
        else:
            raise ValueError(
                _('Incorrect number of arguments ({}) for function {}'.format
                  (len(args), 'next_day')))

    def build_functions_dict(self):

        str_builtin_functions = ['split', 'capitalize', 'casefold', 'center',
                                 'count', 'encode', 'endswith', 'expandtabs',
                                 'find', 'index', 'isalnum', 'isalpha', 'isdecimal',
                                 'isdigit', 'isidentifier', 'islower', 'isnumeric',
                                 'isprintable', 'isspace', 'istitle', 'isupper',
                                 'ljust', 'lower', 'lstrip', 'partition', 'replace',
                                 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
                                 'rstrip', 'split', 'splitlines', 'startswith',
                                 'strip', 'swapcase', 'title', 'translate', 'upper',
                                 'zfill']

        str_builtin_functions = {k: self.get_built_function_call
                                 for k in str_builtin_functions}

        numpy_functions = {
            # ----- Mathematical operations -------#
            # See more at:
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html

            'abs': self.get_numpy_function_call,

            # Trigonometric functions:
            'sin': self.get_numpy_function_call,
            'cos': self.get_numpy_function_call,
            'tan': self.get_numpy_function_call,
            'arcsin': self.get_numpy_function_call,
            'arccos': self.get_numpy_function_call,
            'arctan': self.get_numpy_function_call,

            'asin': lambda s, p: self.get_numpy_function_call(s, p, 'arcsin'),
            'acos': lambda s, p: self.get_numpy_function_call(s, p, 'arccos'),
            'atan': lambda s, p: self.get_numpy_function_call(s, p, 'arctan'),

            'hypot': self.get_numpy_function_call,
            'arctan2': self.get_numpy_function_call,
            'deg2rad': self.get_numpy_function_call,
            'rad2deg': self.get_numpy_function_call,

            # Hyperbolic functions:
            'sinh': self.get_numpy_function_call,
            'cosh': self.get_numpy_function_call,
            'tanh': self.get_numpy_function_call,
            'arccosh': self.get_numpy_function_call,
            'arcsinh': self.get_numpy_function_call,
            'arctanh': self.get_numpy_function_call,

            # Rounding:
            'around': self.get_numpy_function_call,
            'rint': self.get_numpy_function_call,
            'fix': self.get_numpy_function_call,
            'floor': self.get_numpy_function_call,
            'ceil': self.get_numpy_function_call,
            'trunc': self.get_numpy_function_call,

            # Exponents and logarithms:
            'exp': self.get_numpy_function_call,
            'expm1': self.get_numpy_function_call,
            'exp2': self.get_numpy_function_call,
            'log': self.get_numpy_function_call,
            'log10': self.get_numpy_function_call,
            'log2': self.get_numpy_function_call,
            'log1p': self.get_numpy_function_call,
            'logaddexp': self.get_numpy_function_call,
            'logaddexp2': self.get_numpy_function_call,

            # Arithmetic operations:
            'add': self.get_numpy_function_call,
            'reciprocal': self.get_numpy_function_call,
            'negative': self.get_numpy_function_call,
            'multiply': self.get_numpy_function_call,
            'divide': self.get_numpy_function_call,
            'power': self.get_numpy_function_call,
            'subtract': self.get_numpy_function_call,
            'true_divide': self.get_numpy_function_call,
            'floor_divide': self.get_numpy_function_call,
            'float_power': self.get_numpy_function_call,
            'fmod': self.get_numpy_function_call,
            'remainder': self.get_numpy_function_call,

            # Miscellaneous
            'clip': self.get_numpy_function_call,
            'sqrt': self.get_numpy_function_call,
            'cbrt': self.get_numpy_function_call,
            'square': self.get_numpy_function_call,
            'fabs': self.get_numpy_function_call,
            'sign': self.get_numpy_function_call,
            'nan_to_num': self.get_numpy_function_call,

            # --------- String operations ---------#
            # See more at:
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html

            # String operations
            'mod': self.get_numpy_function_call,
            'decode': self.get_numpy_function_call,
            'encode': self.get_numpy_function_call,


            # --------- Logic operations ----------#
            # See more at:
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.logic.html

            # Logical operations
            'logical_and': self.get_numpy_function_call,
            'logical_or': self.get_numpy_function_call,
            'logical_not': self.get_numpy_function_call,
            'logical_xor': self.get_numpy_function_call,

            # Comparison
            'array_equiv': self.get_numpy_function_call,
            'equal': self.get_numpy_function_call,
            'not_equal': self.get_numpy_function_call,
            'greater_equal': self.get_numpy_function_call,
            'less_equal': self.get_numpy_function_call,
            'greater': self.get_numpy_function_call,
            'less': self.get_numpy_function_call,

            #
        }

        date_functions = {

            #  -------- Date and time operations
            # See more at: https://docs.python.org/2/library/datetime.html

            'timedelta': self.get_date_function_call,

            'str2time': self.get_date_function_call,

            'date': self.get_date_function_call,
            'today': self.get_date_function_call,
            'now': self.get_date_function_call,
            'utcnow': self.get_date_function_call,
            'fromtimestamp': self.get_date_function_call,
            'utcfromtimestamp': self.get_date_function_call,
            'fromordinal': self.get_date_function_call,
            'combine': self.get_date_function_call,

        }

        date_instance_functions = {
            #  -------- Date and time operations
            # See more at: https://docs.python.org/2/library/datetime.html

            'toordinal': self.get_date_instance_function_call,
            'weekday': self.get_date_instance_function_call,
            'isoweekday': self.get_date_instance_function_call,
            'isoformat': self.get_date_instance_function_call,
            'total_seconds': self.get_date_instance_function_call,
        }

        self.functions.update(str_builtin_functions)
        self.functions.update(numpy_functions)
        self.functions.update(date_functions)
        self.functions.update(date_instance_functions)

        translate_functions = {

            'timedelta': 'datetime.timedelta',

            'str2time': 'parser.parse',

            'date': 'datetime.date',
            'today': 'date.today',
            'now': 'datetime.now',
            'utcnow': 'datetime.utcnow',
            'fromtimestamp': 'date.fromtimestamp',
            'utcfromtimestamp': 'date.utcfromtimestamp',
            'fromordinal': 'date.fromordinal',
            'combine': 'date.combine',
            'length': 'len',

            # Numpy string operations
            'mod': 'char.mod',
            'capitalize': 'char.capitalize',
            'center': 'char.center',
            'decode': 'char.decode',
            'encode': 'char.encode',
            'join': 'char.join',
            'ljust': 'char.ljust',
            'lower': 'char.lower',
            'lstrip': 'char.lstrip',
            'partition': 'char.partition',
            'replace': 'char.replace',
            'rjust': 'char.rjust',
            'rpartition': 'char.rpartition',
            'rsplit': 'char.rsplit',
            'rstrip': 'char.rstrip',
            'splitlines': 'char.splitlines',
            'swapcase': 'char.swapcase',
            'upper': 'char.upper',
            'zfill': 'char.zfill',

            # Numpy string information
            'count': 'char.count',
            'find': 'char.find',
            'isalpha': 'char.isalpha',
            'isdecimal': 'char.isdecimal',
            'isdigit': 'char.isdigit',
            'islower': 'char.islower',
            'isnumeric': 'char.isnumeric',
            'isspace': 'char.isspace',
            'istitle': 'char.istitle',
            'isupper': 'char.isupper'
        }
        self.translate_functions.update(translate_functions)

        # define functions for strings
        self.imports_functions = {
            "str2time": "from dateutil import parser",
            "strip_accents": "import unicodedata",
            "strip_punctuation": "import string",
            "translate": "from string import maketrans",
            "split": "import functools",
            "sha1": "import hashlib",
            "sha2": "import hashlib",
            "md5": "import hashlib",

        }

        # Functions that does not exist in Python, but can be implemented as a
        # lambda function. For now, and due simplicity, we require that every
        # custom function is necessarily defined here.
        others_functions = {
            'group_datetime': self.get_window_function,
            'strip_accents': self.get_strip_accents_function,
            'strip_punctuation': self.get_strip_punctuation_function,
            'str': self.get_function_call,
            'len': self.get_function_call,

            'array_contains': lambda s, p: '{1} in {0}'.format(
                self.parse(s['arguments'][0], p), self.parse(
                    s['arguments'][1], p)),
            'array_distinct': lambda s, p: 'np.unique({0}).ravel()'.format(
                self.parse(s['arguments'][0], p)),

            'ascii': self.get_function_call,
            'atan2': lambda s, p: self.get_numpy_function_call(s, p, 'arctan2'),
            # TODO handle differences: python adds 0b to the result
            'bin': self.get_function_call,
            'concat': self.get_concat_call,
            'concat_ws': self.get_concat_call,
            'capitalize': lambda s, p: self.get_function_call(s, p, 'str.capitalize'),
            'current_date': lambda s, p: 'datetime.date.today()',
            'current_timestamp': lambda s, p: 'datetime.datetime.now()',
            # 'datediff': lambda s, p: '{}[::-1]'.format(self.parse(s['arguments'][0], p))
            'dayofmonth': lambda s, p: self.get_date_instance_attribute_call(s, p, 'day'),
            'dayofweek': self.get_date_instance_attribute_call,
            'dayofyear': self.get_date_instance_attribute_call,
            'degrees': self.get_numpy_function_call,
            'instr': lambda s, p: self.get_function_call(s, p, 'str.find'),
            'hex': self.get_function_call,
            'hour': self.get_date_instance_attribute_call,
            'initcap': lambda s, p: self.get_function_call(s, p, 'str.title'),
            'isnan': lambda s, p: self.get_numpy_function_call(s, p, 'isnan'),
            'least': lambda s, p: self.get_numpy_function_call(s, p, 'min'),
            'length': lambda s, p: self.get_function_call(s, p, 'len'),
            # 'levenshtein':
            'lit': lambda s, p: self.parse(s['arguments'][0], p),
            # 'locate': lambda s, p: 'str.find()'.format(self.parse(s['arguments'][0], p)), #TODO: inverse order of params
            'lower': lambda s, p: self.get_function_call(s, p, 'str.lower'),
            'lpad': lambda s, p: self.get_function_call(s, p, 'str.ljust'),
            'ltrim': lambda s, p: self.get_function_call(s, p, 'str.rstrip'),
            'md5': lambda s, p:
                "hashlib.md5({val}.encode()).hexdigest()".format(
                    val=self.parse(s['arguments'][0], p)
            ),
            'minute': self.get_date_instance_attribute_call,
            'month': self.get_date_instance_attribute_call,
            'next_day': self.get_next_day_function_call,
            'pow': lambda s, p: self.get_numpy_function_call(s, p, 'power'),
            'quarter': self.get_date_instance_attribute_call,
            'radians': self.get_numpy_function_call,
            'rand': lambda s, p: self.get_numpy_function_call(
                s, p,                 'random.rand'),
            'randn': lambda s, p: self.get_numpy_function_call(
                s, p, 'random.randn'),
            'regexp_extract': lambda s, p:
                ("functools.partial(lambda t, inx, expr: expr.findall(t)[inx], "
                 "expr=re.compile({expr}))"
                 "({val}, {inx})").format(
                    val=self.parse(s['arguments'][0], p),
                    expr=(self.parse(s['arguments'][1], p)
                          if s['arguments'][1]['type'] != 'Literal' else
                          "r'{}'".format(s['arguments'][1]['value'])),
                    inx=self.parse(s['arguments'][2], p)),
            'regexp_replace': lambda s, p:
                ("functools.partial(lambda t, rep, expr: expr.sub(rep, t), "
                 "expr=re.compile({expr}))"
                 "({val}, {rep})").format(
                    val=self.parse(s['arguments'][0], p),
                    expr=(self.parse(s['arguments'][1], p)
                          if s['arguments'][1]['type'] != 'Literal' else
                          "r'{}'".format(s['arguments'][1]['value'])),

                    rep=self.parse(s['arguments'][2], p),
            ),
            'repeat': lambda s, p: "{val} * {times}".format(
                val=self.parse(s['arguments'][0], p),
                times=self.parse(s['arguments'][1], p),),
            'replace': lambda s, p: "{val}.replace({search}, {replace})".format(
                val=self.parse(s['arguments'][0], p),
                search=self.parse(s['arguments'][1], p),
                replace=self.parse(s['arguments'][2], p)
                    if len(s['arguments']) > 2 else "''"),
            'reverse': lambda s, p: '{}[::-1]'.format(
                self.parse(s['arguments'][0], p)),
            'round': lambda s, p: self.get_numpy_function_call(s, p, 'around'),
            'rpad': lambda s, p: self.get_function_call(s, p, 'str.rjust'),
            'rtrim': lambda s, p: self.get_function_call(s, p, 'str.rstrip'),
            'second': self.get_date_instance_attribute_call,
            # Sequence implementation does not support date type
            'sequence': self.get_sequence_function_call,
            'sha1': lambda s, p:
                'hashlib.sha1({val}.encode("utf8")).hexdigest()'.format(
                    val=self.parse(s['arguments'][0], p)
            ),
            'sha2': lambda s, p:
                ("functools.partial(lambda v, h: h.update(v) or h.hexdigest(),"
                 "h=hashlib.sha{alg}())({val}.encode('utf8'))").format(
                    val=self.parse(s['arguments'][0], p),
                    alg=self.parse(s['arguments'][1], p)
            ),
            'shiftLeft': lambda s, p: self.get_numpy_function_call(
                s, p, 'right_shift'),
            'shiftRight': lambda s, p: self.get_numpy_function_call(
                s, p, 'right_shift'),
            # Shuffle changes original array. In order to avoid it, partial is
            # used here.
            'shuffle': lambda s, p:
                ('functools.partial(lambda arr: np.random.shuffle(arr) or arr)'
                 '({arr}[:])').format(arr=self.parse(s['arguments'][0], p)
                                      ),
            'signum': lambda s, p: self.get_numpy_function_call(s, p, 'sign'),
            'size': lambda s, p: self.get_function_call(s, p, 'len'),
            # Slice implementation does not support negative index :(
            'slice': lambda s, p: '{val}[{start}-1:{start}+{length}-1]'.format(
                val=self.parse(s['arguments'][0], p),
                start=self.parse(s['arguments'][1], p),
                length=self.parse(s['arguments'][2], p)
            ),
            'sort_array': lambda s, p: 'sorted({val}, reverse={rev})'.format(
                val=self.parse(s['arguments'][0], p),
                rev=self.parse(s['arguments'][1], p)
                    if len(s['arguments']) > 1 else 'False'),
            # Split supports regex
            'split': lambda s, p:
                ("functools.partial(lambda s, r: r.split(s),"
                 "r=re.compile(r{expr}))({txt})").format(
                    txt=self.parse(s['arguments'][0], p),
                    expr=self.parse(s['arguments'][1], p),
            ),
            'substring': self.get_substring_function_call,
            'substring_index': self.get_substring_index_function_call,
            'title': lambda s, p: self.get_function_call(s, p, 'str.title'),
            'to_date': self.get_to_timestamp_function,
            # TODO: Handle some data types in JSON
            'to_json': lambda s, p: self.get_function_call(s, p, 'json.dumps'),
            'to_timestamp': self.get_to_timestamp_function,
            'to_utc_timestamp': lambda s, p: '{0}.tz_localize({1})'.format(
                self.parse(s['arguments'][0], p),
                self.parse(s['arguments'][1], p)),
            'translate': lambda s, p:
                '{}.translate(str.maketrans({}, {}))'.format(
                    self.parse(s['arguments'][0], p),
                    self.parse(s['arguments'][1], p),
                    self.parse(s['arguments'][2], p)),
            'trim': lambda s, p: self.get_function_call(s, p, 'str.strip'),
            'trunc': lambda s, p: '{}.replace(day=1, hour=0, minute=0, second=0, microsecond=0)'.format(
                self.parse(s['arguments'][0], p)) if s['arguments'][1] in ['month', 'mon', 'mm']
            else '{}.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)'.format(
                self.parse(s['arguments'][0], p)),
            'unbase64': lambda s, p: self.get_function_call(s, p, 'base64.b64decode'),
            'unhex': lambda s, p: self.get_function_call(
                self.inject_argument(
                    s, 1, {'type': 'Literal', 'value': 16, 'raw': 16}),
                p, 'int'),
            'unix_timestamp':
                lambda s, p: self.get_function_call(s, p, 'pd.to_datetime'),
            'upper': lambda s, p: self.get_function_call(s, p, 'str.upper'),

            'when': self.get_when_function
        }
        self.functions.update(others_functions)
