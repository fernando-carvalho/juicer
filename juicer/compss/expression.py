#!/usr/bin/python
# -*- coding: utf-8 -*-


class Expression:
    def __init__(self, json_code, params):
        self.code = json_code
        self.functions = {}
        self.imports_functions = {}
        self.translate_functions = {}
        self.build_functions_dict()

        self.imports = ""
        self.parsed_expression = "lambda col: " + self.parse(json_code, params)

    def parse(self, tree, params):

        if tree['type'] == 'BinaryExpression':
            result = "{} {} {}".format(
                self.parse(tree['left'], params),
                tree['operator'],
                self.parse(tree['right'], params))

        # Literal parsing
        elif tree['type'] == 'Literal':
            v = tree['value']
            result = "'{}'".format(v) if type(v) in [str, unicode] else str(v)

        # Expression parsing
        elif tree['type'] == 'CallExpression':
            if tree['callee']['name'] not in self.functions:
                raise ValueError(_('Function {f}() does not exists.').format(
                    f=tree['callee']['name']))
            result = self.functions[tree['callee']['name']](tree, params)

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            if 'input' in params:
                result = "{}['{}']".format('col', tree['name'])
            else:
                result = "functions.col['{}']".format(tree['name'])

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            if tree['operator'] == '!':
                tree['operator'] = '~'
                result = "({} {})".format(tree['operator'],
                                          self.parse(tree['argument'], params))

        elif tree['type'] == 'LogicalExpression':
            operators = {"&&": "&", "||": "|", "!": "~"}
            operator = operators[tree['operator']]
            result = "({}) {} ({})".format(self.parse(tree['left'], params),
                                           operator,
                                           self.parse(tree['right'], params))

        elif tree['type'] == 'ConditionalExpression':
            spec = {'arguments': [tree['test'],
                                  tree['consequent'],
                                  tree['alternate']]}
            result = self.get_when_function(spec, params)

        else:
            raise ValueError("Unknown type: {}".format(tree['type']))

        return result

    def get_numpy_function_call(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join([self.parse(x, params)
                               for x in spec['arguments']])
        function_name = spec['callee']['name']
        result = " np.{}({})".format(function_name, arguments)
        return result

    def get_function_call(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join([self.parse(x, params) for x in spec['arguments']])
        action = spec['callee']['name']

        action = self.translate_functions.get(action, action)
        result = " {}({})".format(action, arguments)
        return result

    def get_date_function_call(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        action = spec['callee']['name']
        action = self.translate_functions.get(action, action)
        # Evaluates if column name is wrapped in a col() function call
        args = [self.parse(x, params) for x in spec['arguments']]

        arguments = ', '.join(args)

        result = " {}({})".format(action, arguments)
        return result

    def get_date_instance_function_call(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        action = spec['callee']['name']
        if action in self.imports_functions:
            imp = self.imports_functions[action] + "\n"
            if imp not in self.imports:
                self.imports += imp
        # Evaluates if column name is wrapped in a col() function call
        args = [self.parse(x, params) for x in spec['arguments']]
        origin = args[0]

        arguments = ', '.join(args[1:])

        action = self.translate_functions.get(action, action)
        result = " {}.{}({})".format(origin, action, arguments)
        return result

    def get_window_function(self, spec, params):
        """
            Window function: group a datetime in bins
        """
        action = spec['callee']['name']
        if action in self.imports_functions:
            imp = self.imports_functions[action] + "\n"
            if imp not in self.imports:
                self.imports += imp
        args = [self.parse(x, params) for x in spec['arguments']]
        bins_size = args[1]
        var_date = args[0]

        result = "group_datetime(col[{date}], {bins_size})"\
            .format(date=var_date, bins_size=bins_size)

        return result

    def get_strip_accents_function(self, spec, params):
        action = spec['callee']['name']
        if action in self.imports_functions:
            imp = self.imports_functions[action] + "\n"
            if imp not in self.imports:
                self.imports += imp
        arguments = ', '.join([self.parse(x, params)
                               for x in spec['arguments']])
        result = \
            " ''.join(c for c in unicodedata.normalize('NFD', unicode({}))" \
            " if unicodedata.category(c) != 'Mn')".format(arguments)

        return result

    def get_strip_punctuation_function(self, spec, params):
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join([self.parse(x, params)
                               for x in spec['arguments']])
        strip_punctuation = ".translate(None, string.punctuation)"
        result = '{}{}'.format(arguments, strip_punctuation )
        return result

    def convertToPandas_function(self, spec, params):
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join([self.parse(x, params)
                               for x in spec['arguments']])

        result = '{}'.format(arguments)
        return result

    def build_functions_dict(self):
        numpy_functions = {
            # ----- Mathematical operations -------#
            # See more at:
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html

            #   *   Trigonometric functions:
            'sin':	self.get_numpy_function_call,  # Trigonometric sine, element-wise.
            'cos':	self.get_numpy_function_call, #	Cosine element-wise.
            'tan':	self.get_numpy_function_call, #	Compute tangent element-wise.
            'arcsin':	self.get_numpy_function_call, #	Inverse sine, element-wise.
            'arccos':	self.get_numpy_function_call, #	Trigonometric inverse cosine, element-wise.
            'arctan':	self.get_numpy_function_call, #	Trigonometric inverse tangent, element-wise.
            'hypot':	self.get_numpy_function_call, #	Given the “legs” of a right triangle, return its hypotenuse.
            'arctan2':	self.get_numpy_function_call, #	Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
            'deg2rad':	self.get_numpy_function_call, #	Convert angles from degrees to radians.
            'rad2deg':	self.get_numpy_function_call, #	Convert angles from radians to degrees.

            #   * Hyperbolic functions:
            'sinh':	self.get_numpy_function_call, #	Hyperbolic sine, element-wise.
            'cosh':	self.get_numpy_function_call, #	Hyperbolic cosine, element-wise.
            'tanh':	self.get_numpy_function_call, #	Compute hyperbolic tangent element-wise.
            'arccosh':	self.get_numpy_function_call, #	Inverse hyperbolic cosine, element-wise.
            'arcsinh':	self.get_numpy_function_call, #	Inverse hyperbolic sine element-wise.
            'arctanh':	self.get_numpy_function_call, #	Inverse hyperbolic tangent element-wise.

            #       *  Rounding:
            'around':	self.get_numpy_function_call, #	Evenly round to the given number of decimals.
            'rint':	    self.get_numpy_function_call, #	Round elements of the array to the nearest integer.
            'fix':	    self.get_numpy_function_call, #	Round to nearest integer towards zero.
            'floor':	self.get_numpy_function_call, #	Return the floor of the input, element-wise.
            'ceil':	    self.get_numpy_function_call, #	Return the ceiling of the input, element-wise.
            'trunc':	self.get_numpy_function_call, #	Return the truncated value of the input, element-wise.

            #   *   Exponents and logarithms:
            'exp':	    self.get_numpy_function_call, #	Calculate the exponential of all elements in the input array.
            'expm1':	self.get_numpy_function_call, #	Calculate exp(x) - 1 for all elements in the array.
            'exp2':	    self.get_numpy_function_call, #	Calculate 2**p for all p in the input array.
            'log':	    self.get_numpy_function_call, #	Natural logarithm, element-wise.
            'log10':	self.get_numpy_function_call, #	Return the base 10 logarithm of the input array, element-wise.
            'log2':	    self.get_numpy_function_call, #	Base-2 logarithm of x.
            'log1p':	self.get_numpy_function_call, #	Return the natural logarithm of one plus the input array, element-wise.
            'logaddexp':	self.get_numpy_function_call, #	Logarithm of the sum of exponentiations of the inputs.
            'logaddexp2':	self.get_numpy_function_call, #	Logarithm of the sum of exponentiations of the inputs in base-2.

            #   *   Arithmetic operations:
            'add':	        self.get_numpy_function_call, #	Add arguments element-wise.
            'reciprocal':	self.get_numpy_function_call, #	Return the reciprocal of the argument, element-wise.
            'negative':	    self.get_numpy_function_call, #	Numerical negative, element-wise.
            'multiply':	    self.get_numpy_function_call, #	Multiply arguments element-wise.
            'divide':	    self.get_numpy_function_call, #	Divide arguments element-wise.
            'power':	    self.get_numpy_function_call, #	First array elements raised to powers from second array, element-wise.
            'subtract':	    self.get_numpy_function_call, #	Subtract arguments, element-wise.
            'true_divide':	self.get_numpy_function_call, #	Returns a true division of the inputs, element-wise.
            'floor_divide':	self.get_numpy_function_call, #	Return the largest integer smaller or equal to the division of the inputs.
            'float_power':	self.get_numpy_function_call, #	First array elements raised to powers from second array, element-wise.
            'fmod':	        self.get_numpy_function_call, #	Return the element-wise remainder of division.
            'mod':	        self.get_numpy_function_call, #	Return element-wise remainder of division.
            'remainder':	self.get_numpy_function_call, #	Return element-wise remainder of division.


            #   *   Miscellaneous
            'clip':         self.get_numpy_function_call, # Clip (limit) the values in an array.
            'sqrt':	        self.get_numpy_function_call, #	Return the positive square-root of an array, element-wise.
            'cbrt':	        self.get_numpy_function_call, #	Return the cube-root of an array, element-wise.
            'square':	    self.get_numpy_function_call, #	Return the element-wise square of the input.
            'fabs':	        self.get_numpy_function_call, #	Compute the absolute values element-wise.
            'sign':	        self.get_numpy_function_call, #	Returns an element-wise indication of the sign of a number.
            'nan_to_num':   self.get_numpy_function_call, #	Replace nan with zero and inf with finite numbers.


            # --------- String operations ---------#
            # See more at:
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html

            #   *   String operations
            'capitalize':   self.get_numpy_function_call, #	Return a copy of a with only the first character of each element capitalized.
            'center':   self.get_numpy_function_call, #	Return a copy of a with its elements centered in a string of length width.
            'decode':   self.get_numpy_function_call, #	Calls str.decode element-wise.
            'encode':   self.get_numpy_function_call, #	Calls str.encode element-wise.
            'join':     self.get_numpy_function_call, #	Return a string which is the concatenation of the strings in the sequence seq.
            'ljust':    self.get_numpy_function_call, #	Return an array with the elements of a left-justified in a string of length width.
            'lower':    self.get_numpy_function_call, #	Return an array with the elements converted to lowercase.
            'lstrip':   self.get_numpy_function_call, #	For each element in a, return a copy with the leading characters removed.
            'partition':    self.get_numpy_function_call, #	Partition each element in a around sep.
            'replace':      self.get_numpy_function_call, #	For each element in a, return a copy of the string with all occurrences of substring old replaced by new.
            'rjust':        self.get_numpy_function_call, #	Return an array with the elements of a right-justified in a string of length width.
            'rpartition':   self.get_numpy_function_call, #	Partition (split) each element around the right-most separator.
            'rsplit':       self.get_numpy_function_call, #	For each element in a, return a list of the words in the string, using sep as the delimiter string.
            'rstrip':       self.get_numpy_function_call, #	For each element in a, return a copy with the trailing characters removed.
            'split':        self.get_numpy_function_call, #	For each element in a, return a list of the words in the string, using sep as the delimiter string.
            'splitlines':   self.get_numpy_function_call, #	For each element in a, return a list of the lines in the element, breaking at line boundaries.
            'strip':        self.get_numpy_function_call, #	For each element in a, return a copy with the leading and trailing characters removed.
            'swapcase':     self.get_numpy_function_call, #	Return element-wise a copy of the string with uppercase characters converted to lowercase and vice versa.
            'upper':        self.get_numpy_function_call, #	Return an array with the elements converted to uppercase.
            'zfill':        self.get_numpy_function_call, #	Return the numeric string left-filled with zeros

            #   *   String information
            'count':        self.get_numpy_function_call, #	Returns an array with the number of non-overlapping occurrences of substring sub in the range [start, end].
            'find':         self.get_numpy_function_call, #	For each element, return the lowest index in the string where substring sub is found.
            'isalpha':      self.get_numpy_function_call, #	Returns true for each element if all characters in the string are alphabetic and there is at least one character, false otherwise.
            'isdecimal':    self.get_numpy_function_call, #	For each element, return True if there are only decimal characters in the element.
            'isdigit':      self.get_numpy_function_call, #	Returns true for each element if all characters in the string are digits and there is at least one character, false otherwise.
            'islower':      self.get_numpy_function_call, #	Returns true for each element if all cased characters in the string are lowercase and there is at least one cased character, false otherwise.
            'isnumeric':    self.get_numpy_function_call, #	For each element, return True if there are only numeric characters in the element.
            'isspace':      self.get_numpy_function_call, #	Returns true for each element if there are only whitespace characters in the string and there is at least one character, false otherwise.
            'istitle':      self.get_numpy_function_call, #	Returns true for each element if the element is a titlecased string and there is at least one character, false otherwise.
            'isupper':      self.get_numpy_function_call, #	Returns true for each element if all cased characters in the string are uppercase and there is at least one character, false otherwise.


            # --------- Logic operations ----------#
            # See more at:
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.logic.html

            #   *   Logical operations
            'logical_and':  self.get_numpy_function_call,  # Compute the truth value of x1 AND x2 element-wise.
            'logical_or':   self.get_numpy_function_call,  # Compute the truth value of x1 OR x2 element-wise.
            'logical_not':  self.get_numpy_function_call,  # Compute the truth value of NOT x element-wise.
            'logical_xor':  self.get_numpy_function_call,  # Compute the truth value of x1 XOR x2, element-wise.


            #   *   Comparison
            'array_equiv':      self.get_numpy_function_call,  # Returns True if input arrays are shape consistent and all elements equal.
            'equal':            self.get_numpy_function_call,  # Return (x1 == x2) element-wise.
            'not_equal':        self.get_numpy_function_call,  # Return (x1 != x2) element-wise.
            'greater_equal':    self.get_numpy_function_call,  # Return (x1 >= x2) element-wise.
            'less_equal':       self.get_numpy_function_call,  # Return (x1 <= x2) element-wise.
            'greater':          self.get_numpy_function_call,  # Return (x1 > x2) element-wise.
            'less':             self.get_numpy_function_call,  # Return (x1 < x2) element-wise.

        }

        date_functions = {
            #  -------- Date and time operations
            # See more at:
            # https://docs.python.org/2/library/datetime.html

            'timedelta':        self.get_date_function_call,

            'str2time':         self.get_date_function_call,

            # For example, date(2002, 12, 4)
            'date': self.get_date_function_call,

            # Return the current local date.
            'today':            self.get_date_function_call,

            # Return the current local date and time.
            'now':              self.get_date_function_call,

            # Return the current UTC date and time, with tzinfo None.
            'utcnow':           self.get_date_function_call,
            'fromtimestamp':    self.get_date_function_call,  # Return the local date and time corresponding to the POSIX timestamp, such as is returned by time.time()
            'utcfromtimestamp': self.get_date_function_call,  # Return the UTC datetime corresponding to the POSIX timestamp, with tzinfo None.
            'fromordinal':      self.get_date_function_call,  # Return the datetime corresponding to the proleptic Gregorian ordinal, where January 1 of year 1 has ordinal 1.
            'combine':          self.get_date_function_call,  # Return a new datetime object whose date components are equal to the given date object’s, and whose time components and tzinfo attributes are equal to the given time object’s.


        }

        date_instance_functions = {
            #  -------- Date and time operations
            # See more at:
            # https://docs.python.org/2/library/datetime.html

            # Return the proleptic Gregorian ordinal of the date,
            # where January 1 of year 1 has ordinal 1.
            # For any date object d, date.fromordinal(d.toordinal()) == d.
            'toordinal': self.get_date_instance_function_call,

            # Return the day of the week as an integer, where Monday is 0
            # and Sunday is 6. For example, date(2002, 12, 4). weekday() == 2,
            # a Wednesday. See also isoweekday().
            'weekday': self.get_date_instance_function_call,

            # Return the day of the week as an integer, where Monday is 1
            # and Sunday is 7.
            # For example, date(2002, 12, 4).isoweekday() == 3, a Wednesday.
            'isoweekday': self.get_date_instance_function_call,

            # For a date d, str(d) is equivalent to d.isoformat()
            'isoformat': self.get_date_instance_function_call,

            # Return a date with the same value, except for those
            # parameters given new values by whichever keyword
            # arguments are specified.
            'replace': self.get_date_instance_function_call,

            # Return the total number of seconds contained in the duration.
            'total_seconds': self.get_date_instance_function_call,

            # Convert datetime to date
            'datetime2date': self.get_date_instance_function_call,
        }

        self.functions.update(numpy_functions)
        self.functions.update(date_functions)
        self.functions.update(date_instance_functions)

        translate_date_functions = {

            'datetime2date': 'date',
            'timedelta': 'datetime.timedelta',
            'str2time': 'parser.parse',
            # 'date': 'datetime.date',
            'today': 'date.today',
            # 'now': 'datetime.now',
            'utcnow': 'datetime.utcnow',
            'fromtimestamp': 'date.fromtimestamp',
            'utcfromtimestamp': 'date.utcfromtimestamp',
            'fromordinal': 'date.fromordinal',
            'combine': 'date.combine',


        }
        self.translate_functions.update(translate_date_functions)

        self.imports_functions = {
            "str2time": "from dateutil import parser",
            "strip_accents": "import unicodedata",
            "strip_punctuation": "import string",

        }

        # Functions that does not exist in Python, but can be implemented as a
        # lambda function. For now, and due simplicity, we require that every
        # custom function is necessarily defined here.
        custom_functions = {
            'group_datetime': self.get_window_function,
            'strip_accents': self.get_strip_accents_function,
            'strip_punctuation': self.get_strip_punctuation_function,
            'str': self.get_function_call
        }
        self.functions.update(custom_functions)
