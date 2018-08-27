from textwrap import dedent
from juicer.operation import Operation
from itertools import izip_longest


class ApplyModelOperation(Operation):
    NEW_ATTRIBUTE_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
            [len(self.named_inputs) == 2, self.contains_results()])

        self.new_attribute = parameters.get(self.NEW_ATTRIBUTE_PARAM,
                                            'new_attribute')

        self.feature = parameters['features'][0]
        if not self.has_code and len(self.named_outputs) > 0:
            raise ValueError(
                _('Model is being used, but at least one input is missing'))

    def get_data_out_names(self, sep=','):
        return self.output

    def generate_code(self):
        input_data1 = self.named_inputs['input data']
        output = self.named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        model = self.named_inputs.get(
            'model', 'model_task_{}'.format(self.order))

        code = dedent("""
            {out} = {in1}
            X_train = {in1}['{features}'].values.tolist()
            {out}['{new_attr}'] = {in2}.predict(X_train).tolist()
            """.format(out=output, in1=input_data1, in2=model,
                       new_attr=self.new_attribute, features=self.feature))

        return dedent(code)



class LoadModel(Operation):
    """LoadModel.

    REVIEW: 2017-10-20
    ??? - Juicer ?? / Tahiti ok/ implementation ok

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'name' not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('name', self.__class__))

        self.filename = parameters['name']
        self.output = named_outputs.get('output data',
                                        'output_data_{}'.format(self.order))

        self.has_code = len(named_outputs) > 0
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('output data', self.__class__))

    def generate_code(self):
        """Generate code."""
        code = """
        import pickle
        filename = '{filename}'
        {model} = pickle.load(open(filename, 'rb'))
        """.format(model=self.output, filename=self.filename)
        return dedent(code)


class SaveModel(Operation):
    """SaveModel.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = 'name' in parameters and len(named_inputs) == 1
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('name', self.__class__))
        self.filename = self.parameters['name']
        self.overwrite = parameters.get('write_nome', 'OVERWRITE')
        if self.overwrite == 'OVERWRITE':
            self.overwrite = True
        else:
            self.overwrite = False

    def generate_code(self):
        """Generate code."""
        code = """
        import pickle
        filename = '{filename}'
        pickle.dump({IN}, open(filename, 'wb'))

        """.format(IN=self.named_inputs['input data'],
                   filename=self.filename, overwrite=self.overwrite)
        return dedent(code)
