# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation


class OptimizatedOperation(Operation):
    """OptimizatedOperation.

    Merge many tasks in only one
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if not 'code_0' in self.parameters and 'code_1' in self.parameters:
            raise ValueError(
                    _("Parameter {} and {} must be informed for task {}")
                    .format('code_0', 'code_1', self.__class__))

        self.order = self.parameters['task']['order']
        self.code_0 = self.parameters.get('code_0', '')
        self.code_1 = self.parameters.get('code_1', '')
        self.has_code = True
        self.has_code_otm = True
        self.number_tasks = self.parameters.get('number_tasks', 0)
        self.fist_id = self.parameters.get('fist_id', 0)

        if 'output data' in self.named_outputs:
            self.output = self.named_outputs['output data']
        elif 'output projected data' in self.named_outputs:
            self.output = self.named_outputs['output projected data']
        else:
            self.output = 'output_data_{}'.format(self.order)

        self.has_import = "from pycompss.api.task import task\n" \
                          "from pycompss.api.parameter import *\n"

    def generate_code(self):
        """Generate code."""
        self.input_data = self.named_inputs.get('input data', '')
        code = ""
        if self.parameters['first_slug'] == 'data-reader':
            self.input_data = 'hdfs_blocks'
        if self.parameters['first_slug'] == 'read-shapefile':
            self.input_data = 'shapefile_data'

        if self.parameters['first_slug'] == 'apply-model':

            code += """
        conf = []
        {code}
        {out} = [[] for _ in range(numFrag)]
        for f in range(numFrag):
            {out}[f] = otm_call_{order}({model}, {input}[f], conf, f)
        """.format(code=self.code_0, out=self.output,
                   order=self.order,
                   model=self.named_inputs['model'],
                   input=self.input_data)
        else:
            code += """
        conf = []
        {code}
        {output} = [[] for _ in range(numFrag)]
        for f in range(numFrag):
            {output}[f] = otm_call_{order}({input}[f], conf, f)
        """.format(code=self.code_0, output=self.output,
                   order=self.order,
                   input=self.input_data)

        return dedent(code)

    def generate_optimization_code(self):
        """Generate code."""
        code = ""
        for idx, c in enumerate(self.code_1):
            c = c.replace('conf_X', 'conf_{}'.format(idx))
            code += """conf_{idx} = settings[{idx}]{code}"""\
                .format(idx=idx, code=c)

        return dedent(code)

