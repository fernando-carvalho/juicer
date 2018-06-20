# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class AprioriOperation(Operation):
    """AprioriOperation.

    REVIEW: 2017-10-20
    OK - Juicer / Tahiti / implementation
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'min_support' not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format('min_support', self.__class__))

        self.perform_genRules = 'rules output' in self.named_outputs

        if not self.perform_genRules:
            self.rules_output = 'rules_{}'.format(self.order)
        else:
            self.rules_output = parameters['rules output']

        self.column = parameters.get('attribute', '')
        self.confidence = parameters.get('confidence', 0.9)
        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.has_import = "from functions.ml.associative.apriori.apriori "\
                              "import Apriori\n"

    def get_output_names(self, sep=', '):
        return sep.join([self.output,
                         self.rules_output])

    def generate_code(self):
        """Generate code."""
        if len(self.column) > 0:
            code = """
                settings = dict()
                settings['col'] = '{col}'
                settings['minSupport'] = {supp}

                apriori = Apriori()
                {output} = apriori.runApriori({input}, settings, numFrag)

                """.format(output=self.output, col=self.column[0],
                           input=self.named_inputs['input data'],
                           supp=self.parameters['min_support'])
        else:
            code = """
                settings = dict()
                settings['minSupport'] = {supp}

                apriori = Apriori()
                {output} = apriori.runApriori({input}, settings, numFrag)
                """.format(output=self.output,
                           input=self.named_inputs['input data'],
                           supp=self.parameters['min_support'])

        if self.perform_genRules:
            code += """
                settings['confidence']  = {conf}
                {rules} = apriori.generateRules({output},settings)
                """.format(conf=self.confidence, output=self.output,
                           rules=self.rules_output)
        else:
            code += """
                {rules} = None
                """.format(rules=self.rules_output)
        return dedent(code)


class AssociationRulesOperation(Operation):
    """AssociationRulesOperation.

    REVIEW: 2017-10-20

    Note: Attributos insistem em ficar la
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        exep = ['confidence', 'rules_count', 'col_items', 'col_support']
        if any([att not in parameters for att in exep]):
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                .format(exep, self.__class__))

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.has_import = \
                "from functions.ml.associative.AssociationRules " \
                "import AssociationRulesOperation\n"

    def generate_code(self):
        """Generate code."""
        code = """
            settings = dict()
            settings['col_item'] = "{items}"
            settings['col_freq'] = "{freq}"
            settings['rules_count'] = {total}
            settings['confidence']  = {conf}
            {output} = AssociationRulesOperation({input}, settings)
            """.format(output=self.output,
                       input=self.named_inputs['input data'],
                       items=self.parameters['col_items'][0],
                       freq=self.parameters['col_support'][0],
                       conf=self.parameters['confidence'],
                       total=self.parameters['rules_count'])

        return dedent(code)
