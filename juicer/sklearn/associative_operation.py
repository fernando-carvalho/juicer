# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class FrequentItemSetOperation(Operation):
    """FP-growth"""

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
            self.rules_output = self.named_outputs['rules output']

        self.column = parameters.get('attribute', '')
        self.confidence = parameters.get('confidence', 0.9)
        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)

        self.has_code = len(self.named_inputs) == 1

        if len(self.column) == 0:
            self.column = "{input}.columns[0]"

    def get_output_names(self, sep=', '):
        return sep.join([self.output,
                         self.rules_output])

    def generate_code(self):
        """Generate code."""
        code = """
        from fim import fpgrowth
        
        transactions = {input}['{col}'].values.tolist()
        min_support = 100 * {min_support}
        
        result = fpgrowth(transactions, target="s",
          supp=min_support, report="s")
        

        {output} = pd.DataFrame(result, columns=['itemsets', 'support'])

        """.format(output=self.output, col=self.column[0],
                   input=self.named_inputs['input data'],
                   min_support=self.parameters['min_support'])

        if self.perform_genRules:
            code += """
             
        from itertools import chain, combinations
        def _filter_rules(rules, count, max_rules, pos):
            total, partial = count
            if total > max_rules:
                gets = 0
                for i in range(pos):
                    gets += partial[i]
                number = max_rules-gets
                if number > partial[pos]:
                    number = partial[pos]
                if number < 0:
                    number = 0
                return rules.head(number)
            else:
                return rules
        
        def _get_rules(freq_items, col_item, col_freq,  min_confidence):
            list_rules = []
            for index, row in freq_items.iterrows():
                item = row[col_item]
                support = row[col_freq]
                if len(item) > 0:
                    subsets = [list(x) for x in _subsets(item)]
        
                    for element in subsets:
                        remain = list(set(item).difference(element))
        
                        if len(remain) > 0:
                            num = float(support)
                            den = _get_support(element, freq_items,
                            col_item, col_freq)
                            confidence = num/den
                            if confidence > min_confidence:
                                r = [element, remain, confidence]
                                list_rules.append(r)
        
            cols = ['Pre-Rule', 'Post-Rule', 'confidence']
            rules = pd.DataFrame(list_rules, columns=cols)
        
            return rules
        
        def _subsets(arr):
            return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])
        
        def _get_support(element, freqset, col_item, col_freq):
        
            for t, s in zip(freqset[col_item].values, freqset[col_freq].values):
                    if element == list(t):
                        return s
            return float("inf")
            
            
        min_conf = {min_conf}
        col_item = 'itemsets'
        col_freq = 'support'
        {rules} = _get_rules({output}, col_item, col_freq, min_conf)    
                
        """.format(min_conf=self.confidence, output=self.output,
                   rules=self.rules_output)
        else:
            code += """
        {rules} = None
        """.format(rules=self.rules_output)
        return dedent(code)


class SequenceMiningOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if 'min_support' not in parameters:
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                        .format('min_support', self.__class__))

        self.column = parameters.get('attribute', '')

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)

        self.has_code = len(self.named_inputs) == 1
        self.min_support = self.parameters.get('min_support', 0.5)
        self.max_length = self.parameters.get('max_length', 10)

    def generate_code(self):
        """Generate code."""
        code = """
        counter = 0
        class Prefixspan:
            def __init__(self, db = []):
                self.db = db
                self.genSdb()
        
            def genSdb(self):
                '''
                Generate mutual converting tables between db and sdb
                '''
                self.db2sdb = dict()
                self.sdb2db = list()
                count = 0
                self.sdb = list()
                for seq in self.db:
                    newseq = list()
                    for item in seq:
                        if self.db2sdb.has_key(item):
                            pass
                        else:
                            self.db2sdb[item] = count
                            self.sdb2db.append(item)
                            count += 1
                        newseq.append( self.db2sdb[item] )
                    self.sdb.append( newseq )
                self.itemCount = count
        
            def run(self, min_sup = 0.2, max_len= 3):
                '''
                mine patterns with min_sup as the min support threshold
                '''
                self.min_sup = min_sup * len(self.db)
                L1Patterns = self.genL1Patterns()
                patterns = self.genPatterns( L1Patterns, max_len )
                self.sdbpatterns = L1Patterns + patterns
        
            def getPatterns(self):
                oriPatterns = list()
                for (pattern, sup, pdb) in self.sdbpatterns:
                    oriPattern = list()
                    for item in pattern:
                        oriPattern.append(self.sdb2db[item])
                    oriPatterns.append( (oriPattern, sup) )
                return oriPatterns
        
            def genL1Patterns(self):
                pattern = []
                sup = len(self.sdb)
                pdb = [(i,0) for i in range(len(self.sdb))]
                L1Prefixes = self.span( (pattern, sup, pdb) )
        
                return L1Prefixes
        
            def genPatterns(self, prefixes, max_len):
                results = []
                for prefix in prefixes:
                    if len(prefix[0]) < max_len:
                        result = self.span(prefix)
                        results += result
        
                if results != []:
                    results += self.genPatterns( results, max_len )
                return results
        
            def span(self, prefix):
                (pattern, sup, pdb) = prefix
        
                itemSups = [0] * self.itemCount
                for (sid, pid) in pdb:
                    itemAppear = [0] * self.itemCount
                    for item in self.sdb[sid][pid:]:
                        itemAppear[item] = 1
                    itemSups = map(lambda x,y: x+y, itemSups, itemAppear)
                prefixes = list()
                for i in range(len(itemSups)):
                    if itemSups[i] >= self.min_sup:
                        newPattern = pattern + [i]
                        newSup = itemSups[i]
                        newPdb = list()
                        for (sid, pid) in pdb:
                            for j in range(pid, len(self.sdb[sid])):
                                item = self.sdb[sid][j]
                                if item == i:
                                    newPdb.append( (sid, j+1) )
                                    break
                        prefixes.append( (newPattern, newSup, newPdb) )
        
                return prefixes
                
        col = '{col}'
        transactions = {input}[col].values.tolist()
        min_support = {min_support}
        max_length = {max_length}

        span = Prefixspan(transactions)
        span.run(min_support, max_length)
        result = span.getPatterns()

        {output} = pd.DataFrame(result, columns=['itemset', 'support'])

        """.format(output=self.output, col=self.column[0],
                   input=self.named_inputs['input data'],
                   min_support=self.min_support,
                   max_length=self.max_length)

        return dedent(code)


class AssociationRulesOperation(Operation):
    """AssociationRulesOperation.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)

        self.has_code = len(self.named_inputs) == 1
        self.col_items = self.parameters.get('col_items', ''),
        self.confidence = self.parameters.get('confidence', 0.5)

    def generate_code(self):
        """Generate code."""

        if not len(self.col_items) > 1:
            self.col_items = "{input}.columns[0]"\
                .format(input=self.named_inputs['input data'])
        else:
            self.col_items = "'{}'".format(self.col_items)

        code = """
                         
        from itertools import chain, combinations
        def _filter_rules(rules, count, max_rules, pos):
            total, partial = count
            if total > max_rules:
                gets = 0
                for i in range(pos):
                    gets += partial[i]
                number = max_rules-gets
                if number > partial[pos]:
                    number = partial[pos]
                if number < 0:
                    number = 0
                return rules.head(number)
            else:
                return rules
        
        def _get_rules(freq_items, col_item, col_freq,  min_confidence):
            list_rules = []
            for index, row in freq_items.iterrows():
                item = row[col_item]
                support = row[col_freq]
                if len(item) > 0:
                    subsets = [list(x) for x in _subsets(item)]
        
                    for element in subsets:
                        remain = list(set(item).difference(element))
        
                        if len(remain) > 0:
                            num = float(support)
                            den = _get_support(element, freq_items,
                            col_item, col_freq)
                            confidence = num/den
                            if confidence > min_confidence:
                                r = [element, remain, confidence]
                                list_rules.append(r)
        
            cols = ['Pre-Rule', 'Post-Rule', 'confidence']
            rules = pd.DataFrame(list_rules, columns=cols)
        
            return rules
        
        def _subsets(arr):
            return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])
        
        def _get_support(element, freqset, col_item, col_freq):
        
            for t, s in zip(freqset[col_item].values, freqset[col_freq].values):
                    if element == list(t):
                        return s
            return float("inf")
            
                
        min_conf = {min_conf}
        col_item = {items}
        col_freq = "{freq}"
        {output} = _get_rules({input}, col_item, col_freq, min_conf)    
        
        """.format(min_conf=self.confidence, output=self.output,
                   input=self.named_inputs['input data'],
                   items=self.col_items,
                   freq='support',
                   total=self.parameters['rules_count'])

        return dedent(code)
