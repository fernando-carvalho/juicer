import base64
import itertools
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

try:
    from html import escape  # python 3.x
except ImportError:
    from cgi import escape  # python 2.x


class BaseHtmlReport(object):
    pass


class HtmlImageReport(BaseHtmlReport):
    def __init__(self, image):
        self.image = image

    def generate(self):
        return base64.encodestring(self.image)


class ConfusionMatrixImageReport(BaseHtmlReport):
    def __init__(self, cm, classes, normalize=False,
                 title='Confusion matrix', cmap=None,
                 axis=None):
        """
       This function prints and plots the confusion matrix.
       Normalization can be applied by setting `normalize=True`.
       """
        self.cm = cm
        self.classes = classes
        self.normalize = normalize
        self.title = title
        self.cmap = cmap,
        if axis is not None:
            self.axis = axis
        else:
            self.axis = ['True label', 'Predicted label']

        if cmap is None:
            self.cmap = plt.cm.Blues

    def generate(self):

        if self.normalize:
            self.cm = self.cm.astype(
                'float') / self.cm.sum(axis=1)[:, np.newaxis]

        plt.figure()
        plt.imshow(self.cm, interpolation='nearest', cmap=self.cmap)
        plt.title(self.title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        fmt = '.2f' if self.normalize else 'd'
        thresh = self.cm.max() / 2.
        for i, j in itertools.product(range(self.cm.shape[0]),
                                      range(self.cm.shape[1])):
            plt.text(j, i, format(int(self.cm[i, j]), fmt),
                     horizontalalignment="center",
                     color="white" if self.cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel(self.axis[0])
        plt.xlabel(self.axis[1])
        fig_file = BytesIO()
        plt.savefig(fig_file, format='png')

        return base64.b64encode(fig_file.getvalue())


class SimpleTableReport(BaseHtmlReport):
    def __init__(self, table_class, headers, rows, title=None, numbered=False):
        self.table_class = table_class
        self.headers = headers
        self.rows = rows
        self.title = title
        self.numbered = numbered

    def generate(self):
        code = []
        if self.title:
            code.append('<h4>{}</h4>'.format(self.title))
        code.append('<table class="{}"><thead><tr>'.format(self.table_class))
        if self.numbered:
            code.append('<th>#</th>')

        for col in self.headers:
            code.append(u'<th>{}</th>'.format(escape(unicode(col))))
        code.append('</tr></thead>')

        code.append('<tbody>')
        for i, row in enumerate(self.rows):
            code.append('<tr>')
            if self.numbered:
                code.append('<td>{}</td>'.format(i + 1))
            for col in row:
                code.append(u'<td>{}</td>'.format(escape(unicode(col))))
            code.append('</tr>')

        code.append('</tbody>')
        code.append('</table>')
        return ''.join(code)
