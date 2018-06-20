# coding=utf-8
import base64
from io import BytesIO

from pyspark.ml.linalg import Vectors

import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from jinja2 import Environment, BaseLoader

REPORT_TEMPLATE = '''
<html>
<head>
<link rel="stylesheet"
href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
<style>
    table {
        font-size: 8pt;
    }
</style>
</head>
<body>
    <div class="container">
        <div class="row">
            <table
                class="table table-condensed">
            <tr>
                <th class="col-md-4">Features</th>
                <th class="col-md-2">Predictions</th>
                <th class="col-md-6">Explanation</th>
                <th class="col-md-6">Chart</th>
            </tr>
            {%- for row in data %}
            <tr>
                <td>
                    {%- for f in row['features']%}
                    {{ translate_column(f[0]) }}={{ translate_value(*f) }}
                    {%- endfor %}
                </td>
                <td>
                {%- for f in row['predictions']%}
                    {%- if loop.index0 == 0 %}
                    <strong>
                        {{ f[0] }}={{ "%.4f"|format(f[1]) }}
                    </strong>
                    {%- else %}
                    {{ f[0] }}={{ "%.4f"|format(f[1]) }}
                    {%- endif %}
                    {%- endfor %}
                </td>
                <td>
                <table class="table table-condensed">
                {%- for col in row['results'] %}
                <tr>
                    <td><b>{{ translate_expr(col[0]) }}</b></td>
                    <td>{{"%.4f"|format(col[1])}}</td>
                </tr>
                {%- endfor %}
                </table>
                </td>
                <td>
                    <img src="data:image/png;base64, {{chart(row['results'])}}"/>
                </td>
            </tr>
            {%- endfor %}
            </table>
        </div>
    </div>
</body>
</html>
'''


class LemonadeLime(object):
    """
    Integrate Lemonade with LIME (https://github.com/marcotcr/lime), a novel
    explanation technique that explains the predictions of any classifier in an
    interpretable and faithful manner, by learning an interpretable model
    locally around the prediction.
    """
    __slots__ = ('template', 'training_data', 'spark_model', 'testing_data',
                 '_features_col', '_class_names', '_probability_col', '_schema')

    _xpr = re.compile('([_\w\d]+?)=([\d\.]+?)')

    def __init__(self, training_data, testing_data, spark_model,
                 class_names=None,
                 template=REPORT_TEMPLATE):
        self.template = template
        self.training_data = training_data
        self.testing_data = testing_data
        self.spark_model = spark_model

        # Class names in classification
        if class_names is None:
            self._class_names = ['died', 'survived']
        else:
            self._class_names = class_names

        self._features_col = str(self.spark_model.getOrDefault('featuresCol'))

        # Assuming all algorithms will generate a column with this name.
        # Some algorithms have setProbabilityCol() method, but not all (e.g
        # GBTClassifier).
        self._probability_col = 'probability'
        self._schema = self.training_data.schema

    def _transform_lime_data(self, spark_session):
        """
        Convert a LIME data set (NumPy array) into a Spark data frame in order
        to transform it by using Spark ML model.
        According to LIME author (marcotr):
        All one would have to do would be to redefine the prediction function:
            def new_predict_fn(texts):
                spark_objects = transform_texts(texts)
                predictions = spark_predict(spark_objects)
                return predictions
        """

        def lime_transform(data):
            df = self.spark_model.transform(
                spark_session.createDataFrame(
                    pd.DataFrame([Vectors.dense(row) for row in data]),
                    [self._features_col]))

            return self._spark_to_np_array(df, self._probability_col)

        return lime_transform

    def _get_categorical_column_names(self, feature_names):
        """
        Return all categorical features (defined in the feature vector assembled
        and indexed in Spark.
        """
        categorical = [i for i, v in enumerate(feature_names) if
                       self._schema[str(v)].metadata.get('ml_attr', {}).get(
                           'type') == 'nominal']
        return categorical

    @staticmethod
    def _get_feature_column_names(feature_columns):
        """
        Return all feature columns defined in the assembled vector.
        """
        feature_names = [
            str(x[0]) for x in sorted(
                [(item['name'], item['idx']) for k in feature_columns.values()
                 for item in k], key=lambda column_name: column_name[1])]
        return feature_names

    @staticmethod
    def _spark_to_np_array(df, column):
        """
        Convert a data frame to a numpy array of numpy arrays. This is the data
        format used to call LIME.
        """
        return np.array(
            [np.array(r[column]) for r in df.select(column).collect()])

    def _translate_column_value(self, name, value):
        """
        Convert an indexed column name to its original name.
        """
        field_meta = self._schema[str(name)].metadata
        if 'ml_attr' in field_meta and 'vals' in field_meta['ml_attr']:
            return field_meta['ml_attr']['vals'][int(value)]
        return value

    def _translate_expr(self, expression):
        """
        Translate an expression generated by LIME when feature is categorical
        and indexed in Spark.
        """
        found = self._xpr.findall(expression)
        if found:
            return '{}={}'.format(self._translate_column_name(found[0][0]),
                                  self._translate_column_value(*found[0]))
        else:
            return expression

    def _translate_column_name(self, name):
        """
        Translate a label (value) for a categorical and indexed feature in
        Spark. Instead of showing a number for the feature, the original
        categorical value is shown.
        """
        schema_field = [field for field in self.testing_data.schema if
                        field.name == name]
        if schema_field:
            return schema_field[0].metadata.get(
                'lemon_attr', {'indexed_from': name}).get('indexed_from', name)
        else:
            return name

    def generate_html_report(self, spark_session):

        template = Environment(loader=BaseLoader).from_string(self.template)

        lime_training = self._spark_to_np_array(self.training_data,
                                                self._features_col)

        # Spark ML algorithms expect a feature vector. This will extract the
        # name of columns assembled in the vector.
        feature_columns = self._schema[
            self._features_col].metadata.get('ml_attr', {}).get('attrs')
        if not feature_columns:
            raise ValueError(
                _('Training data do not have a column {} as a feature '
                  'assembled vector.'.format(self._features_col)))

        feature_names = self._get_feature_column_names(feature_columns)

        # Assume that all categorical columns were transformed into an indexed
        # representation (required by Spark), by using a StringIndexer or
        # VectorIndexer (not tested)
        categorical = self._get_categorical_column_names(feature_names)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            lime_training,
            feature_names=feature_names,
            categorical_features=categorical,
            class_names=self._class_names,
            discretize_continuous=True,
            random_state=42)
        data = []

        for row in self.testing_data.collect():
            exp = explainer.explain_instance(
                np.array(row['features']),
                self._transform_lime_data(spark_session),
                num_features=len(row[self._features_col]),
                top_labels=1, num_samples=500)
            label = exp.local_exp.keys()[0]
            data.append({
                'predictions': sorted(zip(self._class_names, exp.predict_proba),
                                      key=lambda x: x[1], reverse=True),
                'results': exp.as_list(label),
                'features': zip(feature_names, row[self._features_col])
            })
        return template.render(
            data=data, translate_value=self._translate_column_value,
            translate_expr=self._translate_expr,
            chart=self._generate_chart_as_base64,
            translate_column=self._translate_column_name).encode('utf8')

    # noinspection PyUnresolvedReferences
    def _generate_chart_as_base64(self, lime_results):
        # import pdb
        # pdb.set_trace()
        # plt.rcdefaults()
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 2)

        labels = [self._translate_expr(r[0]) for r in lime_results]
        values = np.array([r[1] for r in lime_results])

        y_pos = np.arange(len(labels))
        colors = np.array([(1, 0, 0)] * len(values))
        colors[values >= 0] = (0, 0, 1)

        ax.barh(y_pos, values, align='center',
                color=colors, ecolor='black', height=.6)

        # ax.set_yticks(y_pos)
        ax.get_xaxis().set_ticks([])
        ax.axvline(x=0, linewidth=1)

        ax.get_yaxis().set_ticks([])
        # ax.set_yticklabels(people)
        ax.invert_yaxis()  # labels read top-to-bottom
        # ax.set_xlabel('Performance')
        # ax.set_title('How fast do you want to go today?')
        red = '#FF4136'
        blue = '#0074D9'

        font_size = 8
        for i, v in enumerate(values):
            if v > 0:
                offset = 0.005
                h_alignment = 'left'
                color = blue
            else:
                offset = -0.005
                h_alignment = 'right'
                color = red
            ax.text(v + offset, i + 0.1, str(round(v, 4)),
                    color=color,
                    horizontalalignment=h_alignment, wrap=True,
                    fontsize=font_size)
            ax.text(offset, i - .4, labels[i], color='black',
                    fontsize=font_size,
                    horizontalalignment=h_alignment,
                    wrap=True)

        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        ax.grid(True)
        plt.subplots_adjust(left=0.3, right=.70)

        fig_file = BytesIO()
        plt.savefig(fig_file, format='png', dpi=75)
        plt.close('all')
        return base64.b64encode(fig_file.getvalue())
