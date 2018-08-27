from __future__ import print_function

import logging
import sys
import unicodedata

from jinja2 import nodes
from jinja2.ext import Extension
from juicer.exceptions import JuicerException
from six import reraise as raise_

log = logging.getLogger(__name__)


class HandleExceptionExtension(Extension):
    # a set of names that trigger the extension.
    tags = {'handleinstance'}

    def __init__(self, environment):
        super(HandleExceptionExtension, self).__init__(environment)
        environment.extend()

    def parse(self, parser):
        lineno = parser.stream.next().lineno

        # Retrieves instance
        args = [parser.parse_expression()]
        body = parser.parse_statements(['name:endhandleinstance'],
                                       drop_needle=True)

        result = nodes.CallBlock(self.call_method('_handle', args),
                                 [], [], body).set_lineno(lineno)
        return result

    @staticmethod
    def _handle(instance, caller):
        try:
            return caller()
        except KeyError:
            msg = _('Key error parsing template for instance {instance} {id}. '
                    'Probably there is a problem with port specification') \
                .format(instance=instance.__class__.__name__,
                        id=instance.parameters['task']['id'])
            raise_(JuicerException(msg), None, sys.exc_info()[2])
        except TypeError:
            logging.exception(_('Type error in template'))
            msg = _('Type error parsing template for instance {id} '
                    '{instance}.').format(instance=instance.__class__.__name__,
                                          id=instance.parameters['task']['id'])
            raise_(JuicerException(msg), None, sys.exc_info()[2])


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')
