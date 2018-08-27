# coding=utf-8


class FakeResponse(object):
    def __init__(self, status, text, args, kwargs):
        self.status_code = status
        self.text = text
        self.args = args
        self.kwargs = kwargs


def fake_req(status, text):
    def g():
        def f(*args, **kwargs):
            return FakeResponse(status, text, args, kwargs)

        return f

    return g
