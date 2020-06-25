# FIXME: https://github.com/hynek/structlog/issues/268
from copy import deepcopy
import structlog


class Test:
    def __init__(self, id, works=True):
        self.id = id
        self.works = works
        self.log = structlog.get_logger()

    def example(self):
        self.log.info('Works')


test = Test(1)
copied_test = deepcopy(test)
