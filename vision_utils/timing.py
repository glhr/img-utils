# from https://github.com/justasb/linetimer

import timeit
import time

### usage:
# with CodeTimer() as timer:
#     do_something()
# time = timer.took
###

def get_timestamp():
    return int(time.time())

class CodeTimer:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        # print('Code block' + self.name + ' took: ' + str(self.took) + ' ms')
        return None
