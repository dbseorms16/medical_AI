import os
from data import srdata


class brain(srdata.SRData):
    def __init__(self, args, name='brain', train=True, benchmark=False):
        super(brain, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

