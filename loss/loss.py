from abc import ABCMeta, abstractmethod

class Loss(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        raise NotImplementedError