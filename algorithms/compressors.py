import numpy as np

class Compressor:
    def __init__(self, param1=None, param2=None, param3=None):
        pass

    def zip(self, params: dict, ctx: dict, arr: np.ndarray) -> np.ndarray:
        '''
        Args:
            ctx: Контекст итерации со следующей структурой:
                'w': Смотря что передается в конкретном алгоритме. Чаще - точка, 
                в которой находится процесс в данный момент. 
                'k': Номер итерации.
                'ws': История точек 'w'.
                'i': Номер устройства, на котором происходит сжатие.
                <other>: См. в файле algorithms.py для изучения конкретного набора 
                переменных в контексте.
            arr: Массив для сжатия.
        Returns:
            np.ndarray: Массив с шейпом как у arr, состоящий только из 0 и 
                произвольных чисел, где 0 означают сжатые компоненты.
        '''
        pass

class UncorrelatedRandK(Compressor):
    def __init__(self, K=30, param2=None, param3=None):
        self.K = K

    def zip(self, params: dict, ctx: dict, arr: np.ndarray) -> np.ndarray:
        number_of_components = int(self.K / 100 * arr.size) + 1
        chosen = np.random.choice(np.arange(arr.size), number_of_components, replace=False)
        chosen = np.sort(chosen)
        to_be_inserted = np.ones_like(chosen)
        vec_new = np.zeros(arr.size)
        vec_new = np.insert(vec_new, chosen, to_be_inserted)
        vec_new = np.delete(vec_new, chosen + np.arange(chosen.size) + 1)
        return vec_new * 100 / self.K
    
class CorrelatedRandK(Compressor):
    def __init__(self, K=30, param2=None, param3=None):
        self.K = K

    def zip(self, params: dict, ctx: dict, arr: np.ndarray) -> np.ndarray:
        # TODO: реализовать
        pass

class TopK(Compressor):
    def __init__(self, K=30, param2=None, param3=None):
        self.K = K

    def zip(self, params: dict, ctx: dict, arr: np.ndarray) -> np.ndarray:
        K = int(self.K / 100 * arr.size) + 1
        chosen = np.flip(np.argsort(np.abs(arr)))[0:K]
        chosen = np.sort(chosen)
        to_be_inserted = np.ones_like(chosen)

        vec_new = np.zeros(arr.size)
        vec_new = np.insert(vec_new, chosen, to_be_inserted)
        vec_new = np.delete(vec_new, chosen + np.arange(chosen.size) + 1)

        return vec_new

class NoCompressor(Compressor):
    def __init__(self, param1=None, param2=None, param3=None):
        pass

    def zip(self, params: dict, ctx: dict, arr: np.ndarray) -> np.ndarray:
        return np.ones(arr.size)

class MyCompressor(Compressor):
    def __init__(self, param1=None, param2=None, param3=None):
        pass
    
    def zip(self, params: dict, ctx: dict, arr: np.ndarray) -> np.ndarray:
        pass
