import numpy as np

class Compressor:
    def __init__(self, param1=None, param2=None, param3=None):
        pass

    def zip(self, params: dict, ctx: dict, arr: np.ndarray):
        '''
        Args:
            params: Параметры запуска рассчета, хранящий датасеты 
                (X, Y, Xs, Ys, X_diag_y, X_diag_ys), вычислимую 
                таргет-функцию с её градиентом (f, gradf), константы Липшица (L)
                и mu-сильной выпуклости (mu) градиента, функцию подсчёта 
                значения критерия (criterion), начальную точку (w0), оптимум (act_val),
                а также все прочее, описанное в playbook.ipynb.
            ctx: Контекст итерации со следующей структурой:
                'w': Смотря что передается в конкретном алгоритме. Чаще - точка, 
                в которой находится процесс в данный момент. 
                'k': Номер итерации.
                'ws': История точек 'w'.
                'i': Номер устройства, на котором происходит сжатие. i=-1, если
                значение передано вне контекста какого-либо воркера.
                <other>: См. в файле algorithms.py для изучения конкретного набора 
                переменных в контексте.
            arr: Массив для сжатия.
        Returns:
            np.ndarray: Массив с шейпом как у arr, представляющий из себя сжатый вектор arr.
            np.float64: Отношение количества сжатой информации к исходно хранившейся в arr.
        '''
        pass

class UncorrelatedRandK(Compressor):
    def __init__(self, K=30, param2=None, param3=None):
        self.K = K

    def zip(self, params: dict, ctx: dict, arr: np.ndarray):
        number_of_components = int(self.K / 100 * arr.size) + 1
        chosen = np.random.choice(np.arange(arr.size), number_of_components, replace=False)
        chosen = np.sort(chosen)
        to_be_inserted = arr[chosen]

        vec_new = np.zeros(arr.size)
        vec_new = np.insert(vec_new, chosen, to_be_inserted)
        vec_new = np.delete(vec_new, chosen + np.arange(chosen.size) + 1)

        density = self.K / 100
        return vec_new * 100 / self.K, density
    
class CorrelatedRandK(Compressor):
    def __init__(self, K=30, param2=None, param3=None):
        self.K = K

    def zip(self, params: dict, ctx: dict, arr: np.ndarray):
        # TODO: реализовать
        pass

class TopK(Compressor):
    def __init__(self, K=30, param2=None, param3=None):
        self.K = K

    def zip(self, params: dict, ctx: dict, arr: np.ndarray):
        number_of_components = int(self.K / 100 * arr.size) + 1
        chosen = np.flip(np.argsort(np.abs(arr)))[0:number_of_components]
        chosen = np.sort(chosen)
        to_be_inserted = arr[chosen]

        vec_new = np.zeros(arr.size)
        vec_new = np.insert(vec_new, chosen, to_be_inserted)
        vec_new = np.delete(vec_new, chosen + np.arange(chosen.size) + 1)

        density = self.K / 100
        return vec_new, density

class NoCompressor(Compressor):
    def __init__(self, param1=None, param2=None, param3=None):
        pass

    def zip(self, params: dict, ctx: dict, arr: np.ndarray):
        return arr, 1

class MyCompressor(Compressor):
    def __init__(self, param1=None, param2=None, param3=None):
        pass
    
    def zip(self, params: dict, ctx: dict, arr: np.ndarray):
        pass
