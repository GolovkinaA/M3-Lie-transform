import numpy as np
import math
import matplotlib.pyplot as plt
import sympy
from scipy.integrate import odeint


def right_hand_side_vdp(x, t):
    """
    функция расчёта правых частей СДУ осциллятора Ван-дер-Поля
    :param x: вектор 2 фазовых переменных
    :param t: скаляр, время - не используется в автономной системе
    :return: список (вектор) 2 значений правых частей СДУ
    """
    f = [0, 0]
    f[0] = x[1]
    f[1] = x[1] - x[0] - (x[0]*x[0]*x[1])
    return f

def right_hand_side_deflector(x, t):
    """
    функция расчёта правых частей СДУ дефлектора
    :param x: вектор 2 фазовых переменных
    :param t: скаляр, время - не используется в автономной системе
    :return: список (вектор) 2 значений правых частей СДУ
    """
    f = [0, 0]
    f[0] = x[1]
    f[1] = - 2*x[0] + (x[0]*x[0]/10)
    return f

def right_hand_side_vdp_linear(x, t):
    """
    функция расчёта правых частей линеаризованной СДУ осциллятора Ван-дер-Поля
    :param x: вектор 2 фазовых переменных
    :param t: скаляр, время - не используется в автономной системе
    :return: список (вектор) 2 значений правых частей СДУ
    """
    f = [0, 0]
    f[0] = x[1]
    f[1] = x[1] - x[0]
    return f

def matrix_row_23(x, y):
    """
    функция вычисления объединённого вектора кронекеровских степеней вектора (x, y) до 3 - го порядка
    :param x: скалярное значение x
    :param y: скалярное значение y
    :return: numpy массив (вектор-строка) длины 9: a_row = (x, y, x**2, xy, y**2, x**3, yx**2, xy**2, y**3)
    """
    a_row = np.zeros(9)
    a_row[0] = x
    a_row[1] = y
    a_row[2] = x**2
    a_row[3] = x*y
    a_row[4] = y**2
    a_row[5] = x**3
    a_row[6] = (x**2)*y
    a_row[7] = x*(y**2)
    a_row[8] = y**3
    return a_row

def matrix_23(z, stride):
    """
    функция вычисления матрицы 9х9 составленной из строк - третьих кронекеровских степеней
    фазовых векторов (x, y) вычисленных в девяти временнЫх отсчётах с шагом stride
    :param z: двумерный массив NT x 2, NT - количество временнЫх отсчётов, 1-й столбец - значения х, 2-й - y
    :param stride: количество временнЫх отсчётов между используемыми при построении матрицы фазовыми точками
    :return: numpy массив (двумерная матрицы 9х9)
    """
    matrix = np.zeros((9, 9))
    for ind in range(1, 10, 1):
        matrix[ind-1] = matrix_row_23(z[(ind-1)*stride + 1, 0], z[(ind-1)*stride + 1, 1])
    return matrix

def right_hand_vector(x, number, h, stride):
    """
    функция вычисления правых частей СЛАУ для нахождения матриц-констант тейлоровского разложения
    правых частей СДУ по кронекеровским степеням фазового вектора
    правые части СЛАУ вычисляются как центральные конечные разности

    :param x: вектор значений одной из компонент фазового вектора для различных временнЫх отсчётов
    :param number: количество временнЫх точек в которых вычисляется центральная конечная разность - длина вектра правых частей СЛАУ
    :param h: длина постоянного временнОго промежутка между соседними временнЫми отсчётами
    :param stride: количество временнЫх отсчётов между точками в которых вычисляется центральная конечная разность
    :return: vector - numpy массив (строка длины number) значений правых частей СЛАУ (центральные конечные разности)
                n_list - numpy массив (строка длины 3*number) индексов - номеров временных отсчётов,
                использованных при вычислении вектора правых частей СЛАУ
    """
    vector = np.zeros(number)
    n_list = np.zeros(3*number)
    for ind in range(number):
        vector[ind] = (x[ind*stride + 2] - x[ind*stride])/(2*h)
        n_list[3*ind] = ind*stride
        n_list[3*ind + 1] = ind*stride + 1
        n_list[3*ind + 2] = ind*stride + 2
    return vector, n_list

def right_hand_23_calculation(p1, p2):
    """
    функция преобразования решений СЛАУ из двух векторов p1 = (p11[0,0], p11[1,0], p12[0,0],...,p13[3,0] )
    и p1 = (p11[0,1], p11[1,1], p12[0,1],...,p13[3,1] ) в три матрицы p11, p12 и p13 - матрицы тейлоровского
    разложения правых частей СДУ для двумерного фазового вектора до третьей кронекеровской степени
    :param p1: вектор длины 9 - решение первой СЛАУ (соответствующей производной первой фазовой переменной)
    :param p2: вектор длины 9 - решение второй СЛАУ (соответствующей производной второй фазовой переменной)
    :return: три numpy массива  - три матрицы размерности 2х2, 2х3 и 2х4
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    matrix_p11 = np.zeros((2, 2))
    matrix_p12 = np.zeros((2, 3))
    matrix_p13 = np.zeros((2, 4))
    matrix_p11[0, :] = p1[0:2]
    matrix_p11[1, :] = p2[0:2]
    matrix_p12[0, :] = p1[2:5]
    matrix_p12[1, :] = p2[2:5]
    matrix_p13[0, :] = p1[5:9]
    matrix_p13[1, :] = p2[5:9]
    return matrix_p11, matrix_p12, matrix_p13

def right_hand_23_vdp():
    """
    функция вычисления трех матриц - матриц p11, p12 и p13 - матрицы тейлоровского
    разложения правых частей СДУ для уравнения осциллятора Ван-дер-Поля
    :return: три numpy массива  - три матрицы размерности 2х2, 2х3 и 2х4
    """
    matrix_p11 = np.zeros((2, 2))
    matrix_p12 = np.zeros((2, 3))
    matrix_p13 = np.zeros((2, 4))
    matrix_p11[0] = [0, 1]
    matrix_p11[1] = [-1, 1]
    matrix_p13[1, 1] = -1
    return matrix_p11, matrix_p12, matrix_p13

def right_hand_23_vdp_linear():
    """
    функция вычисления трех матриц - матриц p11, p12 и p13 - матрицы тейлоровского
    разложения правых частей СДУ для ЛИНЕАРИЗОВАННОГО уравнения осциллятора Ван-дер-Поля
    :return: три numpy массива  - три матрицы размерности 2х2, 2х3 и 2х4
    """
    matrix_p11 = np.zeros((2, 2))
    matrix_p12 = np.zeros((2, 3))
    matrix_p13 = np.zeros((2, 4))
    matrix_p11[0] = [0, 1]
    matrix_p11[1] = [-1, 1]
    return matrix_p11, matrix_p12, matrix_p13

def right_hand_23_deflector():
    """
    функция вычисления трех матриц - матриц p11, p12 и p13 - матрицы тейлоровского
    разложения правых частей СДУ для  уравнения дефлектора с радиусом R=10
    :return: три numpy массива  - три матрицы размерности 2х2, 2х3 и 2х4
    """
    R = 10
    matrix_p11 = np.zeros((2, 2))
    matrix_p12 = np.zeros((2, 3))
    matrix_p13 = np.zeros((2, 4))
    matrix_p11[0] = [0, 1]
    matrix_p11[1] = [-2, 0]
    matrix_p12[1] = [1.0/R, 0, 0]
    return matrix_p11, matrix_p12, matrix_p13


def right_hand_side_vdp_1(x, t):
    """
    функция расчёта правых частей СДУ осциллятора Ван-дер-Поля осуществляемого через матричное разложение
    в ряд Тейлора до третьего порядка
    :param x: вектор 2 фазовых переменных
    :param t: скаляр, время - не используется в автономной системе
    :return: список (вектор) 2 значений правых частей СДУ
    """
    x = np.array(x)
    xx = np.array([x[0]**2, x[0]*x[1], x[1]**2])
    xxx = np.array([x[0]**3, (x[0]**2)*x[1], x[0]*x[1]**2, x[1]**3])
    matr = right_hand_23_vdp()
    p11 = np.array(matr[0])
    p12 = np.array(matr[1])
    p13 = np.array(matr[2])
    f = np.dot(p11, x) + np.dot(p12, xx) + np.dot(p13, xxx)
    return list(f)


def solve_ode_deflector_calculation(xy0, t0, n_time, step):
    """
    функция расчёта решения ОДУ дефлектора,  численное интегрирование ОДУ
    :param xy0: вектор начальных значений фазовых переменных xy0 = [x0, y0]
    :param t0: начальное значение времени - параметр задачи Коши
    :param n_time: конечное значение временнОго отрезка на котором строится решение
    :param step: шаг по времени
    :return: numpy массив (матрица n_time х 2) решения СДУ
    """
    t = []
    for ind in range(t0, n_time + 1, 1):
        t.append(ind/step)
    z = odeint(right_hand_side_deflector, xy0, t)
    return z

def solve_ode_vdp_calculation(xy0, t0, n_time, step):
    """
    функция расчёта решения ОДУ осциллятора Ван-дер-Поля,  численное интегрирование ОДУ
    :param xy0: вектор начальных значений фазовых переменных xy0 = [x0, y0]
    :param t0: начальное значение времени - параметр задачи Коши
    :param n_time: конечное значение временнОго отрезка на котором строится решение
    :param step: шаг по времени
    :return: numpy массив (матрица n_time х 2) решения СДУ
    """
    t = []
    for ind in range(t0, n_time + 1, 1):
        t.append(ind/step)
    z = odeint(right_hand_side_vdp, xy0, t)
    return z

def ode_approximation_calculation(z, number, h, stride):
    """
    функция аппроксимации (восстановления) матриц Тейлоровского разложения правых частей СДУ до 3-го порядка
    :param z: решение аппроксимируемой СДУ
    :param number: количество временнЫх точек в которых вычисляется центральная конечная разность - длина вектра правых частей СЛАУ
    :param h: длина постоянного временнОго промежутка между соседними временнЫми отсчётами ДОЛЖНА БЫТЬ СОГЛАСОВАНА С РЕШЕНИЕМ
    :param stride: количество временнЫх отсчётов между точками в которых вычисляется центральная конечная разность
    :return: три аппроксимированных матрицы Тейлоровского разложения правых частей СДУ до 3-го порядка
    """
    matrix = matrix_23(z, stride)
    v1 = right_hand_vector(z[:, 0], number, h, stride)
    v2 = right_hand_vector(z[:, 1], number, h, stride)

    p1 = np.linalg.solve(matrix, v1[0])
    p2 = np.linalg.solve(matrix, v2[0])
    matrices = right_hand_23_calculation(p1, p2)

    return matrices

def test_calc_approc_deflector():

    xy0 = [-2, 4]
    t0 = 0
    step = 100
    n_time = 10000

    h = 1.0/step     # ОБЯЗАТЕЛЬНО!
    number = 9
    stride = 200

    z = solve_ode_deflector_calculation(xy0, t0, n_time, step)
    matrices = ode_approximation_calculation(z, number, h, stride)

    return matrices

def test_calc_approc_vdp():

    xy0 = [-2, 4]
    t0 = 0
    step = 100
    n_time = 10000

    h = 1.0/step # ОБЯЗАТЕЛЬНО!
    number = 9
    stride = 200

    z = solve_ode_vdp_calculation(xy0, t0, n_time, step)
    matrices = ode_approximation_calculation(z, number, h, stride)

    return matrices



if __name__ == '__main__':

    matrices = test_calc_approc_deflector()
    print('p11', matrices[0])
    print('p12', matrices[1])
    print('p13', matrices[2])