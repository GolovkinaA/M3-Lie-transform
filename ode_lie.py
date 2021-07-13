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

def right_hand_side_2_3(z, t, p11, p12, p13):
    """
    функция расчёта правых частей СДУ 2 уравнения, 3 порядок полиномиальной правой части
    :param x: вектор 2 фазовых переменных
    :param t: скаляр, время - не используется в автономной системе
    :return: список (вектор) 2 значений правых частей СДУ
    """

    # p11 = [[0, a], [b, 0]]
    # p12 = [[c, 0, 0], [0, 0, 0]]
    # p13 = [[0, 0, 0, 0], [0, 0, 0, 0]]


    p11 = np.array(p11)
    p12 = np.array(p12)
    p13 = np.array(p13)
    x = z[0]
    y = z[1]
    x1 = np.array([x, y])
    x2 = np.array([x**2, x*y, y**2])
    x3 = np.array([x**3, y*(x**2), x*(y**2), y**3])
    f = np.dot(p11, x1.transpose()) + np.dot(p12, x2.transpose()) + np.dot(p13, x3.transpose())
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
        # print('!!!', ind)
        # print((ind-1)*stride + 1)
        matrix[ind-1] = matrix_row_23(z[(ind-1)*stride + 1, 0], z[(ind-1)*stride + 1, 1])
    return matrix

def right_hand_vector(x, number, h, stride):
    """
    УСТАРЕВШАЯ
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


def right_hand_vector_calculation(x, t, number, stride):
    """
    функция вычисления правых частей СЛАУ для нахождения матриц-констант тейлоровского разложения
    правых частей СДУ по кронекеровским степеням фазового вектора
    правые части СЛАУ вычисляются как центральные конечные разности

    :param x: вектор значений одной из компонент фазового вектора для различных временнЫх отсчётов
    :param t: сетка на которой задан x
    :param number: количество временнЫх точек в которых вычисляется центральная конечная разность - длина вектра правых частей СЛАУ7
    :param stride: количество временнЫх отсчётов между точками в которых вычисляется центральная конечная разность
    :return: vector - numpy массив (строка длины number) значений правых частей СЛАУ (центральные конечные разности)
                n_list - numpy массив (строка длины 3*number) индексов сетки t - номеров временных отсчётов,
                использованных при вычислении вектора правых частей СЛАУ
                t_new - numpy массив (строка длины number)  - новая сетка, на которой вычислены значения
                разностной производной
    """
    h = t[1]-t[0]                        # шаг сетки
    vector = np.zeros(number)
    n_list = np.zeros(3*number)
    t_new = np.zeros(number)
    for ind in range(number):
        vector[ind] = (x[ind*stride + 2] - x[ind*stride])/(2*h)
        t_new[ind] = t[ind*stride + 1]
        n_list[3*ind] = ind*stride
        n_list[3*ind + 1] = ind*stride + 1
        n_list[3*ind + 2] = ind*stride + 2
    return vector, n_list, t_new


def get_grid_stride(t, number, stride):
    """
    функция расчёта сетки длины number полученной из исходной начиная с элемента 1 с шагом stride
    :param t: исходная сетка
    :param number: количество узлов новой сетки
    :param stride: шаг с которым берутся значения из исходной сетки
    :return: новая сетка
    """
    t_new = []
    for ind in range(number):
        t_new.append(t[ind*stride + 1])
    return t_new

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


def get_grid(t0, n_step, step):
    """
    функция расчёта одномерной сетки на отрезке [t0, n_step*step] с шагом step
    :param t0: левый край сетки- начальное значение
    :param n_step: число шагов сетки
    :param step: шаг сетки
    :return: сетка (список)
    """
    t = []
    for ind in range(0, int(n_step) + 1, 1):
        t.append(t0 + ind*step)
    return t


def solve_ode_deflector_calculation(xy0, t0, n_step, step):
    """
    функция расчёта решения ОДУ дефлектора,  численное интегрирование ОДУ
    :param xy0: вектор начальных значений фазовых переменных xy0 = [x0, y0]
    :param t0: начальное значение времени - параметр задачи Коши
    :param n_step: число шагов по времени, конец временного отрезка - n_step*step
    :param step: шаг по времени
    :return: numpy массив (матрица n_time х 2) решения СДУ, список (вектор) значений времени
    """
    t = get_grid(t0, n_step, step)
    z = odeint(right_hand_side_deflector, xy0, t)
    return z, t

def solve_ode_vdp_calculation(xy0, t0, n_step, step):
    """
    функция расчёта решения ОДУ осциллятора Ван-дер-Поля,  численное интегрирование ОДУ
    :param xy0: вектор начальных значений фазовых переменных xy0 = [x0, y0]
    :param t0: начальное значение времени - параметр задачи Коши
    :param n_step: число шагов по времени, конец временного отрезка - n_step*step
    :param step: шаг по времени
    :return: numpy массив (матрица n_step х 2) решения СДУ, список (вектор) значений времени
    """
    t = get_grid(t0, n_step, step)
    z = odeint(right_hand_side_vdp, xy0, t)
    return z, t

def ode_approximation_calculation(z, t, number, stride):
    """
    функция аппроксимации (восстановления) матриц Тейлоровского разложения правых частей СДУ до 3-го порядка
    :param z: решение аппроксимируемой СДУ
    :param t: сетка на которой задано решение аппроксимируемой СДУ
    :param number: количество временнЫх точек в которых вычисляется центральная конечная разность - длина вектра правых частей СЛАУМ
    :param stride: количество временнЫх отсчётов между точками в которых вычисляется центральная конечная разность
    :return: три аппроксимированных матрицы Тейлоровского разложения правых частей СДУ до 3-го порядка
            сетка временных точек в которых вычислялись значения правых частей СДУ для аппроксимации матриц
    """
    matrix = matrix_23(z, stride)
    v1 = right_hand_vector_calculation(z[:, 0], t, number, stride)
    v2 = right_hand_vector_calculation(z[:, 1], t, number, stride)

    p1 = np.linalg.solve(matrix, v1[0])
    p2 = np.linalg.solve(matrix, v2[0])
    matrices = right_hand_23_calculation(p1, p2)

    return matrices

def test_calc_approc_deflector():

    xy0 = [-2, 4]
    t0 = 0
    step = 0.01
    n_time = 10000

    number = 9
    stride = 200

    z = solve_ode_deflector_calculation(xy0, t0, n_time, step)
    matrices = ode_approximation_calculation(z[0], z[1], number, stride)

    return matrices

def test_calc_approc_vdp():

    xy0 = [-2, 4]
    t0 = 0
    step = 0.01
    n_step = 10000

    number = 9
    stride = 200

    z = solve_ode_vdp_calculation(xy0, t0, n_step, step)
    matrices = ode_approximation_calculation(z[0], z[1], number, stride)

    return matrices

def trigonometric_approximation(f, length, t, tt):
    """
    функция расчёта тригонометрического полинома в точках вектора t на отрезке [0, length]
    и в точках вектора tt на отрезке [0, length]
    :param f: значения аппроксимируемой функции
    :param length: длина отрезка на котором аппроксимируется функция
    :param t: вектор точек в которых вычисляются значения тригонометрического полинома
    :param tt: вектор точек в которых вычисляются значения тригонометрического полинома (контрольная сетка)
    :return: вектор значений тригонометрического полинома на сетке t
                вектор значений тригонометрического полинома на контрольной сетке tt
    """

    nz = len(f)
    nt = len(t)
    ntt = len(tt)

    M = (nz - 1)//2

    fkc = []
    fks = []

    z = []
    zz = []

    fkc.append((2/nz)*sum(f))
    fks.append(0)

    for ind_k in range(1, M+1, 1):
        fkc_t = 0
        fks_t = 0
        for ind_i in range(nz):
            cos_ki = math.cos(2*math.pi*ind_k*ind_i/nz)
            sin_ki = math.sin(2*math.pi*ind_k*ind_i/nz)
            fkc_t = fkc_t + f[ind_i]*cos_ki
            fks_t = fks_t + f[ind_i]*sin_ki
        fkc.append(fkc_t*2/nz)
        fks.append(fks_t*2/nz)

    for ind_i in range(nt):
        z_t = fkc[0]/2
        for ind_k in range(1, M+1, 1):
            omegak = 2 * math.pi * ind_k / length
            cos_kz = math.cos(omegak*t[ind_i])
            sin_kz = math.sin(omegak*t[ind_i])
            z_t = z_t + fkc[ind_k]*cos_kz + fks[ind_k]*sin_kz
        z.append(z_t)

    for ind_i in range(ntt):
        zz_tt = fkc[0] / 2
        for ind_k in range(1, M + 1, 1):
            omegak = 2 * math.pi * ind_k / length
            cos_kz = math.cos(omegak * tt[ind_i])
            sin_kz = math.sin(omegak * tt[ind_i])
            zz_tt = zz_tt + fkc[ind_k] * cos_kz + fks[ind_k] * sin_kz
        zz.append(zz_tt)
    return z, zz

def get_grid_func(f, t, nz):
    """
    функция формирования сеточной функции на более редкой сетке
    :param f: список значений сеточной функции
    :param t: исходная сетка
    :param nz: количество узлов новой сетки НЕЧЕТНОЕ!!!
    :return: z - список значений сеточной функции на новой сетке с nz узлами,
             t_new - новая сетка
    """
    nz = int(nz)
    if nz%2 == 0:
        nz = nz + 1

    nt = len(t)
    if nt%(nz - 1)==0:
        hz = nt//(nz)
    else:
        hz = nt//(nz - 1)

    z = []
    t_new = []

    for ind in range(nz):
        # print(ind, ind*hz, t[ind*hz])
        t_new.append(t[ind*hz])
        z.append(f[ind*hz])

    return z, t_new

def get_reduction_curve(curve, reduction_coeff):
    """
    функция формирования незамкнутой кривой на более редкой сетке
    :param curve: кривая задана numpy массивом размерности n_step x 2
    :param reduction_coeff: коэффициент уменьшения числа точек на кривой - количество раз (по умолчанию 2)
    :return: z - список значений сеточной функции на новой сетке с (n_step-1)/reduction_coeff узлами,
             t_new - новая сетка
    """
    reduction_coeff = int(reduction_coeff)
    if reduction_coeff <=1:
        reduction_coeff = 2

    t = list(curve[1])
    z = curve[0]

    if reduction_coeff >= len(z):
        reduction_coeff = 2

    length = int(len(z)//reduction_coeff)

    z_new = np.zeros((length, 2))
    t_new = []

    for ind in range(length):
        z_new[ind, :] = z[ind*reduction_coeff, :]
        t_new.append(t[ind*reduction_coeff])

    return z_new, t_new



def get_linear(x1, t1, x2, t2, n_step):
    """
    функция расчёта сеточной линейной функции построенной по точкам (t1, x1) и (t2, x2) на сетке с шагом t_step
    :param x1: ордината начальной точки прямой
    :param t1: абсцисса начальной точки прямой
    :param x2: ордината конечной точки прямой
    :param t2: абсцисса конечной точки прямой
    :param n_step: шаг сетки на которой строится сеточная функция
    :return: x - список значений линейной функции на сетке
             t - сетка (список)
    """
    t = []
    x = []
    t_time = int((t2 - t1)//n_step)
    alpha = (x2 - x1)/(t2 - t1)
    beta = x1 - alpha * t1
    for ind in range(t_time + 2):
        temp_t = t1 + n_step*ind
        temp_x = alpha * temp_t + beta
        t.append(temp_t)
        x.append(temp_x)
    return x, t

def get_linear_addition(curve, close_coeff):
    """
    функция дополнения незамкнутой кривой замыкающим линейным отрезком
    :param curve: кривая задана numpy массивом размерности n_time x 2
    :param close_coeff: коэффициент увеличения отрезка времени на котором строится замкнутая кривая
    :return: z - кривая задана numpy массивом размерности n_time* close_coeff x 2
            t - сетка (время, аргумент сеточной функции), список
    """
    if close_coeff<=1:
        close_coeff = 1.5

    t = list(curve[1])
    z = curve[0]

    z_x = list(z[:, 0])
    z_y = list(z[:, 1])

    x0 = z_x[0]
    y0 = z_y[0]

    t1 = t[-1]
    x1 = z_x[-1]
    y1 = z_y[-1]

    t_close = close_coeff * t1

    step = t[1] - t[0]

    lin_x = get_linear(x1, t1, x0, t_close, step)
    lin_y = get_linear(y1, t1, y0, t_close, step)
    x_add = lin_x[0]
    y_add = lin_y[0]
    t_add = lin_x[1]

    del x_add[0]
    del y_add[0]
    del t_add[0]

    z_x = z_x + x_add
    z_y = z_y + y_add
    t = t + t_add

    z = np.array((z_x, z_y))
    z = z.transpose()

    return z, t

def linear_approximation(x, y, t, t_new):
    """
    функция расчёта линейной аппроксимации двумерной кривой заданной на равномерной сетке t
    в точках сетки t_new, в общем случае неравномерной
    :param x:
    :param y:
    :param t: сетка на которой задана кривая
    :param t_new: новая сетка на которой вычмсляется линейная аппроксимация
    :return:
    """
    n_t = int(len(t))
    t_start = t[0]
    t_end = t[-1]
    h = t[1] - t[0]
    t_period = t_end - t_start

    n_t_new = int(len(t_new))
    x_new = []
    y_new = []

    for ind_t_new in range(n_t_new):
        ind_t = int(t_new[ind_t_new]//h) + 1

        t1 = t[ind_t - 1]
        t2 = t[ind_t]
        x1 = x[ind_t - 1]
        x2 = x[ind_t]
        y1 = y[ind_t - 1]
        y2 = y[ind_t]

        alpha_x = (x2 - x1) / h
        beta_x = x1 - alpha_x * t1

        alpha_y = (y2 - y1) / h
        beta_y = y1 - alpha_y * t1


        temp_x = alpha_x * t_new[ind_t_new] + beta_x
        temp_y = alpha_y * t_new[ind_t_new] + beta_y
        x_new.append(temp_x)
        y_new.append(temp_y)

    return x_new, y_new


def del_linear(t, x, y, t1):
    """
    функция удаления отрезка кривой, заданной как сеточная функция
    :param t: сетка (список) на которой задана кривая (две сеточные функции x и y)
    :param x: значения первой функции (фазовой переменной) на сетке
    :param y: значения первой функции (фазовой переменной) на сетке
    :param t1: скалярное значение аргумента начиная с которого удаляется часть кривой
    :return: t - новая (укороченная) сетка на которой задана кривая без удалённого отрезка,
                x - значения первой функции (фазовой переменной) на укороченной сетке,
                y - значения второй функции (фазовой переменной) на укороченной сетке
    """
    for ind in range(len(t)):
        if t[ind] >= t1:
            ind_exit = ind
            break
    del t[ind_exit:]
    del x[ind_exit:]
    del y[ind_exit:]

    return t, x, y


def errors_calculation(x1, y1, x2, y2):
    """
    функция расчёта метрик отличия двух кривых, заданных на одинаковой сетке
    :param x1: абсцисса первой кривой заданная на сетке
    :param y1: ордината первой кривой заданная на сетке
    :param x2: абсцисса второй кривой заданная на сетке
    :param y2: ордината второй кривой заданная на сетке
    :return: суммарная разность модулей абсцисс и ординат вычисленных на сетке,
            максимальная разность модулей абсцисс и ординат вычисленных на сетке
    """
    if len(x1)==len(y1) and len(y1)==len(x2) and len(x2)==len(y2):
        n_t_error = int(len(x1))
    else:
        exit(2)
    sum_error = 0
    error_max = 0
    for ind in range(n_t_error):
        error_x = abs(x2[ind] - x1[ind])
        error_y = abs(y2[ind] - y1[ind])
        sum_error = sum_error + error_x + error_y
        if error_x > error_max:
            error_max = error_x
        if error_y > error_max:
            error_max = error_y
    return sum_error, error_max


def get_training_data_deflector(x0, y0, t0, t1, step, reduction_coeff=10):
    """
    функция формирования тренировочных данных для восстановления правых частей ОДУ дефлектора
    :param x0: задача Коши x0
    :param y0: задача Коши y0
    :param t0: начальное значение времени
    :param t1: конечное значение времени
    :param step: шаг по времени для сетки на которой получается численное решение СДУ
    :param reduction_coeff: коэффициент урежения сетки для имитации исходных данных
    :return: sol_new - список значений  - исходные данные задачи восстановления (x, y),
            сетка
    """

    xy0 = [x0, y0]
    n_step = int((t1 - t0) // step)  # количество шагов сетки на которой получается численное решение СДУ

    sol = solve_ode_deflector_calculation(xy0, t0, n_step, step)    # расчёт численного решения СДУ на сетке с шагом step
    t_sol = list(sol[1])                                            # сетка с шагом step на которой вычислено решение

    sol_new = get_reduction_curve(sol, reduction_coeff)  # формирование решения на уменьшенной сетке - имитация исходных данных

    return sol_new

def get_approc_matrices_with_approc_trig_pol(train_data, close_coeff, nz, step_new, stride):
    """
    GAMWATP
    функция восстановления матриц полиномиальной правой части СДУ 2 уравнения 3 степени и расчёта ошибок
    приближения тригонометрическим полиномом тренировочных данных (с использованием их линейной аппроксимации)
    в точках используемых для рапсчёта значений восстанавливаемых правых частей СДУ
    :param train_data: тренировочные данные : [numpy массив len(t) x 2, список значений t]
    :param close_coeff: во сколько раз увеличивается временнОй отрезок для замыкания кривой
    :param nz: количество узлов сетки для расч1та коэффициентов тригонометрического полинома
    :param step_new: шаг сетки на которой вычисляются значения тригонометрического полинома
    :param stride: количество шагов сетки между узлами в которых вычисляются значения для восстановления матриц СДУ
    :return: matrices  - три восстановленные матрицы p11, p12  и p13 полиномиальной правой части СДУ,
            sum_error - сумма абсолютных значений отклонений значения тригонометрического полинома от линейной аппроксимации тренировочных данных,
            error_max - максимальное абсолютное значение отклонений значения тригонометрического полинома от линейной аппроксимаци
    """

    z_train = train_data[0]
    t_train = train_data[1]
    x_train = z_train[:, 0]
    y_train = z_train[:, 1]

    t0 = t_train[0]
    t1 = t_train[-1]

    z_common = get_linear_addition(train_data, close_coeff)   # дополнение (замыкание)  кривой линейным отрезком
    t = list(z_common[1])
    z = z_common[0]

    t_close = t[-1]              # конечное значение времени для замкнутой линейным отрезком кривой

    z_x = list(z[:, 0])
    z_y = list(z[:, 1])

    # plt.plot(z_x, z_y)
    # plt.show()
    # plt.plot(t, z_x)
    # plt.show()
    # plt.plot(t, z_y)
    # plt.show()

    grid_x = get_grid_func(z_x, t, nz)     # формирование значений для построения триг. полинома первой фазовой переменной
    grid_y = get_grid_func(z_y, t, nz)     # формирование значений для построения триг. полинома второй фазовой переменной

    length = grid_x[1][-1]                 # конечное значение времени в сетке на которой строится полином

    n_step_new = int((t_close - t0)//step_new)  # количество узлов сетки на которой вычисляются значения триг. полинома

    number = 9     # треть количества точек для нахождения 18 неизвестных параметров полиномиальной СДУ 2 уравнения 3 степени
    stride_max = n_step_new//number
    # print('stride_max', stride_max)

    t_new = get_grid(t0, n_step_new, step_new)  # формирование сетки на которой вычисляются значения триг. полинома

    t_error = get_grid_stride(t_new, number, stride)  # формирование сетки на которой вычисляются значения полинома и линейной аппрексимации тренировочных данных для расчё та ошибки

    pol_x_common = trigonometric_approximation(grid_x[0], length, t_new, t_error) # вычисляются значения триг. полинома
    pol_x = pol_x_common[0]
    pol_x_error = pol_x_common[1]
    pol_y_common = trigonometric_approximation(grid_y[0], length, t_new, t_error) # вычисляются значения триг. полинома
    pol_y = pol_y_common[0]
    pol_y_error = pol_y_common[1]

    curves = del_linear(t_new, pol_x, pol_y, t1)  # удаление аппроксимации линейного отрезка тригонометрическим полиномом

    t_new_1 = curves[0]
    pol_x = curves[1]
    pol_y = curves[2]

    z_linear = linear_approximation(x_train, y_train, t_train, t_error)

    z_x_linear = z_linear[0]
    z_y_linear = z_linear[1]

    errors = errors_calculation(pol_x_error, pol_y_error, z_x_linear, z_y_linear)

    sum_error = errors[0]
    error_max = errors[1]

    # plt.figure()
    # plt.plot(z_x, z_y, 'ro')
    # plt.plot(pol_x, pol_y, 'bo')
    # plt.figure()
    # plt.plot(t, z_x, 'ro')
    # plt.plot(t_new_1, pol_x, 'bo')
    # plt.figure()
    # plt.plot(t, z_y, 'ro')
    # plt.plot(t_new_1, pol_y, 'bo')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    curve = np.array((pol_x, pol_y))
    curve = curve.transpose()

    matrices = ode_approximation_calculation(curve, t_new_1, number, stride)

    return matrices, sum_error, error_max

def test_approc_matrices_with_approc_trig_pol_deflector(x0, y0, t0, t1, step, reduction_coeff):


    #1. Матрицс СДУ дефлектора  - для сравнения

    p11 = [[0, 1], [-2, 0]]
    p12 = [[0.1, 0, 0], [0, 0, 0]]
    p13 = [[0, 0, 0, 0], [0, 0, 0, 0]]

    matr0 = (p11, p12, p13)

    #2. формирование тренировочных данных для воостановления матриц правой части СДУ дефлектора

    # x0 = -2
    # y0 = 4
    # t0 = 0
    # t1 = 4
    # step = 0.01
    # reduction_coeff = 10

    train_data = get_training_data_deflector(x0, y0, t0, t1, step, reduction_coeff)

    step_new = 0.001     # шаг сетки на которой вычисляются значения тригонометрического полинома

    min_pol_er = 10**10
    min_sol_er = 10**10

    for stride in [400, 300]:
        for ind_nz in range(5):
            nz = 5 + 2 * ind_nz

            er_pol = []
            er_pol_max = []
            er_p11 = []
            list_close_coeff = []

            for ind_close_coeff in range(20):
                close_coeff = 1.5 + 0.1 * ind_close_coeff   # коэффициент увеличения отрезка времени для замыкания линейным отрезком

                matrices_1 = get_approc_matrices_with_approc_trig_pol(train_data, close_coeff, nz, step_new, stride)
                pp11 = matrices_1[0][0]
                pp12 = matrices_1[0][1]
                pp13 = matrices_1[0][2]

                matr1 = (pp11, pp12, pp13)

                ttt = get_grid(0, 1600, 0.01)
                z0 = odeint(right_hand_side_2_3, [-2, 4], ttt, matr0)
                z1 = odeint(right_hand_side_2_3, [-2, 4], ttt, matr1)

                # print('pp11', pp11)
                # print('pp12', pp12)
                # print('pp13', pp13)

                sol_err = errors_calculation(z0[:, 0], z0[:, 1], z1[:, 0], z1[:, 1])

                p11_error = abs(p11 - matrices_1[0][0])
                er_p11.append(p11_error.max())
                er_pol.append(matrices_1[1])
                er_pol_max.append(matrices_1[2])
                list_close_coeff.append(close_coeff)

                if matrices_1[1] < min_pol_er:
                    min_pol_er = matrices_1[1]
                    pp11_min_pol_er = pp11
                    pp12_min_pol_er = pp12
                    pp13_min_pol_er = pp13

                if sol_err[0]/len(ttt) < min_sol_er:
                    min_sol_er = sol_err[0]/len(ttt)
                    pp11_min_sol_er = pp11
                    pp12_min_sol_er = pp12
                    pp13_min_sol_er = pp13

                # print('p11_er', stride, nz, close_coeff, p11_error.max())
                # print('pol_er', stride, nz, close_coeff, matrices_1[1])
                # print('sol_er', stride, nz, close_coeff, sol_err[0]/len(ttt))
                # print('------------------------------------------------------')

                # plt.plot(z0[:, 0], z0[:, 1], '-bo')
                # plt.plot(z1[:, 0], z1[:, 1], '-ro')
                # plt.xlabel('x')
                # plt.ylabel('y')
                # plt.show()

                # print('p11_error', abs(matrices_0[0] - matrices_1[0][0]))
                # print('p12_error', abs(matrices_0[1] - matrices_1[0][1]))
                # print('p13_error', abs(matrices_0[2] - matrices_1[0][2]))

                # print('er_pol', er_pol)
                # print('er_pol_max', er_pol_max)
                # print('er_p11', er_p11)
                # print(list_close_coeff)

            np_er_pol = np.array(er_pol)
            np_er_p11 = np.array(er_p11)

            # print('min p11', stride, nz, np_er_p11.min(), np_er_p11.argmin())
            # print('min pol', stride, nz, np_er_pol.min(), np_er_pol.argmin())
            #
            # plt.plot(list_close_coeff, er_pol, 'ro')
            # plt.xlabel('close_coeff')
            # plt.ylabel('error pol')
            # plt.show()
            # plt.plot(list_close_coeff, er_pol_max, 'go')
            # plt.xlabel('close_coeff')
            # plt.ylabel('error pol max')
            # plt.show()
            # plt.plot(list_close_coeff, er_p11, 'bo')
            # plt.xlabel('close_coeff')
            # plt.ylabel('error p11')
            # plt.show()

    print('min_pol_er', min_pol_er)
    print('pp11_min_pol_er', pp11_min_pol_er)
    print('pp12_min_pol_er', pp12_min_pol_er)
    print('pp13_min_pol_er', pp13_min_pol_er)
    print('min_sol_er', min_sol_er)
    print('pp11_min_sol_er', pp11_min_sol_er)
    print('pp12_min_sol_er', pp12_min_sol_er)
    print('pp13_min_sol_er', pp13_min_sol_er)

    ttt = get_grid(0, 1600, 0.01)
    matr1 = (pp11_min_pol_er, pp12_min_pol_er, pp13_min_pol_er)
    matr2 = (pp11_min_sol_er, pp12_min_sol_er, pp13_min_sol_er)
    z0 = odeint(right_hand_side_2_3, [-2, 4], ttt, matr0)
    z1 = odeint(right_hand_side_2_3, [-2, 4], ttt, matr1)
    z2 = odeint(right_hand_side_2_3, [-2, 4], ttt, matr2)

    plt.plot(z0[:, 0], z0[:, 1], '-bo')
    plt.plot(z1[:, 0], z1[:, 1], '-ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.plot(z0[:, 0], z0[:, 1], '-bo')
    plt.plot(z2[:, 0], z2[:, 1], '-ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



    return 0

if __name__ == '__main__':



    test_approc_matrices_with_approc_trig_pol_deflector(-4, 6, 0, 4, 0.01, 20)
    #
    test_approc_matrices_with_approc_trig_pol_deflector(-2, 4, 0, 4, 0.01, 20)

    test_approc_matrices_with_approc_trig_pol_deflector(-2, 4, 0, 4, 0.01, 10)
    #
    #
    # matrices_0 = test_calc_approc_deflector()
    #
    # p11 = matrices_0[0]
    # p12 = matrices_0[1]
    # p13 = matrices_0[2]
    # print('p11', matrices_0[0])
    # print('p12', matrices_0[1])
    # print('p13', matrices_0[2])
