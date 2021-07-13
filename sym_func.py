import numpy as np
import sympy
from sympy import *
from sympy import Symbol, Matrix, pprint
from sympy.matrices.dense import matrix2numpy
from sympy.solvers.ode.systems import matrix_exp
from scipy.integrate import quad
import matplotlib.pyplot as plt
import ode_lie
from scipy.integrate import odeint
import time

def sym_matrix_exponenta(A, t0):
    """
    функция вычисления матричной экспененты в символьном виде
    :param A: квадратная матрица - константа
    :param t0: символьное значение t0
    :return: матричная экспонента в символьнои виде
    """
    t = Symbol('t')
    t0 = sympy.S(t0)
    B = Matrix(A)
    return matrix_exp(B, t-t0)

def sym_matrix_kronecker_2_deg2(A):
    """
    функция вычисления второй кронекеровской степени квадратной матрицы 2х2 в символьном виде
    :param A: матрица 2х2
    :return: символьная матрица 3х3
    """
    A = Matrix(A)
    #pprint(A)
    B = Matrix([[A[0]**2,     2*A[0]*A[1],            A[1]**2        ],
                [A[0]*A[2],   A[0]*A[3] + A[1]*A[2],  A[1]*A[3]   ],
                [A[2]**2,     2*A[2]*A[3],            A[3]**2]       ])

    return B

def sym_matrix_kronecker_2_deg3(A):
    """
    функция вычисления третьей кронекеровской степени квадратной матрицы 2х2 в символьном виде
    :param A: матрица 2х2
    :return: символьная матрица 4х4
    """
    A = Matrix(A)
    #pprint(A)
    B = Matrix([[A[0]**3,       3*A[1]*(A[0]**2),                   3*A[0]*(A[1]**2),                   A[1]**3       ],
                [A[2]*(A[0]**2),  A[3]*(A[0]**2) + 2*A[0]*A[1]*A[2],  A[2]*(A[1]**2) + 2*A[0]*A[1]*A[3],  A[3]*(A[1]**2)  ],
                [A[0]*(A[2]**2),  A[1]*(A[2]**2) + 2*A[0]*A[2]*A[3],  A[0]*(A[3]**2) + 2*A[1]*A[2]*A[3],  A[1]*(A[3]**2)  ],
                [A[2]**3,       3*A[3]*(A[2]**2),                   3*A[2]*(A[3]**2),                   A[3]**3       ]
                ])

    return B

def s_matrix_r11_calculation(matrix_p11, t0=0):
    """
    функция расчёта первой матрицы r11 в тейлоровском разложении решения СДУ для случая двух фазовых переменных
    :param matrix_p11: матрица при первой степени разложения СДУ в ряд Тейлора, размерность 2х2
    :param t0: параметр задачи Коши
    :return: символьная матрица 2х2
    """
    s_matrix_r11 = sym_matrix_exponenta(matrix_p11, t0)
    return s_matrix_r11

def s_matrix_r12_val_calculation(zz, aa, bb, s_matrix_r12_int):
    """
    функция вычисления значения символьного элемента матрицы  s_matrix_r12_int[aa,bb] для значения
    символьной переменной z = zz
    :param zz: значение замещающее символьную переменную z
    :param aa: номер строки
    :param bb: номер столбца
    :param s_matrix_r12_int: символьная матрица s_matrix_r12
    :return: значение элемента символьной матрицы на строке aa и столбце bb при z = zz
    """
    z = Symbol('z')
    s_matrix_r12_v = s_matrix_r12_int.subs(z, zz)
    return s_matrix_r12_v[aa, bb]

def s_matrix_r132_val_calculation(zz, aa, bb, s_matrix_r132_int):
    """
     функция вычисления значения символьного элемента матрицы  s_matrix_r132_int[aa,bb] для значения
     символьной переменной z = zz
     ПОВТОР ФУНКЦИИ s_matrix_r12_val_calculation
     :param zz: значение замещающее символьную переменную z
     :param aa: номер строки
     :param bb: номер столбца
     :param s_matrix_r12_int: символьная матрица s_matrix_r132
     :return: значение элемента символьной матрицы на строке aa и столбце bb при z = zz
     """
    z = Symbol('z')
    s_matrix_r132_v = s_matrix_r132_int.subs(z, zz)
    return s_matrix_r132_v[aa, bb]

def matrix_r12_integrate(s_matrix_r12_int, t0, tt):
    """
    функция расчёта матрицы r12 - матрицы второго слагаемого Тейлоровского разложения решения СДУ
    для двумерного фазового вектора (через численное интегрирование подынтегрального выражения)
    :param s_matrix_r12_int: символьная матрица - подынтегральное выражение для расчёта матрицы r12
    :param t0: параметр задачи Коши - НИЖНИЙ предел интегрирования
    :param tt: ВЕРХНИЙ предел интегрирования
    :return: матрица r12
    """
    a = s_matrix_r12_int.shape[0]
    b = s_matrix_r12_int.shape[1]
    matrix = s_matrix_r12_int
    for index_1 in range(a):
        for index_2 in range(b):
            matrix[index_1, index_2] = quad(s_matrix_r12_val_calculation, t0, tt, args=(index_1, index_2, s_matrix_r12_int))[0]
    return matrix

def matrix_r132_integrate(s_matrix_r132_int, t0, tt):
    """
     функция расчёта второго слагаемого для матрицы r13 - матрицы третьего слагаемого Тейлоровского разложения решения СДУ
     для двумерного фазового вектора (через численное интегрирование подынтегрального выражения)
     :param s_matrix_r132_int: символьная матрица - подынтегральное выражение для расчёта второго слагаемого матрицы r13
     :param t0: параметр задачи Коши - НИЖНИЙ предел интегрирования
     :param tt: ВЕРХНИЙ предел интегрирования
     :return: матрица - второе слагаемое для матрицы r13
     """
    a = s_matrix_r132_int.shape[0]
    b = s_matrix_r132_int.shape[1]
    # print(a, b)
    # pprint(s_matrix_r132_int)
    matrix = s_matrix_r132_int
    # pprint(matrix)
    for index_1 in range(a):
        for index_2 in range(b):
            # print(index_1, index_2)
            # pprint(matrix[index_1, index_2])
            matrix[index_1, index_2] = quad(s_matrix_r132_val_calculation, t0, tt, args=(index_1, index_2, s_matrix_r132_int))[0]
            # print(quad(s_matrix_r132_val_calculation, t0, tt, args=(index_1, index_2, s_matrix_r132_int))[1])
            # pprint(matrix[index_1, index_2])
            # print('###################')
    return matrix


def weight_calculation(matrix_p11, s_matrix_p12, s_matrix_p13, t0, tt):
    """
    функция расчёта трёх матриц - весовых матриц в Тейлоровском разложении до третьего порядка
    правой части СДУ с двумя фазовыми переменными
    !!! ДЛЯ ТРЕТЬЕЙ МАТРИЦЫ ИСПОЛЬЗУЕТСЯ ТОЛЬКО ВТОРОЕ СЛАГАЕМОЕ !!!
    :param matrix_p11: первая матрица Тейлоровского разложения правой части СДУ (для 1-й степени Кронекера)
    :param s_matrix_p12: вторая матрица Тейлоровского разложения правой части СДУ (для 2-й степени Кронекера)
    :param s_matrix_p13: третья матрица Тейлоровского разложения правой части СДУ (для 3-й степени Кронекера)
    :param t0: параметр задачи Коши
    :param tt: параметр (время) для которого строится отображение Ли
    :return: три матрицы весов - матрицы в Тейлоровском разложении до 3-й степени
    """

    z = Symbol('z')
    t = Symbol('t')

    s_matrix_r11 = s_matrix_r11_calculation(matrix_p11, t0)

    s_matrix_r11_z = s_matrix_r11_calculation(matrix_p11, z)
    # pprint(s_matrix_r11_z)

    s_matrix_r11_z_t0 = s_matrix_r11.subs(t, z)

    s_matrix_r22 = sym_matrix_kronecker_2_deg2(s_matrix_r11)
    s_matrix_r33 = sym_matrix_kronecker_2_deg3(s_matrix_r11)

    s_matrix_r22_z = s_matrix_r22.subs(t, z)
    s_matrix_r33_z = s_matrix_r33.subs(t, z)

    s_matrix_r12_int = s_matrix_r11_z * s_matrix_p12
    s_matrix_r12_int = s_matrix_r12_int * s_matrix_r22_z

    s_matrix_r12_int = s_matrix_r12_int.subs(t, tt)

    s_matrix_r12 = matrix_r12_integrate(s_matrix_r12_int, t0, tt)


    s_matrix_r132_int = s_matrix_r11_z * s_matrix_p13
    s_matrix_r132_int = s_matrix_r132_int * s_matrix_r33_z
    s_matrix_r132_int = s_matrix_r132_int.subs(t, tt)

    s_matrix_r132 = matrix_r132_integrate(s_matrix_r132_int, t0, tt)

    s_matrix_r13 = s_matrix_r132

    s_matrix_r11 = s_matrix_r11.subs(t, tt)
    s_matrix_r11 = s_matrix_r11.evalf()
    s_matrix_r12 = s_matrix_r12.evalf()
    s_matrix_r13 = s_matrix_r13.evalf()

    return s_matrix_r11, s_matrix_r12, s_matrix_r13

def lie_solver_23(p11, p12, p13, t0, xy0, n_time, step):
    """
    функция расчёта решения ОДУ с 2 фазовыми переменными и разложением правой части до 3-го порядка
    :param p11: матрица размерности 2х2 при 1-й Кронекеровской степени в разложении правой части
    :param p12: матрица размерности 2х3 при 2-й Кронекеровской степени в разложении правой части
    :param p13: матрица размерности 2х4 при 3-й Кронекеровской степени в разложении правой части
    :param t0: начальное значение времени - параметр задачи Коши
    :param xy0: вектор начальных значений фазовых переменных xy0 = [x0, y0]
    :param n_time: конечное значение временнОго отрезка на котором строится решение
    :param step: шаг по времени
    :return: numpy массив (матрица n_time х 2) решения СДУ
    """

    t = Symbol('t')
    x0 = xy0[0]
    y0 = xy0[1]

    matrix_p11 = Matrix(p11)
    s_matrix_p12 = Matrix(p12)
    s_matrix_p13 = Matrix(p13)

    xy_sol = np.zeros((n_time, 2))

    for i in range(0, n_time, 1):
        tt = i / step
        print('tt = ',tt)
        weights = weight_calculation(matrix_p11, s_matrix_p12, s_matrix_p13, t0, tt)
        w1 = np.array(weights[0])
        w2 = np.array(weights[1])
        w3 = np.array(weights[2])

        x1 = np.array([[x0], [y0]])
        x2 = np.array([[x0**2], [x0*y0], [y0**2]])
        x3 = np.array([[x0**3], [y0*(x0**2)], [x0*(y0**2)], [y0**3]])
        rez = np.dot(w1, x1) + np.dot(w2, x2) + np.dot(w3, x3)
        xy_sol[i, 0] = rez[0]
        xy_sol[i, 1] = rez[1]

    return xy_sol

def test_deflector_1():
    matrices = ode_lie.test_calc_approc_deflector()
    matrix_p11 = Matrix(matrices[0])
    s_matrix_p12 = Matrix(matrices[1])
    s_matrix_p13 = Matrix(matrices[2])

    t0 = 0
    xy0 = [-3, 5]
    n_time = 41
    step = 10

    zzz = ode_lie.solve_ode_deflector_calculation(xy0, t0, n_time, step)
    z = lie_solver_23(matrix_p11, s_matrix_p12, s_matrix_p13, t0, xy0, n_time, step)

    fig = plt.figure()
    plt.plot(z[:, 0], z[:, 1], '-ro')
    plt.plot(zzz[:, 0], zzz[:, 1], '-bo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return 0


if __name__ == '__main__':

    test_deflector_1()


