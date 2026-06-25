"""Тестовые модели и их градиенты

Каждая функция принимает точку p = [x, y]
Градиент принимает p и возвращает numpy-массив [df/dx, df/dy].
"""

import numpy as np


def f_quad_good(p):
    x, y = p
    return x ** 2 + 2 * y ** 2


def grad_quad_good(p):
    x, y = p
    return np.array([2 * x, 4 * y])


def g_quad_bad(p):
    x, y = p
    return x ** 2 + 100 * y ** 2


def grad_quad_bad(p):
    x, y = p
    return np.array([2 * x, 200 * y])


def rosenbrock(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def grad_rosenbrock(p):
    x, y = p
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])


def ackley(p):
    x, y = p
    r = np.sqrt(0.5 * (x ** 2 + y ** 2))
    s = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    return -20 * np.exp(-0.2 * r) - np.exp(s) + 20 + np.e


def grad_ackley(p):
    x, y = p
    r = np.sqrt(0.5 * (x ** 2 + y ** 2))
    s = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    if r == 0:
        first_x = first_y = 0.0
    else:
        first_x = 2 * x * np.exp(-0.2 * r) / r
        first_y = 2 * y * np.exp(-0.2 * r) / r
    dx = first_x + np.pi * np.sin(2 * np.pi * x) * np.exp(s)
    dy = first_y + np.pi * np.sin(2 * np.pi * y) * np.exp(s)
    return np.array([dx, dy])


def himmelblau(p):
    x, y = p
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def grad_himmelblau(p):
    x, y = p
    a = x ** 2 + y - 11
    b = x + y ** 2 - 7
    dx = 4 * x * a + 2 * b
    dy = 2 * a + 4 * y * b
    return np.array([dx, dy])
