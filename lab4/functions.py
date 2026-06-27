"""Оракул задачи, генератор квадратичных функций и тестовые функции.

Problem хранит f, grad, hess и считает число их вызовов (nf, ng, nh).
Для квадратичных функций дополнительно доступны матрица A и вектор b.
"""

import numpy as np


class Problem:
    def __init__(self, f, grad, hess, x_star=None, name=""):
        self._f, self._grad, self._hess = f, grad, hess
        self.x_star = x_star
        self.name = name
        self.nf = self.ng = self.nh = 0

    def f(self, x):
        self.nf += 1
        return self._f(np.asarray(x, dtype=float))

    def grad(self, x):
        self.ng += 1
        return self._grad(np.asarray(x, dtype=float))

    def hess(self, x):
        self.nh += 1
        return self._hess(np.asarray(x, dtype=float))

    def reset(self):
        self.nf = self.ng = self.nh = 0


def make_quadratic(n, k, seed=0):
    rng = np.random.default_rng(seed)
    if n == 1:
        eig = np.array([float(k)])
    else:
        eig = np.exp(np.linspace(0.0, np.log(k), n))  # собственные значения от 1 до k
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eig) @ Q.T
    A = 0.5 * (A + A.T)
    x_star = rng.standard_normal(n)
    b = A @ x_star
    f = lambda x: 0.5 * x @ (A @ x) - b @ x
    grad = lambda x: A @ x - b
    hess = lambda x: A
    prob = Problem(f, grad, hess, x_star=x_star, name=f"quad n={n} k={k}")
    prob.A = A
    prob.b = b
    return prob


# ---- тестовые функции (2D) с аналитическими градиентом и гессианом ----

def rosenbrock(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def grad_rosenbrock(p):
    x, y = p
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])


def hess_rosenbrock(p):
    x, y = p
    return np.array([[2 - 400 * (y - x ** 2) + 800 * x ** 2, -400 * x],
                     [-400 * x, 200.0]])


def himmelblau(p):
    x, y = p
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def grad_himmelblau(p):
    x, y = p
    a = x ** 2 + y - 11
    b = x + y ** 2 - 7
    return np.array([4 * x * a + 2 * b, 2 * a + 4 * y * b])


def hess_himmelblau(p):
    x, y = p
    a = x ** 2 + y - 11
    b = x + y ** 2 - 7
    return np.array([[4 * a + 8 * x ** 2 + 2, 4 * x + 4 * y],
                     [4 * x + 4 * y, 4 * b + 8 * y ** 2 + 2]])


def make_rosenbrock():
    return Problem(rosenbrock, grad_rosenbrock, hess_rosenbrock,
                   x_star=np.array([1.0, 1.0]), name="Розенброка")


def make_himmelblau():
    return Problem(himmelblau, grad_himmelblau, hess_himmelblau,
                   x_star=None, name="Химмельблау")
