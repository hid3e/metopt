"""Методы градиентного спуска из лекции 6

Одинаковый критерий остановки: ||grad f(x)|| <= eps
Дополнительно есть ограничение по числу итераций и проверка на расходимость

Каждый метод принимает обертку-счетчик oracle через которую считаются
вызовы функции и градиента и возвращает словарь с результатами
"""

import numpy as np


class Counter:
    # Обертка над функцией и градиентом, которая считает число вызовов

    def __init__(self, f, grad):
        self._f = f
        self._grad = grad
        self.nf = 0
        self.ng = 0

    def f(self, x):
        self.nf += 1
        return self._f(x)

    def grad(self, x):
        self.ng += 1
        return self._grad(x)


def _result(x, traj, iters, converged, oracle):
    return {
        "x": x,
        "traj": np.array(traj),
        "iters": iters,
        "converged": converged,
        "nf": oracle.nf,
        "ng": oracle.ng,
    }


def gd_const(oracle, x0, alpha, eps, max_iter=100000):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        g = oracle.grad(x)
        if np.linalg.norm(g) <= eps:
            converged = True
            break
        x = x - alpha * g
        if np.linalg.norm(x) > 1e6:
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged, oracle)


def gd_armijo(oracle, x0, eps, c1=1e-4, q=0.5, alpha0=1.0, max_iter=100000):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        g = oracle.grad(x)
        if np.linalg.norm(g) <= eps:
            converged = True
            break
        fx = oracle.f(x)
        gg = g @ g
        alpha = alpha0
        while oracle.f(x - alpha * g) > fx - c1 * alpha * gg:
            alpha *= q
            if alpha < 1e-20:
                break
        x = x - alpha * g
        if np.linalg.norm(x) > 1e6:
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged, oracle)


def _wolfe_step(oracle, x, g, fx, c1, c2, a_max=10.0):
    """Поиск шага вдоль направления p = -g, удовлетворяющего сильным условиям Вольфе

    phi(a)  = f(x - a g),  phi'(a) = <grad f(x - a g), -g>
    phi(0) = fx, phi'(0) = -||g||^2 < 0
    Возвращает длину шага a
    """
    dphi0 = -(g @ g)

    def phi(a):
        return oracle.f(x - a * g)

    def dphi(a):
        return oracle.grad(x - a * g) @ (-g)

    def zoom(a_lo, a_hi, phi_lo):
        for _ in range(50):
            a = 0.5 * (a_lo + a_hi)
            phi_a = phi(a)
            if phi_a > fx + c1 * a * dphi0 or phi_a >= phi_lo:
                a_hi = a
            else:
                if abs(dphi(a)) <= -c2 * dphi0:
                    return a
                if dphi(a) * (a_hi - a_lo) >= 0:
                    a_hi = a_lo
                a_lo, phi_lo = a, phi_a
        return 0.5 * (a_lo + a_hi)

    a_prev, phi_prev = 0.0, fx
    a = 1.0
    for i in range(50):
        phi_a = phi(a)
        if phi_a > fx + c1 * a * dphi0 or (i > 0 and phi_a >= phi_prev):
            return zoom(a_prev, a, phi_prev)
        if abs(dphi(a)) <= -c2 * dphi0:
            return a
        if dphi(a) >= 0:
            return zoom(a, a_prev, phi_a)
        a_prev, phi_prev = a, phi_a
        a = min(2 * a, a_max)
    return a


def gd_wolfe(oracle, x0, eps, c1=1e-4, c2=0.9, max_iter=100000):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        g = oracle.grad(x)
        if np.linalg.norm(g) <= eps:
            converged = True
            break
        fx = oracle.f(x)
        alpha = _wolfe_step(oracle, x, g, fx, c1, c2)
        x = x - alpha * g
        if np.linalg.norm(x) > 1e6:
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged, oracle)


def _bracket_min(phi, step=1.0):
    # Расширяем интервал удвоением, пока функция убывает (схема из лекции 6)
    # Возвращает интервал [a, c], внутри которого лежит минимум phi(a), a >= 0
    a, fa = 0.0, phi(0.0)
    b, fb = step, phi(step)
    if fb > fa:
        return a, b
    while True:
        c = 2 * b
        fc = phi(c)
        if fc > fb:
            return a, c
        a, fa = b, fb
        b, fb = c, fc


def _golden(phi, a, b, tol=1e-8):
    # Метод золотого сечения из лекции 2 для минимизации phi на [a, b]
    inv_phi = (np.sqrt(5) - 1) / 2
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = phi(c), phi(d)
    while (b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = phi(d)
    return 0.5 * (a + b)


def gd_steepest(oracle, x0, eps, max_iter=100000):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        g = oracle.grad(x)
        if np.linalg.norm(g) <= eps:
            converged = True
            break

        def phi(a):
            return oracle.f(x - a * g)

        a, b = _bracket_min(phi)
        alpha = _golden(phi, a, b)
        x = x - alpha * g
        if np.linalg.norm(x) > 1e6:
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged, oracle)
