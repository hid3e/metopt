"""Модификации градиентного спуска из лекции 7.

Все методы первого порядка: используют только градиент (функцию не вычисляют),
поэтому считаем число итераций. Все операции с векторами (квадрат, корень,
деление) - покоординатные.

Критерий остановки общий: ||grad f(x)|| <= tol. Плюс ограничение по числу
итераций и проверка на расходимость (точка ушла слишком далеко).

Обозначения: tol - точность остановки по градиенту (ε из условия, 1e-8),
eps - маленький регуляризатор в знаменателе адаптивных методов.
"""

import numpy as np

def _result(x, traj, iters, converged):
    return {"x": x, "traj": np.array(traj), "iters": iters, "converged": converged}

def momentum(grad, x0, alpha=0.01, beta=0.9, tol=1e-8, max_iter=100000):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            converged = True
            break
        m = beta * m + g
        x = x - alpha * m
        if np.linalg.norm(x) > 1e6:   # точка улетела слишком далеко - метод разошелся
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged)

def nesterov(grad, x0, alpha=0.01, beta=0.9, tol=1e-8, max_iter=100000):
    x = np.array(x0, dtype=float)
    x_prev = x.copy()
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        if np.linalg.norm(grad(x)) <= tol:
            converged = True
            break
        y = x + beta * (x - x_prev)
        g = grad(y)
        x_prev = x
        x = y - alpha * g
        if np.linalg.norm(x) > 1e6:   # точка улетела слишком далеко - метод разошелся
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged)

def adagrad(grad, x0, alpha=1.0, eps=1e-8, tol=1e-8, max_iter=100000):
    x = np.array(x0, dtype=float)
    G = np.zeros_like(x)
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            converged = True
            break
        G = G + g * g
        x = x - alpha * g / (np.sqrt(G) + eps)
        if np.linalg.norm(x) > 1e6:   # точка улетела слишком далеко - метод разошелся
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged)

def rmsprop(grad, x0, alpha=0.01, rho=0.9, eps=1e-8, tol=1e-8, max_iter=100000):
    x = np.array(x0, dtype=float)
    G = np.zeros_like(x)
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            converged = True
            break
        G = rho * G + (1 - rho) * g * g
        x = x - alpha * g / (np.sqrt(G) + eps)
        if np.linalg.norm(x) > 1e6:   # точка улетела слишком далеко - метод разошелся
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged)

def adadelta(grad, x0, rho=0.95, eps=1e-6, tol=1e-8, max_iter=100000):
    x = np.array(x0, dtype=float)
    G = np.zeros_like(x)
    u = np.zeros_like(x)
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            converged = True
            break
        G = rho * G + (1 - rho) * g * g
        dx = -np.sqrt(u + eps) / np.sqrt(G + eps) * g
        u = rho * u + (1 - rho) * dx * dx
        x = x + dx
        if np.linalg.norm(x) > 1e6:   # точка улетела слишком далеко - метод разошелся
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged)

def adam(grad, x0, alpha=0.05, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-8, max_iter=100000):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    traj = [x.copy()]
    converged = False
    it = 0
    while it < max_iter:
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            converged = True
            break
        t = it + 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        x = x - alpha * mhat / (np.sqrt(vhat) + eps)
        if np.linalg.norm(x) > 1e6:   # точка улетела слишком далеко - метод разошелся
            break
        traj.append(x.copy())
        it += 1
    return _result(x, traj, it, converged)
