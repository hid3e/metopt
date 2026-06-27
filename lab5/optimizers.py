"""Оптимизаторы для задачи регрессии.

Все итерационные методы возвращают словарь:
  {w, loss_hist, risk_hist, reg_hist, ng, status, ...}.
loss_hist/risk_hist/reg_hist - история полного loss, эмпирического риска Q
и регуляризационного слагаемого (по эпохам или итерациям).

Модель линейна по w, поэтому Гаусс-Ньютон для квадратичного риска совпадает
с методом Ньютона (сходится за 1-2 шага); Левенберг-Марквардт добавляет
регуляризацию (mu I) и правило ее изменения.
"""

import numpy as np


# ---- 1. Аналитическое решение ----

def analytic_means(x, y):
    """Линейная регрессия y = a*x + b через оценки средних (одномерный случай).

    Возвращает (a, b). Это точное решение МНК для степени 1.
    """
    xm, ym = x.mean(), y.mean()
    a = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    b = ym - a * xm
    return a, b


def analytic(task):
    """Нормальные уравнения (точное решение МНК, при l2 - гребневая регрессия).

    Для степени 1 совпадает с analytic_means. L1 аналитически не решается.
    """
    Phi, y = task.Phi, task.y
    n = Phi.shape[1]
    R = task.l2 * task.m * np.eye(n)   # масштаб согласован с risk = (1/m)||.||^2
    R[0, 0] = 0.0                      # w0 не регуляризуем
    w = np.linalg.solve(Phi.T @ Phi + R, Phi.T @ y)
    return {"w": w, "loss_hist": [task.loss(w)], "risk_hist": [task.risk(w)],
            "reg_hist": [task.reg(w)], "ng": 0, "status": "analytic"}


# ---- 2-3. SGD и mini-batch ----

def minibatch_gd(task, w0, epochs=100, batch=16, lr=0.05, seed=0):
    """Mini-batch градиентный спуск. batch=1 -> чистый SGD.

    История loss/risk/reg снимается после каждой эпохи; ng_hist - накопленное
    число вычислений градиента (для сравнения по вычислительной стоимости).
    """
    rng = np.random.default_rng(seed)
    w = np.array(w0, float)
    m = task.m
    loss_h = [task.loss(w)]; risk_h = [task.risk(w)]; reg_h = [task.reg(w)]; ng_h = [0]
    status = "epochs"
    for ep in range(epochs):
        order = rng.permutation(m)
        for s in range(0, m, batch):
            idx = order[s:s + batch]
            w = w - lr * task.grad_batch(w, idx)
        loss_h.append(task.loss(w)); risk_h.append(task.risk(w))
        reg_h.append(task.reg(w)); ng_h.append(task.ng)
        if not np.isfinite(loss_h[-1]) or loss_h[-1] > 1e8:
            status = "diverged"; break
    return {"w": w, "loss_hist": loss_h, "risk_hist": risk_h, "reg_hist": reg_h,
            "ng_hist": ng_h, "epochs": ep + 1, "batch": batch, "lr": lr,
            "ng": task.ng, "status": status}


def sgd(task, w0, epochs=100, lr=0.05, seed=0):
    """Стохастический градиентный спуск (mini-batch с batch=1)."""
    r = minibatch_gd(task, w0, epochs=epochs, batch=1, lr=lr, seed=seed)
    r["status"] = "epochs"
    return r


# ---- 4-5. Гаусс-Ньютон и Левенберг-Марквардт ----

def _gn_grad_hess(task, w):
    """Градиент loss и приближение Гессе по Гауссу-Ньютону: (2/m) J^T J + reg."""
    J = task.Phi                       # якобиан невязок (модель линейна по w)
    g = task.grad(w)
    B = (2.0 / task.m) * (J.T @ J) + np.diag(task.reg_hess_diag(w))
    return g, B


def gauss_newton(task, w0, max_iter=50, tol=1e-8):
    w = np.array(w0, float)
    loss_h = [task.loss(w)]; risk_h = [task.risk(w)]; reg_h = [task.reg(w)]
    status = "max_iter"
    for it in range(max_iter):
        g, B = _gn_grad_hess(task, w)
        if np.linalg.norm(g) <= tol:
            status = "grad"; break
        try:
            dw = np.linalg.solve(B, -g)
        except np.linalg.LinAlgError:
            dw = -g
        w = w + dw
        loss_h.append(task.loss(w)); risk_h.append(task.risk(w)); reg_h.append(task.reg(w))
        if np.linalg.norm(dw) <= tol * (1 + np.linalg.norm(w)):
            status = "grad"; break
    return {"w": w, "loss_hist": loss_h, "risk_hist": risk_h, "reg_hist": reg_h,
            "iters": len(loss_h) - 1, "ng": task.ng, "status": status}


def levenberg_marquardt(task, w0, max_iter=100, mu0=1e-2, tol=1e-8):
    """Гаусс-Ньютон с регуляризацией (B + mu I). Правило: успешный шаг -
    принять и уменьшить mu; неудачный - отвергнуть и увеличить mu.
    """
    w = np.array(w0, float)
    mu = mu0
    cur = task.loss(w)
    loss_h = [cur]; risk_h = [task.risk(w)]; reg_h = [task.reg(w)]
    status = "max_iter"
    n = len(w)
    for it in range(max_iter):
        g, B = _gn_grad_hess(task, w)
        if np.linalg.norm(g) <= tol:
            status = "grad"; break
        try:
            dw = np.linalg.solve(B + mu * np.eye(n), -g)
        except np.linalg.LinAlgError:
            dw = -g / (1 + mu)
        wn = w + dw
        new = task.loss(wn)
        if new < cur:                  # удачный шаг
            w, cur = wn, new
            mu = max(mu * 0.5, 1e-12)
        else:                          # неудачный шаг
            mu = mu * 2.0
        loss_h.append(cur); risk_h.append(task.risk(w)); reg_h.append(task.reg(w))
    return {"w": w, "loss_hist": loss_h, "risk_hist": risk_h, "reg_hist": reg_h,
            "iters": len(loss_h) - 1, "ng": task.ng, "status": status, "mu_final": mu}
