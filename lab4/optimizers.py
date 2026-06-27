"""Методы сопряженных направлений, Ньютона и квазиньютоновские.

Единый интерфейс: метод принимает Problem и стартовую точку x0, возвращает
словарь result = {x, fx, traj, iters, converged, status}. Счетчики nf/ng/nh
берутся из Problem (см. run). Статус остановки: "grad" (норма градиента мала),
"max_iter", "not_pd" (Гессе не положительно определена), "diverged", "ls_fail".

Критерий остановки общий: ||grad f(x)|| <= tol.
"""

import numpy as np
import scipy.linalg as sla
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _result(x, fx, traj, iters, converged, status):
    return {"x": np.asarray(x, float), "fx": float(fx), "traj": np.array(traj),
            "iters": int(iters), "converged": bool(converged), "status": status}


def _fx(prob, x):
    return prob._f(np.asarray(x, float))


def run(method, prob, x0, **kw):
    prob.reset()
    res = method(prob, np.array(x0, dtype=float), **kw)
    res["nf"], res["ng"], res["nh"] = prob.nf, prob.ng, prob.nh
    return res


# ---- общий одномерный поиск: сильные условия Вольфе ----

def line_search_wolfe(prob, x, p, f0, g0, c1=1e-4, c2=0.9, a_max=20.0, maxit=30):
    d0 = float(g0 @ p)
    if d0 >= 0:
        return None  # p не направление спуска
    cache = {}

    def phi(a):
        return prob.f(x + a * p)

    def dphi(a):
        g = prob.grad(x + a * p)
        cache["g"] = g
        return float(g @ p), g

    def zoom(lo, hi, flo):
        a = 0.5 * (lo + hi)
        for _ in range(maxit):
            a = 0.5 * (lo + hi)
            fa = phi(a)
            if fa > f0 + c1 * a * d0 or fa >= flo:
                hi = a
            else:
                da, ga = dphi(a)
                if abs(da) <= -c2 * d0:
                    return a, fa, ga
                if da * (hi - lo) >= 0:
                    hi = lo
                lo, flo = a, fa
        ga = prob.grad(x + a * p)
        return a, phi(a), ga

    a_prev, f_prev = 0.0, f0
    a = 1.0
    for i in range(maxit):
        fa = phi(a)
        if fa > f0 + c1 * a * d0 or (i > 0 and fa >= f_prev):
            return zoom(a_prev, a, f_prev)
        da, ga = dphi(a)
        if abs(da) <= -c2 * d0:
            return a, fa, ga
        if da >= 0:
            return zoom(a, a_prev, fa)
        a_prev, f_prev = a, fa
        a = min(2 * a, a_max)
    ga = prob.grad(x + a * p)
    return a, phi(a), ga


# ---- 1. Линейный метод сопряженных градиентов (для квадратичных) ----
# Использует матрицу A и вектор b квадратичной функции. Каждое умножение
# на A (matvec) считаем за один вызов градиента (это основная операция метода).

def cg_linear(prob, x0, tol=1e-8, max_iter=None):
    A, b = prob.A, prob.b
    n = len(x0)
    if max_iter is None:
        max_iter = 2 * n + 10
    x = np.array(x0, float)
    traj = [x.copy()]
    r = A @ x - b
    prob.ng += 1
    p = -r.copy()
    it = 0
    converged = np.linalg.norm(r) <= tol
    while not converged and it < max_iter:
        Ap = A @ p
        prob.ng += 1
        rr = r @ r
        denom = p @ Ap
        if denom <= 0:
            break
        alpha = rr / denom
        x = x + alpha * p
        r = r + alpha * Ap
        traj.append(x.copy())
        it += 1
        converged = np.linalg.norm(r) <= tol
        beta = (r @ r) / rr
        p = -r + beta * p
    status = "grad" if converged else "max_iter"
    return _result(x, _fx(prob, x), traj, it, converged, status)


# ---- 2-3. Нелинейные CG: Флетчера-Ривса (FR) и Полака-Рибьера (PR) ----

def nonlinear_cg(prob, x0, tol=1e-8, max_iter=10000, variant="FR"):
    x = np.array(x0, float)
    n = len(x)
    traj = [x.copy()]
    g = prob.grad(x)
    fx = prob.f(x)
    if np.linalg.norm(g) <= tol:
        return _result(x, _fx(prob, x), traj, 0, True, "grad")
    p = -g.copy()
    it = 0
    converged = False
    status = "max_iter"
    while it < max_iter:
        if g @ p >= 0:
            p = -g
        ls = line_search_wolfe(prob, x, p, fx, g, c2=0.1)
        if ls is None:
            p = -g
            ls = line_search_wolfe(prob, x, p, fx, g, c2=0.1)
        if ls is None:
            status = "ls_fail"
            break
        alpha, fx, g_new = ls
        step = alpha * p
        x = x + step
        if np.linalg.norm(x) > 1e8:
            status = "diverged"
            break
        traj.append(x.copy())
        it += 1
        if np.linalg.norm(g_new) <= tol:
            converged = True
            status = "grad"
            g = g_new
            break
        if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(x)):
            status = "no_progress"
            break
        if variant == "FR":
            beta = (g_new @ g_new) / (g @ g)
        else:  # PR+
            beta = max(0.0, (g_new @ (g_new - g)) / (g @ g))
        if it % n == 0:  # рестарт каждые n шагов
            beta = 0.0
        p = -g_new + beta * p
        g = g_new
    return _result(x, _fx(prob, x), traj, it, converged, status)


# ---- 4. Метод Ньютона с разложением Холецкого (полный шаг) ----

def newton_cholesky(prob, x0, tol=1e-8, max_iter=200):
    x = np.array(x0, float)
    traj = [x.copy()]
    it = 0
    converged = False
    status = "max_iter"
    while it < max_iter:
        g = prob.grad(x)
        if np.linalg.norm(g) <= tol:
            converged = True
            status = "grad"
            break
        H = prob.hess(x)
        try:
            cf = cho_factor(H)
            p = cho_solve(cf, -g)
        except sla.LinAlgError:
            status = "not_pd"
            break
        x = x + p
        if np.linalg.norm(x) > 1e8:
            status = "diverged"
            break
        traj.append(x.copy())
        it += 1
    return _result(x, _fx(prob, x), traj, it, converged, status)


# ---- 5. Модифицированный Ньютон (выбор направления + одномерный поиск) ----
# Если Гессе не положительно определена или направление не спусковое - откат
# к направлению антиградиента.

def newton_modified(prob, x0, tol=1e-8, max_iter=10000):
    x = np.array(x0, float)
    traj = [x.copy()]
    g = prob.grad(x)
    fx = prob.f(x)
    if np.linalg.norm(g) <= tol:
        return _result(x, _fx(prob, x), traj, 0, True, "grad")
    it = 0
    converged = False
    status = "max_iter"
    while it < max_iter:
        H = prob.hess(x)
        try:
            cf = cho_factor(H)
            p = cho_solve(cf, -g)
        except sla.LinAlgError:
            p = -g
        if g @ p >= 0:
            p = -g
        ls = line_search_wolfe(prob, x, p, fx, g)
        if ls is None:
            p = -g
            ls = line_search_wolfe(prob, x, p, fx, g)
        if ls is None:
            status = "ls_fail"
            break
        alpha, fx, g = ls
        step = alpha * p
        x = x + step
        if np.linalg.norm(x) > 1e8:
            status = "diverged"
            break
        traj.append(x.copy())
        it += 1
        if np.linalg.norm(g) <= tol:
            converged = True
            status = "grad"
            break
        if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(x)):
            status = "no_progress"
            break
    return _result(x, _fx(prob, x), traj, it, converged, status)


# ---- 6. Powell's Dog Leg (доверительная область) ----

def _dogleg_step(g, H, delta):
    gHg = float(g @ (H @ g))
    if gHg <= 0:
        return -delta * g / np.linalg.norm(g)
    pU = -(g @ g) / gHg * g
    try:
        pB = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        pB = None
    if pB is not None and np.linalg.norm(pB) <= delta:
        return pB
    if np.linalg.norm(pU) >= delta or pB is None:
        return delta * pU / np.linalg.norm(pU)
    d = pB - pU
    a = d @ d
    bq = 2 * (pU @ d)
    cq = pU @ pU - delta ** 2
    s = (-bq + np.sqrt(max(bq * bq - 4 * a * cq, 0.0))) / (2 * a)
    return pU + s * d


def dogleg(prob, x0, tol=1e-8, max_iter=10000, delta0=1.0, delta_max=100.0, eta=0.1):
    x = np.array(x0, float)
    traj = [x.copy()]
    delta = delta0
    it = 0
    converged = False
    status = "max_iter"
    while it < max_iter:
        g = prob.grad(x)
        if np.linalg.norm(g) <= tol:
            converged = True
            status = "grad"
            break
        H = prob.hess(x)
        p = _dogleg_step(g, H, delta)
        pred = -(g @ p + 0.5 * p @ (H @ p))
        fx = prob.f(x)
        fnew = prob.f(x + p)
        actual = fx - fnew
        rho = actual / pred if pred > 1e-15 else (1.0 if actual > 0 else -1.0)
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-8:
            delta = min(2 * delta, delta_max)
        if rho > eta:
            x = x + p
            if np.linalg.norm(x) > 1e8:
                status = "diverged"
                break
            traj.append(x.copy())
        it += 1
    return _result(x, _fx(prob, x), traj, it, converged, status)


# ---- 7. Квазиньютоновские методы DFP и BFGS ----

def quasi_newton(prob, x0, tol=1e-8, max_iter=10000, variant="BFGS"):
    x = np.array(x0, float)
    n = len(x)
    traj = [x.copy()]
    Hinv = np.eye(n)
    g = prob.grad(x)
    fx = prob.f(x)
    it = 0
    converged = np.linalg.norm(g) <= tol
    status = "grad" if converged else "max_iter"
    while not converged and it < max_iter:
        p = -Hinv @ g
        if g @ p >= 0:
            Hinv = np.eye(n)
            p = -g
        ls = line_search_wolfe(prob, x, p, fx, g)
        if ls is None:
            status = "ls_fail"
            break
        alpha, fx, g_new = ls
        s = alpha * p
        x_new = x + s
        if np.linalg.norm(x_new) > 1e8:
            status = "diverged"
            break
        traj.append(x_new.copy())
        y = g_new - g
        sy = float(s @ y)
        if sy > 1e-12:
            if variant == "BFGS":
                rho = 1.0 / sy
                I = np.eye(n)
                Hinv = (I - rho * np.outer(s, y)) @ Hinv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            else:  # DFP
                Hy = Hinv @ y
                Hinv = Hinv + np.outer(s, s) / sy - np.outer(Hy, Hy) / (y @ Hy)
        x, g = x_new, g_new
        it += 1
        if np.linalg.norm(g) <= tol:
            converged = True
            status = "grad"
        elif np.linalg.norm(s) <= 1e-12 * (1 + np.linalg.norm(x)):
            status = "no_progress"
            break
    return _result(x, _fx(prob, x), traj, it, converged, status)


# ---- L-BFGS (с ограниченной памятью m) ----

def lbfgs(prob, x0, tol=1e-8, max_iter=10000, m=10):
    x = np.array(x0, float)
    traj = [x.copy()]
    g = prob.grad(x)
    fx = prob.f(x)
    S, Y, RHO = [], [], []
    it = 0
    converged = np.linalg.norm(g) <= tol
    status = "grad" if converged else "max_iter"
    while not converged and it < max_iter:
        q = g.copy()
        alphas = []
        for i in range(len(S) - 1, -1, -1):
            a = RHO[i] * (S[i] @ q)
            alphas.append(a)
            q = q - a * Y[i]
        alphas.reverse()
        gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1]) if S else 1.0
        r = gamma * q
        for i in range(len(S)):
            b = RHO[i] * (Y[i] @ r)
            r = r + (alphas[i] - b) * S[i]
        p = -r
        if g @ p >= 0:
            p = -g
        ls = line_search_wolfe(prob, x, p, fx, g)
        if ls is None:
            status = "ls_fail"
            break
        alpha, fx, g_new = ls
        s = alpha * p
        x_new = x + s
        if np.linalg.norm(x_new) > 1e8:
            status = "diverged"
            break
        traj.append(x_new.copy())
        y = g_new - g
        sy = float(s @ y)
        if sy > 1e-12:
            S.append(s); Y.append(y); RHO.append(1.0 / sy)
            if len(S) > m:
                S.pop(0); Y.pop(0); RHO.pop(0)
        x, g = x_new, g_new
        it += 1
        if np.linalg.norm(g) <= tol:
            converged = True
            status = "grad"
        elif np.linalg.norm(s) <= 1e-12 * (1 + np.linalg.norm(x)):
            status = "no_progress"
            break
    return _result(x, _fx(prob, x), traj, it, converged, status)


# ---- Готовая реализация: scipy Newton-CG ----

def newton_cg_scipy(prob, x0, tol=1e-8, max_iter=1000):
    traj = [np.array(x0, float)]

    def cb(xk):
        traj.append(np.array(xk, float))

    res = minimize(prob.f, np.array(x0, float), jac=prob.grad, hess=prob.hess,
                   method="Newton-CG", callback=cb,
                   options={"xtol": 1e-10, "maxiter": max_iter})
    gnorm = np.linalg.norm(prob._grad(np.asarray(res.x, float)))
    converged = gnorm <= tol
    status = "grad" if converged else f"scipy({res.status})"
    return _result(res.x, _fx(prob, res.x), traj, res.nit, converged, status)
