import math
import numpy as np
np.seterr('raise')


def calc(f, *args, **kwargs):
    kwargs['math'] = math
    # kwargs['np'] = np
    kwargs['clamp'] = lambda q, x, y: max(x, min(y, q))
    kwargs['solver'] = Numeric()
    return eval(f, kwargs)


class Bounds(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def xm(self):
        return self.x + self.w

    @property
    def ym(self):
        return self.y + self.h

    def union(self, r):
        return Bounds.sides(min(self.x, r.x), max(self.xm, r.xm),
                            min(self.y, r.y), max(self.ym, r.ym))


Bounds.sides = lambda x, xm, y, ym: Bounds(x, y, xm - x, ym - y)
Bounds.empty = Bounds(None, None, None, None)
Bounds.empty.union = lambda r: r


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '({:.2f}, {:.2f})'.format(self.x, self.y)

    def clip_to(self, b):
        self.x = min(max(self.x, b.x if b.w > 0 else b.xm), b.xm if b.w > 0 else b.x)
        self.y = min(max(self.y, b.y if b.h > 0 else b.ym), b.ym if b.h > 0 else b.y)
        return self

    def scale_to(self, b):
        return Point(self.x * b.w + b.x, self.y * b.h + b.y)

    def scale_from(self, b):
        return Point((self.x - b.x) / b.w, (self.y - b.y) / b.h)


class Numeric(object):
    modes = {
        'rectangle': lambda f, x, d: f(x + d / 2) * d,
        'triangle': lambda f, x, d: (f(x) + f(x + d)) * d / 2,
        'simpson': lambda f, x, d: (f(x) + 4 * f(x + d / 2) + f(x + d)) * d / 6,
    }

    def __init__(self, mode='simpson'):
        self.mode = mode

    def integral(self, f, x0, f0, steps=100):
        p = Numeric.modes[self.mode]

        def _int(x):
            s = f0
            d = abs(x - x0) / steps
            for i in range(0, steps):
                s = s + p(f, x0 + d * i, d)
            return s

        return _int

    def derivative(self, f, step=0.01):
        return lambda x: (f(x + step) - f(x - step)) / (2 * step)

    def table(self, f, a, b, n=100):
        table = []
        for i in range(n + 1):
            x = a + (b - a) * (i / n)
            table.append(Point(x, f(x)))
        return table

    def freeze(self, f, a, b, n=100):
        return self.interpolate(self.table(f, a, b, n))

    def invert(self, f, df=None, steps=100):
        if df is None:
            df = self.derivative(f)

        def _inv(x0):
            x = x0
            for i in range(steps):
                x -= (f(x) - x0) / df(x)
            return x
        return _inv

    def inv(self, x_mat, LU=False):
        X = x_mat
        E = [1 if i == j else 0
             for i, row
             in enumerate(X)
             for j, cell
             in enumerate(row)]
        U = [cell
             for i, row
             in enumerate(X)
             for j, cell
             in enumerate(row)]

        for i in range(len(X)):
            j = i
            while not U[j][i]:
                if j == len(X) - 1:
                    return None

            if i != j:
                a = E[i]
                b = U[i]
                E[i] = E[j]
                U[i] = U[j]
                E[j] = a
                U[j] = b

            k = i + 1 if LU else 0
            while k < len(X):
                mod = ((k == i) - U[k][i]) / U[i][i]
                for h in range(len(X)):
                    E[k][h] += mod * E[i][h]
                    U[k][h] += mod * U[i][h]
                k += 1

        return U if LU else E

    def det(self, X):
        U = self.inv(X, True)
        r = 1
        if not U:
            return 0
        for i in range(len(X)):
            r *= U[i][i]
        return r

    def dot(self, X, y):
        if isinstance(X[0], list):
            return [self.dot(row, y)
                    for row
                    in X]
        r = 0
        for j in range(len(y)):
            r += X[j] * y[j]
        return r

    def linsolve(self, A, y):
        for i in range(len(A)):
            for j in range(len(A[i])):
                if i != j and i != j - 1 and i != j + 1 and A[i][j] != 0:
                    return self.dot(self.inv(A), y)

        return self.tridiag(
            lambda i: A[i][i - 1],
            lambda i: A[i][i],
            lambda i: A[i][i + 1],
            y)

    def tridiag(self, a, b, c, y):
        n = len(y)

        if n == 0:
            return list()

        if n == 1:
            return list(y[0] / b(0))

        c_ = [0] * n
        y_ = [0] * n
        c_[0] = c(0) / b(0)
        y_[0] = y[0] / b(0)
        for i in range(1, n):
            denom = b(i) - a(i) * c_[i - 1]
            if i != n - 1:
                c_[i] = c(i) / denom
            y_[i] = (y[i] - a(i) * y_[i - 1]) / denom

        n -= 1
        while n:
            y_[n - 1] -= c_[n - 1] * y_[n]
            n -= 1
        return y_

    def interpolate(self, pts):
        def bisect(x):
            L = 0
            R = len(pts)
            while L != R:
                M = (L + R) >> 1
                if x < pts[M].x:
                    R = M
                else:
                    L = M + 1
            if L is 0:
                L += 1
            if L == len(pts):
                L -= 1
            return L

        n = len(pts)
        a = lambda i: 1 / (pts[i + 0].x - pts[i - 1].x)
        b = lambda i: (2 / (pts[i + 1].x - pts[i].x) if i < n - 1 else 0) + \
                      (2 / (pts[i].x - pts[i - 1].x) if i > 0 else 0)
        c = lambda i: 1 / (pts[i + 1].x - pts[i].x)
        y = [0] * len(pts)

        for i in range(n - 1):
            d = 3 * (pts[i + 1].y - pts[i].y) / math.pow(pts[i + 1].x - pts[i].x, 2)
            y[i] += d
            y[i + 1] += d

        u = self.tridiag(a, b, c, y)

        def f(x):
            L = bisect(x)
            dx = pts[L].x - pts[L - 1].x
            dy = pts[L].y - pts[L - 1].y
            t = (x - pts[L - 1].x) / dx
            a = u[L - 1] * dx - dy
            b = dy - u[L] * dx
            return (1 - t) * pts[L - 1].y + t * pts[L].y + t * (1 - t) * (a * (1 - t) + b * t)

        f.points = pts
        return f

    def solve(self, f, df, steps=100):
        def solutioner(x, y, n=steps):
            q = x
            for _ in np.arange(0, n):
                q -= (f(q) - y) / df(q)
            return q

        return solutioner

    def prepare(self, f, df, s, z, function, **kwargs):
        U = self.solve(f, df)
        u = self.freeze(lambda x: U(0.5, x), 0, 1)

        def _f(t, t2):
            return calc(function, t=t, x=t2, u=u, U=f, p=df, S=s, z=z, **kwargs)

        return _f, u

    def solve_differential(self, f, x0, max_t, steps=100):
        if isinstance(x0, list):
            def mat(x, c, m):
                return [e + c[i] * m for i, e in enumerate(x)]
        else:
            def mat(x, c, m):
                return x + c * m

        pts = [Point(0, x0)]
        step = max_t / float(steps)
        for i in range(0, steps):
            t1 = i * step
            x1 = pts[len(pts) - 1].y
            k1 = f(t1, x1)
            k2 = f(t1 + step / 2, mat(x1, k1, step / 2))
            m2 = mat(k1, k2, 2)
            k3 = f(t1 + step / 2, mat(x1, k2, step / 2))
            m3 = mat(m2, k3, 2)
            k4 = f(t1 + step, mat(x1, k3, step))
            m4 = mat(m3, k4, 1)
            x, y = mat(x1, m4, step/6)
            pts.append(Point((i + 1) * step, [x, np.clip(y, 0, 1)]))

        return pts
