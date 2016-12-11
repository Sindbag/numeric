import math
import numpy as np


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
        return Bounds.sides(np.min(self.x, r.x), np.max(self.xm, r.xm),
                            np.min(self.y, r.y), np.max(self.ym, r.ym))


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
        self.x = np.min(np.max(self.x, b.x if b.w > 0 else b.xm), b.xm if b.w > 0 else b.x)
        self.y = np.min(np.max(self.y, b.y if b.h > 0 else b.ym), b.ym if b.h > 0 else b.y)
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

    def __init__(self, mode='rectangle'):
        self.mode = mode

    def integral(self, f, x0, f0, steps=100):
        p = Numeric.modes[self.mode]

        def _int(x):
            s = f0
            d = np.abs(x - x0) / steps
            for i in range(steps):
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
                if (j == len(X) - 1):
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

        def f(x):
            L = bisect(x)
            return (pts[L].y * (x - pts[L - 1].x) + pts[L - 1].y * (pts[L].x - x)) / (pts[L].x - pts[L - 1].x)

        if 2 < len(pts):
            n = len(pts)
            a = lambda i: 1 / (pts[i + 0].x - pts[i - 1].x)
            b = lambda i: (2 / (pts[i + 1].x - pts[i].x) if i < n - 1 else 0) + \
                          (2 / (pts[i].x - pts[i - 1].x) if i > 0 else 0)
            c = lambda i: 1 / (pts[i + 1].x - pts[i].x)
            y = [0] * len(pts)
            for i in range(n - 1):
                d = 3 * (pts[i + 1].y - pts[i].y) / math.pow(pts[i + 1].x - pts[i].x, 2)
                y[i + 0] += d
                y[i + 1] += d

            u = self.tridiag(a, b, c, y)
            A = [b(i) if i == j else
                 c(i) if i == (j - 1) else
                 a(i) if i == (j + 1) else
                 0
                 for i, _
                 in enumerate(pts)
                 for j, _
                 in enumerate(pts)]

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
