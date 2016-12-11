import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render
from numeric.forms import NumericForm
from numeric.numeric import Numeric, Point

logger = logging.getLogger('Solver')


class Solver(object):
    _id = 0

    def __init__(self, density, s, z, x_start, y_start, function,
                 method='manual', b=None, b_start=None, b_end=None):
        self.id = Solver._id
        Solver._id += 1
        self.density = density
        self.s = s
        self.z = z
        self.x_start = x_start
        self.y_start = y_start
        self.function = function

        if method == 'manual':
            self.B = b
        else:
            self.b_start = b_start
            self.b_end = b_end

        self.x = []
        self.y = []

        self.tabs_s = []
        self.tabs_z = []
        self.tabs_prob_distrib = []
        self.tabs_prob_density = []

        self.interpolated = None
        self.integral = []
        self.method = method

    def calculate(self, range):
        for t in range:
            s_val = eval(
                self.s, {
                    '__builtins__': None,
                    't': t
                }
            )
            z_val = eval(
                self.z, {
                    '__builtins__': None,
                    't': t
                }
            )
            density_val = eval(
                self.density, {
                    '__builtins__': None,
                    'w': t
                }
            )

            self.tabs_s.append((t, s_val))
            self.tabs_z.append((t, z_val))
            self.tabs_prob_density.append((t, density_val))
            self.tabs_prob_distrib.append(
                (t, sum([density_val] + [val for x, val in self.tabs_prob_distrib]))
            )
        self.integral = self.tab_integrate(self.function, 1, 100, 1)
        self.interpolated = self.interpolate(self.integral)

        if self.method == 'manual':
            self.solve_cauchy(0.25)
        else:
            pass

    def tab_integrate(self, function, start, end, step):
        integral = []

        for i in np.arange(start, end, step):
            integral.append(
                (i, self.numeric_integration(function, start, i, step))
            )

        return integral

    @staticmethod
    def numeric_integration(function, start, end, step=100):
        result = 0
        for i in np.linspace(start, end, num=step):
            result += eval(function,
                           {'__builtins__': None,
                            'x': i}
                           )

        return (end - start) / step * result

    @staticmethod
    def interpolate(integral):
        return '17*y'

    def save(self, filename):
        with open(filename, 'w') as file:
            # file.write(str(self.json))
            file.write(json.dumps(self.json))

    @property
    def json(self):
        if self.method == 'manual':
            return {
                'tabs_s': [(float(x), float(y)) for x, y in self.tabs_s],
                'tabs_z': [(float(x), float(y)) for x, y in self.tabs_z],
                'B': self.B,
                'tabs_integral': [(float(x), float(y)) for x, y in self.integral],
                'tabs_prob_distrib': [(float(x), float(y)) for x, y in self.tabs_prob_distrib],
                'interpolated': self.interpolated,
                'x': self.x,
                'y': self.y,
                'method': self.method
            }
        else:
            return {
                'tabs_s': [(float(x), float(y)) for x, y in self.tabs_s],
                'tabs_z': [(float(x), float(y)) for x, y in self.tabs_z],
                'b_start': self.b_start,
                'b_end': self.b_end,
                'tabs_integral': [(float(x), float(y)) for x, y in self.integral],
                'tabs_prob_density': [(float(x), float(y)) for x, y in self.tabs_prob_density],
                'interpolated': self.interpolated,
                'method': self.method
            }

    def solve_cauchy(self, step=0.25):
        x = [self.x_start] + [0 for i in range(499)]
        y = [self.y_start] + [0 for i in range(499)]
        u = '17 * y'

        for i in range(len(self.tabs_z)):
            x[i] = self.tabs_z[i - 1][1] * eval(u, {'y': y[i - 1]}) * step + x[i - 1]
            y[i] = eval(self.function, {
                'x': x[i - 1],
                'z': self.tabs_z[i - 1][1],
                'S': self.tabs_s[i - 1][1],
                'B': self.B}) * step + y[i - 1]

        self.x = x
        self.y = y

solver = Numeric()


def calc(f, *args, **kwargs):
    kwargs['__builtins__'] = None
    return eval(f, kwargs)


def tabulate(func, argname, max=1, steps=100):
    try:
        logging.info('Trying to parse as CSV file link')
        t = pd.read_csv(func)
        columns = t.columns

        if len(columns) != 2:
            raise Exception('Wrong columns amount: %s', columns)

        tabs = [Point(float(x.strip()), float(y.strip()))
                for x, y in t.values()]
        ff = solver.interpolate(tabs)
    except Exception as e:
        logging.exception(e)
        logging.info('Could not open as CSV file, try to parse as a text')
        try:
            tabs = [Point(float(x.strip()), float(y.strip()))
                    for line in func.split('\n')
                    for x, y in line.split(',')]
            ff = solver.interpolate(tabs)
        except Exception as e:
            logging.exception(e)
            logging.info('Could not parse as text, try to parse as a function')
            f = lambda x: calc(func, **{argname: x})
            ff = solver.freeze(f, 0, max, 100)
    return ff


def manual(request):
    context = {}
    form = NumericForm()
    if request.method == 'POST':
        try:
            form = NumericForm(request.POST)
            if not form.is_valid():
                raise Exception('Form is not valid')

            function = form.cleaned_data['f']

            T = int(form.cleaned_data['T'])
            p = tabulate(form.cleaned_data['density'], 'w', 1)
            s = tabulate(form.cleaned_data['S'], 't', T)
            z = tabulate(form.cleaned_data['z'], 't', T)

            B = float(form.cleaned_data['B'])
            steps = int(form.cleaned_data['steps'])

            f_b = lambda b: lambda x: calc(
                function,
                t=x,
                x=lambda t: t ** 2,
                S=s,
                z=z,
                B=b,
            )
            f = f_b(B)

            tabs = solver.table(f, 0, T, n=steps)
            context['tabs'] = tabs

            context['pics'] = []

            fig = plt.figure()
            plt.title('f')
            plt.plot([p.x for p in tabs], [p.y for p in tabs])
            fig.savefig(settings.STATIC_DIR + 'somefig.png')
            context['pics'].append('/static/somefig.png')

            fig = plt.figure()
            plt.title('S(t)')
            plt.plot([p.x for p in s], [p.y for p in s])
            fig.savefig(settings.STATIC_DIR + 's.png')
            context['pics'].append('/static/s.png')

            fig = plt.figure()
            plt.title('z(t)')
            plt.plot([p.x for p in z], [p.y for p in z])
            fig.savefig(settings.STATIC_DIR + 'z.png')
            context['pics'].append('/static/z.png')

            fig = plt.figure()
            plt.title('p(w)')
            plt.plot([l.x for l in p], [l.y for l in p])
            fig.savefig(settings.STATIC_DIR + 'dens.png')
            context['pics'].append('/static/dens.png')

            fig = plt.figure()
            plt.title('P')
            q = solver.table(solver.integral(solver.interpolate(p), 0, 0, 100), 0, 1, 100)
            plt.plot([p.x for p in q], [p.y for p in q])
            fig.savefig(settings.STATIC_DIR + 'prob.png')
            context['pics'].append('/static/prob.png')

        except BaseException as e:
            logging.exception(e)
            context['error'] = 'Error occurred: %s' % e
    context['form'] = form

    return render(request, 'manual.html', context)


def auto(request):
    try:
        context = {}
        if request.method == 'POST':
            density = request.POST.get('density')
            s = request.POST.get('S')
            z = request.POST.get('z')
            x_start = int(request.POST.get('x_start'))
            y_start = int(request.POST.get('y_start'))
            b_start = int(request.POST.get('b_start'))
            b_end = int(request.POST.get('b_end'))
            r_start = int(request.POST.get('r_start'))
            r_end = int(request.POST.get('r_end'))
            steps = int(request.POST.get('steps'))
            function = request.POST.get('function')

            f_b = lambda x, b: eval(
                function, {
                    '__builtins__': None,
                    'x': x,
                    'S': s,
                    'z': z,
                    'b': b
                }
            )

            calc = Solver(density, s, z, x_start, y_start, function, 'auto', b_start, b_end)
            calc.calculate(np.arange(0, 100, 0.25))
            calc.save('auto/report_%s.txt' % calc.id)

            context['calc'] = calc.json

        return render(request, 'automatic.html', context)

    except Exception as e:
        logging.exception(e)
        return render(
            request,
            'automatic.html',
            context={'error': 'Error occurred: %s' % e}
        )
