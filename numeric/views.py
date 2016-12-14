import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.cache import never_cache

from numeric.forms import NumericForm
from numeric.numeric import Numeric, Point, calc

logger = logging.getLogger('Solver')
solver = Numeric()

clamp = lambda f, x, y: max(x, min(y, f))


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
        logging.warning(e)
        logging.info('Could not open as CSV file, try to parse as a text')
        try:
            tabs = [Point(float(x.strip()), float(y.strip()))
                    for line in func.split('\n')
                    for x, y in line.split(',')]
            ff = solver.interpolate(tabs)
        except Exception as e:
            logging.warning(e)
            logging.info('Could not parse as text, try to parse as a function')
            f = lambda x: calc(func, **{argname: x})
            ff = solver.freeze(f, 0, max, steps)
    return ff


def draw_pic(name, tabs, filename):
    fig = plt.figure()
    plt.title(name)
    plt.plot([p.x for p in tabs], [p.y for p in tabs])
    fig.savefig(settings.STATIC_DIR + '%s.png' % filename)
    return '/static/%s.png' % filename


@never_cache
def manual(request):
    context = {}
    form = NumericForm()
    if request.method == 'POST':
        try:
            form = NumericForm(request.POST)
            if not form.is_valid():
                raise Exception('Form is not valid')

            function = form.cleaned_data['f']

            steps = int(form.cleaned_data['steps'])
            T = int(form.cleaned_data['T'])
            dens = tabulate(form.cleaned_data['density'], 'w', 1, steps)
            s = tabulate(form.cleaned_data['S'], 't', T, steps)
            z = tabulate(form.cleaned_data['z'], 't', T, steps)
            x_st = s(0)
            y_st = float(form.cleaned_data['y_start'])

            B = float(form.cleaned_data['B'])

            context['pics'] = []
            prob = solver.freeze(solver.integral(dens, 0, 0, steps), 0, 1, steps)

            def f(t, x, **kwargs):
                return calc(function, x=x[0], t=t, y=x[1],
                            B=B, T=T, p=dens,
                            U=lambda y: (1-prob(y)),
                            S=s, z=z, **kwargs)

            dz = solver.derivative(z)
            dz = solver.freeze(dz, 0, T, steps)

            def equation(t, x):
                return [
                    dz(t) * (1 - prob(x[1])),
                    f(t, x)
                ]

            pts = solver.solve_differential(equation, [x_st, y_st], T, steps)
            x_t = [Point(p.x, p.y[0]) for p in pts]
            y_t = [Point(p.x, p.y[1]) for p in pts]

            x = solver.interpolate(x_t)
            y = solver.interpolate(y_t)

            dx = solver.derivative(x)
            wp = lambda w: w * dens(w)
            i_wp = solver.integral(wp, 0, 0, steps)
            _f = lambda t: dx(t) * (i_wp(1) - i_wp(y(t)))
            i_f = solver.integral(_f, 0, _f(0))
            C1 = lambda B: 1 - (i_f(T) - i_f(0)) / (x(T) - x_st)
            C2 = lambda B: abs(x(T) - s(T)) / s(T)

            context['c1'] = C1(B)
            context['c2'] = C2(B)
            context['minim'] = C1(B) + 10 * C2(B)

            tabs = solver.table(y, 0, T, steps)
            prob.tabs = solver.table(prob, 0, 1, steps)
            context['tabs'] = tabs

            fig = plt.figure()
            plt.title('y(t)')
            plt.plot([p.x for p in tabs], [p.y for p in tabs])
            fig.savefig(settings.STATIC_DIR + 'somefig.png')
            context['pics'].append('/static/somefig.png')

            x.tabs = solver.table(x, 0, T, steps)
            s.tabs = solver.table(s, 0, T, steps)
            fig = plt.figure()
            plt.title('x(t)')
            plt.plot([p.x for p in x.tabs], [p.y for p in x.tabs])
            plt.plot([p.x for p in s.tabs], [p.y for p in s.tabs], alpha=0.75, color='red')
            fig.savefig(settings.STATIC_DIR + 'x.png')
            context['pics'].append('/static/x.png')

            # fig = plt.figure()
            # plt.title('S(t)')
            # plt.plot([p.x for p in s_tabs], [p.y for p in s_tabs])
            # fig.savefig(settings.STATIC_DIR + 's.png')
            # context['pics'].append('/static/s.png')

            fig = plt.figure()
            plt.title('z(t)')
            z_tabs = solver.table(z, 0, T, steps)
            plt.plot([p.x for p in z_tabs], [p.y for p in z_tabs])
            fig.savefig(settings.STATIC_DIR + 'z.png')
            context['pics'].append('/static/z.png')

            fig = plt.figure()
            plt.title('p(w)')
            p_tabs = solver.table(dens, 0, 1, steps)
            plt.plot([l.x for l in p_tabs], [l.y for l in p_tabs])
            fig.savefig(settings.STATIC_DIR + 'dens.png')
            context['pics'].append('/static/dens.png')

            fig = plt.figure()
            plt.title('P(y)')
            plt.plot([p.x for p in prob.tabs], [p.y for p in prob.tabs])
            fig.savefig(settings.STATIC_DIR + 'prob.png')
            context['pics'].append('/static/prob.png')

        except BaseException as e:
            logging.exception(e)
            context['error'] = 'Error occurred: %s' % e
    context['form'] = form

    return render(request, 'manual.html', context)


def auto(request):
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

            dz = solver.derivative(z)
            U = solver.integral(p, 0, 0, 100)
            y = solver.prepare(U, p, s, z, function)
            eq = lambda t, x: dz(t) * (1 - U(y(t, x)))
            x = solver.solve_differential(eq, 0)

            b_start = float(form.cleaned_data['b_start'])
            b_end = float(form.cleaned_data['b_end'])
            steps = int(form.cleaned_data['steps'])

            f_b = lambda b: lambda x: calc(
                function,
                t=x,
                x=lambda t: t ** 2,
                S=s,
                z=z,
                B=b,
            )

            for b in range(b_start, b_end, 1):
                f = f_b(b)
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
            s_tabs = solver.table(s, 0, T, steps)
            plt.plot([p.x for p in s_tabs], [p.y for p in s_tabs])
            fig.savefig(settings.STATIC_DIR + 's.png')
            context['pics'].append('/static/s.png')

            fig = plt.figure()
            plt.title('z(t)')
            z_tabs = solver.table(z, 0, T, steps)
            plt.plot([p.x for p in z_tabs], [p.y for p in z_tabs])
            fig.savefig(settings.STATIC_DIR + 'z.png')
            context['pics'].append('/static/z.png')

            fig = plt.figure()
            plt.title('p(w)')
            p_tabs = solver.table(p, 0, T, steps)
            plt.plot([l.x for l in p_tabs], [l.y for l in p_tabs])
            fig.savefig(settings.STATIC_DIR + 'dens.png')
            context['pics'].append('/static/dens.png')

            fig = plt.figure()
            plt.title('P')
            q = solver.table(solver.integral(p, 0, 0, 100), 0, 1, 100)
            plt.plot([p.x for p in q], [p.y for p in q])
            fig.savefig(settings.STATIC_DIR + 'prob.png')
            context['pics'].append('/static/prob.png')

        except BaseException as e:
            logging.exception(e)
            context['error'] = 'Error occurred: %s' % e
    context['form'] = form

    return render(request, 'manual.html', context)