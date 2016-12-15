import json
import logging
import pandas as pd
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.cache import never_cache

from numeric.forms import NumericForm, NumericFormMulti
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
            tabs = [Point(*map(float, line.split(',')))
                    for line in func.split('\n')]
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
            if not (0 <= B <= 1):
                raise Exception('Hyperparam B must be in range [0; 1]')

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

            x = solver.freeze(solver.interpolate(x_t), 0, T, steps)
            y = solver.freeze(solver.interpolate(y_t), 0, T, steps)

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
            x.tabs = solver.table(x, 0, T, steps)
            s.tabs = solver.table(s, 0, T, steps)
            z_tabs = solver.table(z, 0, T, steps)
            p_tabs = solver.table(dens, 0, 1, steps)

            context['tabs'] = tabs

            context['pics'].append(draw_pic('y(t)', tabs, 'y_t'))

            fig = plt.figure()
            plt.title('x(t)')
            plt.plot([p.x for p in x.tabs], [p.y for p in x.tabs])
            plt.plot([p.x for p in s.tabs], [p.y for p in s.tabs],
                     alpha=0.75, color='red')
            fig.savefig(settings.STATIC_DIR + 'x.png')
            context['pics'].append('/static/x.png')

            context['pics'].append(draw_pic('z(t)', z_tabs, 'z_t'))
            context['pics'].append(draw_pic('p(w)', p_tabs, 'p_w'))
            context['pics'].append(draw_pic('P(y)', prob.tabs, 'prob_y'))

        except BaseException as e:
            logging.exception(e)
            context['error'] = 'Error occurred: %s' % e
    context['form'] = form

    return render(request, 'manual.html', context)


def auto(request):
    context = {}
    form = NumericFormMulti()
    if request.method == 'POST':
        try:
            form = NumericFormMulti(request.POST)
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

            b_st = float(form.cleaned_data['b_start'])
            b_end = float(form.cleaned_data['b_end'])
            b_steps = int(form.cleaned_data['b_step'])
            if not (0 <= b_st <= 1) or not (0 <= b_end <= 1) or b_end < b_st:
                raise Exception('Hyperparam B must be in range [0; 1]. '
                                'B end range must be bigger than B start.')

            context['pics'] = []
            prob = solver.freeze(solver.integral(dens, 0, 0, steps), 0, 1, steps)

            def f(t, x, B=b_st, **kwargs):
                return calc(function, x=x[0], t=t, y=x[1],
                            B=B, T=T, p=dens,
                            U=lambda y: (1-prob(y)),
                            S=s, z=z, **kwargs)

            dz = solver.derivative(z)
            dz = solver.freeze(dz, 0, T, steps)

            glob_minimal = 1E20
            B = -1
            X = None
            Y = None
            C1 = None
            C2 = None
            for _b in range(b_steps):
                b = b_st + (b_end - b_st) *_b / b_steps
                print(b)

                def equation(t, x):
                    return [
                        dz(t) * (1 - prob(x[1])),
                        f(t, x, b)
                    ]

                pts = solver.solve_differential(equation, [x_st, y_st], T, steps)
                x_t = [Point(p.x, p.y[0]) for p in pts]
                y_t = [Point(p.x, p.y[1]) for p in pts]
                x = solver.freeze(solver.interpolate(x_t), 0, T, steps)
                y = solver.freeze(solver.interpolate(y_t), 0, T, steps)

                dx = solver.derivative(x)
                wp = lambda w: w * dens(w)
                i_wp = solver.freeze(solver.integral(wp, 0, 0, steps), 0, T, steps)
                _f = lambda t: dx(t) * (i_wp(1) - i_wp(y(t)))
                i_f = solver.integral(_f, 0, 0)
                c1 = lambda B: 1 - (i_f(T) - i_f(0)) / (x(T) - x_st)
                c2 = lambda B: abs(x(T) - s(T)) / s(T)
                minim = c1(b) + 10 * c2(b)
                print(minim)

                if minim < glob_minimal:
                    glob_minimal = minim
                    B = b
                    C1 = c1
                    C2 = c2
                    X = x
                    Y = y

            context['b'] = B
            context['c1'] = C1(B)
            context['c2'] = C2(B)
            context['minim'] = C1(B) + 10 * C2(B)

            tabs = solver.table(Y, 0, T, steps)
            prob.tabs = solver.table(prob, 0, 1, steps)
            X.tabs = solver.table(X, 0, T, steps)
            s.tabs = solver.table(s, 0, T, steps)
            z_tabs = solver.table(z, 0, T, steps)
            p_tabs = solver.table(dens, 0, 1, steps)

            context['tabs'] = tabs

            context['pics'].append(draw_pic('y(t)', tabs, 'y_t'))

            fig = plt.figure()
            plt.title('x(t)')
            plt.plot([p.x for p in X.tabs], [p.y for p in X.tabs])
            plt.plot([p.x for p in s.tabs], [p.y for p in s.tabs], alpha=0.75, color='red')
            fig.savefig(settings.STATIC_DIR + 'x.png')
            context['pics'].append('/static/x.png')

            context['pics'].append(draw_pic('z(t)', z_tabs, 'z_t'))
            context['pics'].append(draw_pic('p(w)', p_tabs, 'p_w'))
            context['pics'].append(draw_pic('P(y)', prob.tabs, 'prob_y'))

        except BaseException as e:
            logging.exception(e)
            context['error'] = 'Error occurred: %s' % e
    context['form'] = form

    return render(request, 'automatic.html', context)