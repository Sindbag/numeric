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
            y = solver.integral()

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