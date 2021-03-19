import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from numpy import diff

import matplotlib
import matplotlib.font_manager
from itertools import cycle

DEBUG = True
LATEX_FLAG = False

LINEWIDTH = 5
AXIS_LABEL_FONT_SIZE = 40 # 25
TICK_SIZE = 35 # 30
LEGEND_LENGTH = 4
LEGEND_WIDTH = 4
LEGEND_TEXT_SIZE = 35 # 30

if(LATEX_FLAG):
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rcParams['xtick.labelsize'] = TICK_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_SIZE
matplotlib.rcParams['legend.handlelength'] = LEGEND_LENGTH

def shrink_arr(arr):
    return arr[1:]

def get_array_der(arr, xs, delta):
    der = diff(arr)/delta
    xs = xs[1:]
    return (xs, der)

def get_func_der(f, xs, delta):
    f_val = f(xs)
    f_der = diff(f_val)/delta
    xs = xs[1:]
    return (xs, f_der)

# Equation of the linear normalization:
#   f_{x0}(x) = (f(x)-f(x0)) / |(f'(x0))| + x0
def get_relative_func(f, xs, delta, x0):
    (xs, f_der_array) = get_func_der(f, xs, delta)
    x0_index = get_array_index_by_value(xs,x0)
    f_der = f_der_array[x0_index]
    f_x = f(xs)
    f_x0 = f(x0)
    f_relative = (f_x - f_x0)/np.abs(f_der) + x0

    if(DEBUG):
        f_relative_2 = []
        for i in range(len(f_x)):
            f_relative_2.append((f_x[i] - f_x0)/np.abs(f_der) + x0)

        assert len(f_relative_2) == len(f_relative)

        for i in range(len(f_relative)):
            assert f_relative_2[i] == f_relative[i]

    print("x0: "+str(x0))
    print("f_x len: "+str(len(f_x)))
    print("f_x0: "+str(f_x0))
    print("f_der: "+str(f_der))
    print("f_relative len: "+str(len(f_relative)))
    return (xs, f_relative)

def test_derivatives(f, delta):
    xs2 = np.arange(0., 10, delta)
    (xs2, f_der) = get_func_der(f, xs2, delta)
    (xs2, f_der_2) = get_array_der(f_der, xs2, delta)
    f_der = shrink_arr(f_der)
    (xs2, f_der_3) = get_array_der(f_der_2, xs2, delta)
    f_der = shrink_arr(f_der)
    f_der_2 = shrink_arr(f_der_2)
    # (xs2, f_der_4) = get_array_der(f_der_3, xs2, delta)
    # f_der = shrink_arr(f_der)
    # f_der_2 = shrink_arr(f_der_2)
    # f_der_3 = shrink_arr(f_der_3)

    plt.plot(xs2, f(xs2), 'b', label="f")
    plt.plot(xs2, f_der, 'g', label="f der")
    plt.plot(xs2, f_der_2, 'r', label="f der 2")
    plt.plot(xs2, f_der_3, 'y', label="f der 3")
    # plt.plot(xs2, f_der_4, 'k', label="f_der_4")
    plt.legend(loc="upper left")
    plt.show()

def get_array_index_by_value(array, val):
    return min(range(len(array)), key=lambda i: abs(array[i]-val))

def get_cycle_colors():
    # colors_list = list(c[0] for c in mcolors.TABLEAU_COLORS.items())
    colors_list = 'brgycm'
    return cycle(colors_list)

x0 = 0.3
# x0 = 0.7
delta = 0.0001

# xs = np.arange(0.0001, 1.4 - 0.0001, delta)
xs = np.arange(0.0001, 1 - 0.0001, delta)

x0_index = get_array_index_by_value(xs,x0)
colors = get_cycle_colors() # get plot colors

fisherSqrt = lambda x: np.sqrt(x*(1-x))
fisher = lambda x: x*(1-x)
entropy = lambda x: -x*np.log(x)-(1-x)*np.log(1-x)
minError = lambda x: np.array([x,1-x]).min(0)

(_, fisherSqrt_relative) = get_relative_func(fisherSqrt,xs,delta,x0)
(_, fisher_relative) = get_relative_func(fisher,xs,delta,x0)
(_, entropy_relative) = get_relative_func(entropy,xs,delta,x0)
(xs, minError_relative) = get_relative_func(minError,xs,delta,x0)

y_at_x0 = fisherSqrt_relative[x0_index]
min_y = min(fisher_relative[0], 
            fisherSqrt_relative[0], 
            entropy_relative[0], 
            minError_relative[0])

figure(num=None, figsize=(14, 14), dpi=80)
plt.plot(xs, fisher_relative, next(colors), label="Fisher Normalized", linewidth=LINEWIDTH)
plt.plot(xs, fisherSqrt_relative, next(colors), label="Root-Fisher Normalized", linewidth=LINEWIDTH)
plt.plot(xs, entropy_relative, next(colors), label="Binary Entropy Normalized", linewidth=LINEWIDTH)
plt.plot(xs, minError_relative, next(colors), label="Error Probability Normalized", linewidth=LINEWIDTH)

# polyFunc = lambda a,x: np.power(x,a)
# (_, linear_relative) = get_relative_func(lambda x: polyFunc(1,x),xs,delta,x0)
# (_, quadratic_relative) = get_relative_func(lambda x: polyFunc(2,x),xs,delta,x0)
# (xs, quartic_relative) = get_relative_func(lambda x: polyFunc(4,x),xs,delta,x0)
# y_at_x0 = quadratic_relative[x0_index]
# min_y = min(linear_relative[0], 
#             quadratic_relative[0], 
#             quartic_relative[0])
# plt.plot(xs, polyFunc(1,xs), next(colors), label="Linear", linewidth=LINEWIDTH)
# plt.plot(xs, polyFunc(2,xs), next(colors), label="Quadratic", linewidth=LINEWIDTH)
# plt.plot(xs, polyFunc(4,xs), next(colors), label="Quartic", linewidth=LINEWIDTH)
# plt.plot(xs, linear_relative, next(colors), label="Linear", linewidth=LINEWIDTH)
# plt.plot(xs, quadratic_relative, next(colors), label="Quadratic Normalized", linewidth=LINEWIDTH)
# plt.plot(xs, quartic_relative, next(colors), label="Quartic Normalized", linewidth=LINEWIDTH)

plt_xmin, plt_xmax, plt_ymin, plt_ymax = plt.axis()

plt.plot([x0, x0], [plt_ymin-1, y_at_x0-0.008], 'k', alpha=0.7, linestyle=":", linewidth=LINEWIDTH)

if(LATEX_FLAG):
    plt.xlabel(r'$\boldsymbol{p}$', fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel(r'$\boldsymbol{g_{p_0}(p)}$', fontsize=AXIS_LABEL_FONT_SIZE)
else:
    plt.xlabel('p', fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel('g_p_0(p)', fontsize=AXIS_LABEL_FONT_SIZE)

plt.locator_params(axis='x', nbins=11)
plt.ylim(plt_ymin, plt_ymax) # this is to compensate on the dotted line of x0

leg = plt.legend()
plt.setp(leg.get_lines(), linewidth=LEGEND_WIDTH)
plt.setp(leg.get_texts(), fontsize=LEGEND_TEXT_SIZE)

# plt.savefig('Linear normalization with pi_0 = {{}}_ver2'.format(pi_0) + '.png')
plt.show()

# test_derivatives(lambda x: np.sin(x), delta)