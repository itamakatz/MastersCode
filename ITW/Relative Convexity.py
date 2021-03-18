import numpy as np
import matplotlib.pyplot as plt
from numpy import diff

import matplotlib
import matplotlib.font_manager
LATEX_FLAG = False
if(LATEX_FLAG):
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

DEBUG = True
FISHER_COLOR = 'b' #'r'
FISHER_SQRT_COLOR = 'r' #'m'
ENTROPY_COLOR = 'g' #'b'
ERROR_COLOR = 'y' #'g'

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
#   f_{x0}(x) = (f(x)-f(x0)) / (f'(x0)) + x0
def get_relative_func(f, xs, delta, x0):
    (xs, f_der_array) = get_func_der(f, xs, delta)
    # x0_index = np.where(xs==x0)
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
    # print("f_der len: "+str(len(f_der)))
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

# x0 = 0.3
x0 = 0.7
delta = 0.0001

# xs = np.arange(0.0001, 0.4, delta)
xs = np.arange(0.0001, 1 - 0.0001, delta)
# xs = np.arange(0.0001, .5, delta)

fisherSqrt = lambda x: np.sqrt(x*(1-x))
fisher = lambda x: x*(1-x)
entropy = lambda x: -x*np.log(x)-(1-x)*np.log(1-x)
minError = lambda x: np.array([x,1-x]).min(0)
expFunc = lambda x: np.exp(x)
polyFunc = lambda a,x: np.power(x,a)

(_, fisherSqrt_relative) = get_relative_func(fisherSqrt,xs,delta,x0)
(_, fisher_relative) = get_relative_func(fisher,xs,delta,x0)
(_, entropy_relative) = get_relative_func(entropy,xs,delta,x0)
(xs, minError_relative) = get_relative_func(minError,xs,delta,x0)

# (_, exp_relative) = get_relative_func(expFunc,xs,delta,x0)
# (xs, quartic_relative) = get_relative_func(lambda x: polyFunc(4,x),xs,delta,x0)

# x0_index = np.where(xs==x0)
x0_index = get_array_index_by_value(xs,x0)
y_at_x0 = fisherSqrt_relative[x0_index]
min_y = min(fisher_relative[0], 
            fisherSqrt_relative[0], 
            entropy_relative[0], 
            minError_relative[0])
# y_at_x0 = exp_relative[x0_index]
# min_y = min(exp_relative[0], quartic_relative[0])

LINE_WIDTH = 2.5
plt.plot(xs, fisher_relative, FISHER_COLOR, label="Fisher Normalized", linewidth=LINE_WIDTH)
plt.plot(xs, fisherSqrt_relative, FISHER_SQRT_COLOR, label="Root-Fisher Normalized", linewidth=LINE_WIDTH)
plt.plot(xs, entropy_relative, ENTROPY_COLOR, label="Binary Entropy Normalized", linewidth=LINE_WIDTH)
plt.plot(xs, minError_relative, ERROR_COLOR, label="Error Probability Normalized", linewidth=LINE_WIDTH)

# plt.plot(xs, exp_relative, FISHER_COLOR, label="Exp Normalized", linewidth=LINE_WIDTH)
# plt.plot(xs, quartic_relative, FISHER_SQRT_COLOR, label="Quartic Normalized", linewidth=LINE_WIDTH)

plt_xmin, plt_xmax, plt_ymin, plt_ymax = plt.axis()

plt.plot([x0, x0], [plt_ymin-1, y_at_x0-0.008], 'k', alpha=0.7, linestyle=":", linewidth=LINE_WIDTH)
# plt.plot([x0+0.1, x0+0.1], [plt_ymin-1, y_at_x0], 'k', alpha=0.7, linestyle="-", linewidth=LINE_WIDTH)
# plt.plot([x0+0.2, x0+0.2], [plt_ymin-1, y_at_x0], 'k', alpha=0.7, linestyle="-.", linewidth=LINE_WIDTH)
# plt.plot([x0+0.3, x0+0.3], [plt_ymin-1, y_at_x0], 'k', alpha=0.7, linestyle=":", linewidth=LINE_WIDTH)

FONT_SIZE = 25
# plt.title(r'Linear normalization with $\displaystyle \pi_0={{{}}}$'.format(pi_0), fontsize=FONT_SIZE)
if(LATEX_FLAG):
    plt.xlabel(r'$\boldsymbol{p}$', fontsize=FONT_SIZE)
    plt.ylabel(r'$\boldsymbol{g_{p_0}(p)}$', fontsize=FONT_SIZE)
else:
    plt.xlabel('p', fontsize=FONT_SIZE)
    plt.ylabel('g_p_0(p)', fontsize=FONT_SIZE)

TICK_SIZE = 30
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.locator_params(axis='x', nbins=11)
plt.ylim(plt_ymin, plt_ymax) # this is to compensate on the dotted line of x0


# plt.ylim(-0.05, 0.55)
# plt.xlim(-0.05, 0.55)
# plt.ylim(-0.05, 1.05)
# plt.xlim(-0.05, 1.05)
# plt.legend(loc="upper left")

leg = plt.legend()
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
plt.setp(leg_lines, linewidth=4)
plt.setp(leg_texts, fontsize=20)

# plt.savefig('Linear normalization with pi_0 = {{}}_ver2'.format(pi_0) + '.png')
plt.show()

# test_derivatives(lambda x: np.sin(x), delta)

# plt.plot(
#     [x[amax], x[amax], xlim[0]], 
#     [xlim[0], y[amax], y[amax]]
    
# One of the conditions for admissible functions is - Not too concave: $g(p) \succ \sqrt{p(1-p)}$ when restricted to $p\in(0,1/2)$