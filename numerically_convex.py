import numpy as np
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

DEBUG = True
# DEBUG = False

SAVE_FIGURES = False
# SAVE_FIGURES = True

USE_LATEX = False
# USE_LATEX = True

T_STEP = 0.0002

# Plot variables:
LINEWIDTH = 5
AXIS_LABEL_FONT_SIZE = 40 # 25
TICK_SIZE = 35 # 30
LEGEND_LENGTH = 4
LEGEND_WIDTH = 4
LEGEND_TEXT_SIZE = 35 # 30

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

if(USE_LATEX):
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rcParams['xtick.labelsize'] = TICK_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_SIZE
matplotlib.rcParams['legend.handlelength'] = LEGEND_LENGTH

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

def get_fisherGeneral(xs,a):
    return np.array((xs*(1-xs))**a)
def get_fisherGeneral_der1(xs,a):
    return np.array(a*(1-2*xs)*((xs*(1-xs))**(a-1)))
def get_fisherGeneral_der2(xs,a):
    return np.array((a-1)*a*((1-2*xs)**2)*(((1-xs)*xs)**(a-2))-2*a*(((1-xs)*xs)**(a-1)))
def get_fisherGeneral_ratio(xs,a):
    return np.array(get_fisherGeneral_der2(xs,a)/get_fisherGeneral_der1(xs,a))
def get_fisherGeneral_normalization(xs, x0, a):
    return get_normalization(xs, get_fisherGeneral(xs,a), get_fisherGeneral_der1(xs,a), x0)

def get_entropy(xs):
    xs = xs[1:-1]
    return np.array(np.append(np.append(0, -xs*np.log2(xs)-(1-xs)*np.log2(1-xs)), 0))
def get_entropy_der1(xs):
    return np.array(np.log2(1-xs) - np.log2(xs))
def get_entropy_der2(xs):
    return np.array(1/(xs*(xs-1)))
def get_entropy_ratio(xs):
    return np.array(get_entropy_der2(xs)/get_entropy_der1(xs))
def get_entropy_normalization(xs, x0):
    return get_normalization(xs, get_entropy(xs), get_entropy_der1(xs), x0)

def get_error(xs):
    return np.array([xs, 1-xs]).min(axis=0)
def get_error_der1(xs):
    return np.array(list(map(lambda x: 1 if x < 0.5 else -1, xs)));
def get_error_der2(xs):
    return np.array(list(map(lambda x: 0, xs)));
def get_error_ratio(xs):
    return np.array(get_error_der2(xs)/get_error_der1(xs))
def get_error_normalization(xs, x0):
    return get_normalization(xs, get_error(xs), get_error_der1(xs), x0)

def get_renyi_entropy(xs, alpha):
    return np.array(list(map(lambda x: np.log2(x**alpha +(1-x)**alpha), xs)))/(1-alpha)
def get_renyi_entropy_der1(xs, alpha):
    return np.array(list(map(lambda x: (alpha*x**(alpha-1)-alpha*(1-x)**(alpha-1))/((1-alpha)*np.log(2)*(x**alpha+(1-x)**alpha)), xs)))
def get_renyi_entropy_normalization(xs, alpha, x0):
    return get_normalization(xs, get_renyi_entropy(xs, alpha), get_renyi_entropy_der1(xs, alpha), x0)

def get_normalization(xs, func_ys, func_der1, x0):
    x0_index = get_array_index_by_value(xs, x0)
    func_at_x0 = func_ys[x0_index]
    func_der1_at_x0 = func_der1[x0_index]
    return np.array((func_ys-func_at_x0) / abs(func_der1_at_x0) +x0)

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

def get_min_arrays(arrays, factor = 1):
    minY = np.array(arrays).min(axis=0).min()
    return minY * factor
def get_max_arrays(arrays, factor = 1):
    maxY = np.array(arrays).max(axis=0).max()
    return maxY * factor

def get_array_index_by_value(array, val):
    return min(range(len(array)), key=lambda i: abs(array[i]-val))

def new_figure():
    return plt.figure(num=None, figsize=(16, 12), dpi=80)

def get_and_set_legend():
    leg = plt.legend()
    plt.setp(leg.get_lines(), linewidth=LEGEND_WIDTH)
    plt.setp(leg.get_texts(), fontsize=LEGEND_TEXT_SIZE)
    return leg

def get_cycle_colors():
    colors_list = list(c[0] for c in mcolors.TABLEAU_COLORS.items())
    # colors_list = 'brgycm'
    return cycle(colors_list)

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

def shrink_arr(arr):
    return arr[1:]

def get_array_der(arr, xs, delta):
    der = np.diff(arr)/delta
    xs = xs[1:]
    return (xs, der)

def get_func_der(f, xs, delta):
    f_val = f(xs)
    f_der = np.diff(f_val)/delta
    xs = xs[1:]
    return (xs, f_der)

def get_relative_func(f, xs, delta, x0):
    ''' # Equation of the linear normalization:
    #   f_{x0}(x) = (f(x)-f(x0)) / |(f'(x0))| + x0
    # '''

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

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

def is_convex_relative_to(above, below, xs):

    for i in range(len(above)):
        if(above[0] < below[0]):
            return False
    return True

def find_alpha(x0):

    xs = np.append(np.arange(0., 1., T_STEP),1)

    epsilon = 1e-50

    alpha_h = 1 - epsilon
    alpha_l = 0 + epsilon
    alpha_last = 0
    count = 0
    while(True):
        count += 1
        alpha_current = (alpha_l + alpha_h) / 2
        if(abs(alpha_last - alpha_current) < epsilon):
            break
        alpha_last = alpha_current
        data = [
            (get_fisherGeneral_normalization(xs, x0, 0.5), 'Root-Fisher'),
            (get_renyi_entropy_normalization(xs,alpha_current,x0), 'Reny Entropy, a=0.001'),
        ]

        result = is_convex_relative_to(data[0][0],data[1][0],xs)

        # print(f"alpha={alpha_current}. alpha is:{'alpha_l' if alpha_current == alpha_l else 'alpha_h'}. result={result}")

        if(result == True):
            alpha_l = alpha_current + epsilon
        else:
            alpha_h = alpha_current - epsilon

    # print(f"Took: {count} iterations")
    return alpha_current

alpha_max = 1
alpha_min = 0

epsilon = 1e-50
x0_step = 2e-4
# x0s = np.arange(0.+epsilon, 0.5-epsilon, x0_step)
x0s = np.arange(0.48+epsilon, 0.5-epsilon, x0_step)
print(f"len(x0s)={len(x0s)}")
for x0 in x0s:
    current_alpha = find_alpha(x0)
    alpha_max = current_alpha if alpha_max == 1 else max(alpha_max, current_alpha)
    alpha_min = current_alpha if alpha_min == 0 else min(alpha_min, current_alpha)
    print(f"x0:{x0}, alpha_min:{alpha_min}, alpha_max:{alpha_max}")
