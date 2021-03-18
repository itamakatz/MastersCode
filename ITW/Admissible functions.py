import numpy as np
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

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

def get_fisherGeneral(t,a):
    return np.array((t*(1-t))**a)
def get_fisherGeneral_der1(t,a):
    return np.array(a*(1-2*t)*((t*(1-t))**(a-1)))
def get_fisherGeneral_der2(t,a):
    return np.array((a-1)*a*((1-2*t)**2)*(((1-t)*t)**(a-2))-2*a*(((1-t)*t)**(a-1)))
def get_fisherGeneral_ratio(t,a):
    return np.array(get_fisherGeneral_der2(t,a)/get_fisherGeneral_der1(t,a))
def get_fisherGeneral_normalization(t, x0, a):
    return get_normalization(t, get_fisherGeneral(t,a), get_fisherGeneral_der1(t,a), x0)

def get_entropy(t):
    t = t[1:-1]
    return np.array(np.append(np.append(0, -t*np.log(t)-(1-t)*np.log(1-t)), 0))
def get_entropy_der1(t):
    return np.array(np.log(1-t) - np.log(t))
def get_entropy_der2(t):
    return np.array(1/(t*(t-1)))
def get_entropy_ratio(t):
    return np.array(get_entropy_der2(t)/get_entropy_der1(t))
def get_entropy_normalization(t, x0):
    return get_normalization(t, get_entropy(t), get_entropy_der1(t), x0)

def get_error(t):
    return np.array([t, 1-t]).min(axis=0)
def get_error_der1(t):
    return np.array(list(map(lambda x: 1 if x < 0.5 else -1, t)));
def get_error_der2(t):
    return np.array(list(map(lambda x: 0, t)));
def get_error_ratio(t):
    return np.array(get_error_der2(t)/get_error_der1(t))
def get_error_normalization(t, x0):
    return get_normalization(t, get_error(t), get_error_der1(t), x0)

def get_normalization(xs, func_ys, func_der1, x0):
    x0_index = get_array_index_by_value(xs, x0)
    func_at_x0 = func_ys[x0_index]
    func_der1_at_x0 = func_der1[x0_index]
    return np.array((func_ys-func_at_x0) / abs(func_der1_at_x0) +x0)

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

def log(arr):
    return np.log(np.abs(np.array(arr)))

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
    # colors_list = list(c[0] for c in mcolors.TABLEAU_COLORS.items())
    colors_list = 'brgycm'
    return cycle(colors_list)

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

def plot_admissible_functions(t):

    data = [
        (get_fisherGeneral(t,1), 'Fisher'),
        (get_fisherGeneral(t,0.5), 'Root-Fisher'),
        (get_entropy(t), 'Binary Entropy'),
        (get_error(t), 'Error Probability'),
    ]

    new_figure() # create new figure
    colors = get_cycle_colors() # get plot colors

    for i in range(len(data)):
        plt.plot(t, data[i][0], next(colors), label=data[i][1], linewidth=LINEWIDTH)

    # Set Figure details #  
    get_and_set_legend()
    plt.ylim(0, get_max_arrays(list( item[0] for item in data ), 1.05)) # set the y scope - min y to max y
    if(USE_LATEX): # latex lables
        plt.xlabel(r'$\boldsymbol{p}$', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel(r'$\boldsymbol{g(p)}$', fontsize=AXIS_LABEL_FONT_SIZE)

    if(SAVE_FIGURES):
        plt.savefig('Admissible Functions.png')

    plt.show()

def plot_normalization_functions(t, x0 = 0.3):

    data = [
        (get_fisherGeneral_normalization(t, x0, 1), 'Fisher'),
        (get_fisherGeneral_normalization(t, x0, 0.5), 'Root-Fisher'),
        (get_entropy_normalization(t, x0), 'Binary Entropy'),
        (get_error_normalization(t, x0), 'Error Probability'),
    ]

    new_figure() # create new figure
    colors = get_cycle_colors() # get plot colors

    plt.plot([x0,x0], [get_min_arrays(list( item[0] for item in data ), 1.05)-1, data[0][0][get_array_index_by_value(t, x0)]], color='k', linestyle='--', alpha=0.7) # dotted line

    for i in range(len(data)):
        plt.plot(t, data[i][0], next(colors), label=data[i][1], linewidth=LINEWIDTH)

    # Set Figure details #
    get_and_set_legend()
    plt.ylim(get_min_arrays(list( item[0] for item in data ), 1.05), get_max_arrays(list( item[0] for item in data ), 1.05)) # set the y scope - min y to max y
    plt.xticks(np.concatenate((np.linspace(0.0, 1.0, num=6, endpoint=True), np.ones(1)*x0))) # set the x axis ticks
    if(USE_LATEX): # latex lables
        plt.xlabel(r'$\boldsymbol{p}$', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel(r'$\boldsymbol{g_{p_0}(p)}$', fontsize=AXIS_LABEL_FONT_SIZE)

    if(SAVE_FIGURES):
        plt.savefig(f'Normalized Functions with pi_0={x0}.png')

    plt.show()

t = np.append(np.arange(0., 1., T_STEP),1)
# plot_admissible_functions(t)
plot_normalization_functions(t,0.7)