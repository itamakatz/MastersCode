import numpy as np
import matplotlib
import multiprocessing
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# ========================= MAGICS ========================= #

s0=2
s1=-3
t=1e-10 # 1e-14
PI_0=0.003
# PI_0=0.3
T_STEP = 0.0002
# T_STEP = 0.02
LEVELS=1
AXIS_LABEL_FONT_SIZE = 40 # 25
LINEWIDTH = 1
TICK_SIZE = 20 # 30
LEGEND_LENGTH = 4
LEGEND_WIDTH = 4
LEGEND_TEXT_SIZE = 30 # 30
TITLE_FONT_SIZE = 15

# USE_LATEX = False
USE_LATEX = True

objective_func = lambda x: renyi(x,0.5)
# objective_func = lambda x: fisher_general(x,1)
# objective_func = lambda x: fisher_general(x,0.5)
# objective_func = lambda x: entropy(x)
# objective_func = lambda x: error(x)

# ========================================================== #

if(USE_LATEX):
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rcParams['xtick.labelsize'] = TICK_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_SIZE
matplotlib.rcParams['legend.handlelength'] = LEGEND_LENGTH

def get_and_set_legend():
    leg = plt.legend()
    plt.setp(leg.get_lines(), linewidth=LEGEND_WIDTH)
    plt.setp(leg.get_texts(), fontsize=LEGEND_TEXT_SIZE)
    return leg
def new_figure():
    return plt.figure(num=None, figsize=(16, 12), dpi=80)

def lambda0(ell):
    return abs(s0+ell)**2
def lambda1(ell):
    return abs(s1+ell)**2
def lambda_bar(pi,ell):
    return pi*lambda0(ell)+(1-pi)*lambda1(ell)

def p0(pi,ell):
    return (1-lambda0(ell)*t)*pi/(1-lambda_bar(pi,ell)*t)
def p1(pi,ell):
    return pi*lambda0(ell)/lambda_bar(pi,ell)

def dolinar_ell(pi):
    return (s0*pi-s1*(1-pi))/(1-2*pi)

def D_recursive(levels, pi, ell_array, g):
    '''
    levels - how many timesteps to compute
    pi - the prior of the current timestep
    ell_array - all control signals to be used
    g - the objective function
    '''
    assert(len(ell_array)==2**(levels+1)-1)
    current_ell = ell_array[0]
    if(levels == 0):
        return g(p0(pi,current_ell))*(1-lambda_bar(pi,current_ell)*t) + g(p1(pi,current_ell))*lambda_bar(pi,current_ell)*t
    ell_array = ell_array[1:]
    #    Todo: check ell_array[:len(ell_array)//2] and ell_array[len(ell_array)//2:]
    return D_recursive(levels-1, p0(pi,current_ell), ell_array[:len(ell_array)//2], g)*(1-lambda_bar(pi,current_ell)*t) + \
           D_recursive(levels-1, p1(pi,current_ell), ell_array[len(ell_array)//2:], g)*lambda_bar(pi,current_ell)*t

def run_D(data):
    ell_pi = data[0]
    ell_p0 = data[1]
    ell_p1 = data[1]
    return (data[0], data[1], D_recursive(LEVELS,PI_0, [ell_pi, ell_p0, ell_p1],objective_func))

def fisher_general(x,a): return (x*(1-x))**a
def fisher_general_der1(x,a):return a*(1-2*x)*((x*(1-x))**(a-1))

def entropy(x): return -x*np.log2(x)-(1-x)*np.log2(1-x)
def entropy_der1(x): return np.log2(1-x) - np.log2(x)

def renyi(x, alpha):return (np.log2(x**alpha +(1-x)**alpha))/(1-alpha)
def renyi_der1(x, alpha): return (alpha*x**(alpha-1)-alpha*(1-x)**(alpha-1))/((1-alpha)*np.log(2)*(x**alpha+(1-x)**alpha))

def error(x): return np.array([x, 1-x]).min(axis=0)

def steps_2_fixed_2():

    ls = np.arange(-4., -1., T_STEP)
    functions = [
        (lambda x: renyi(x,0.5), 'Renyi'),
        (lambda x: fisher_general(x,1), 'Fisher'),
        # (lambda x: fisher_general(x,0.5), 'Root-Fisher'),
        # (entropy, 'Entropy'),
        # (error, 'Error'),
    ]

    for f in functions:
        new_figure()
        ys = np.array(list(map(lambda ell: D_recursive(1,PI_0,[ell,-s0,-s0],f[0]), ls)))
        print(min(ys))
        print(ls[np.argmin(ys)])
        plt.plot(ls,ys,label=f[1], linewidth=LINEWIDTH)
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        # plt.plot([dolinar_ell(PI_0),dolinar_ell(PI_0)],[ymin,ymax], linestyle='--', alpha=0.7, label='$\ell_{Dolinar}$')
        plt.plot([-s0,-s0],[ymin, ymax], linestyle='--', alpha=0.7, label='$-s_0$')
        plt.ylim(ymin, ymax)
        get_and_set_legend()
        if(USE_LATEX):
            plt.xlabel(r'$\boldsymbol{\ell}$', fontsize=AXIS_LABEL_FONT_SIZE)
            plt.ylabel(r'$\boldsymbol{D[\pi]}$', fontsize=AXIS_LABEL_FONT_SIZE)
            plt.title(r'Two timestep. Second $\ell$ is fixed to $-s_0$. $\pi(0)=$'+str(PI_0) , fontsize=TITLE_FONT_SIZE)
        plt.show()

    # fig, axs = plt.subplots(2)
    # fig.suptitle(r'Two timestep. Second $\ell$ is fixed to $-s_0$. $\pi(0)=$'+str(PI_0) , fontsize=TITLE_FONT_SIZE)
    # count = 0
    # for f in functions:
    #     axs[count].plot(ls,np.array(list(map(lambda ell: D_recursive(1,PI_0,[ell,-s0,-s0],f[0]), ls))),label=f[1], linewidth=LINEWIDTH)
    #     ax = plt.gca()
    #     ymin, ymax = axs[count].get_ylim()
    #     axs[count].plot([dolinar_ell(PI_0),dolinar_ell(PI_0)],[ymin,ymax], linestyle='--', alpha=0.7, label='$\ell_{Dolinar}$')
    #     axs[count].plot([-s0,-s0],[ymin, ymax], linestyle='--', alpha=0.7, label='$-s_0$')
    #     get_and_set_legend()
    #     count+=1
    # plt.show()

def steps_2_fixed_1():

    ls = np.arange(-20., 20., T_STEP)
    functions = [
        (lambda x: renyi(x,0.5), 'Renyi'),
        (lambda x: fisher_general(x,1), 'Fisher'),
        # (lambda x: fisher_general(x,0.5), 'Root-Fisher'),
        # (entropy, 'Entropy'),
        # (error, 'Error'),
    ]

    for f in functions:
        new_figure()
        plt.plot(ls,np.array(list(map(lambda ell: D_recursive(1,PI_0,[-s0,ell,ell],f[0]), ls))),label=f[1], linewidth=LINEWIDTH)
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        plt.plot([dolinar_ell(PI_0),dolinar_ell(PI_0)],[ymin,ymax], linestyle='--', alpha=0.7, label='Dolinar')
        plt.plot([-s0,-s0],[ymin, ymax], linestyle='--', alpha=0.7, label='$-s_0$')
        plt.ylim(ymin, ymax)
        get_and_set_legend()
        if(USE_LATEX):
            plt.xlabel(r'$\boldsymbol{\ell}$', fontsize=AXIS_LABEL_FONT_SIZE)
            plt.ylabel(r'$\boldsymbol{D[\pi]}$', fontsize=AXIS_LABEL_FONT_SIZE)
            plt.title(r'Two timestep. First $\ell$ is fixed to $-s_0$. $\pi(0)=$'+str(PI_0) , fontsize=TITLE_FONT_SIZE)
        plt.show()

def steps_1():
    # ls = np.concatenate((np.arange(-200., -20., T_STEP*100), np.arange(-20., 20., T_STEP), np.arange(20., 200., T_STEP*100)))
    ls = np.arange(-20., 20., T_STEP)
    functions = [
        (lambda x: renyi(x,0.5), 'Renyi'),
        # (lambda x: fisher_general(x,1), 'Fisher'),
        # (lambda x: fisher_general(x,0.5), 'Root-Fisher'),
        # (entropy, 'Entropy'),
        # (error, 'Error'),
    ]

    new_figure()

    for f in functions:
        plt.plot(ls,np.array(list(map(lambda ell: D_recursive(0,PI_0,[ell],f[0]), ls))),label=f[1], linewidth=LINEWIDTH)

    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    plt.plot([dolinar_ell(PI_0),dolinar_ell(PI_0)],[ymin,ymax], linestyle='--', alpha=0.7, label='Dolinar')
    plt.plot([-s0,-s0],[ymin, ymax], linestyle='--', alpha=0.7, label='$-s_0$')
    plt.ylim(ymin, ymax)
    get_and_set_legend()
    if(USE_LATEX):
        plt.xlabel(r'$\boldsymbol{\ell}$', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.ylabel(r'$\boldsymbol{D[\pi]}$', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.title(r'Single timestep. $\pi(0)=$'+str(PI_0), fontsize=TITLE_FONT_SIZE)
    plt.show()

def steps_2_3d_plot():
    
    # print(f"optimal l gives D={D(PI_0,-s0,-s0,-s0,objective_func)}")
    print(f"optimal l gives D={D_recursive(1,PI_0, [-s0,-s0,-s0],objective_func)}")

    dolinar_pi = dolinar_ell(PI_0)
    dolinar_p0 = dolinar_ell(p0(PI_0,dolinar_pi))
    dolinar_p1 = dolinar_ell(p1(PI_0,dolinar_pi))
    print(f"dolinar ell_pi is {dolinar_pi}")
    print(f"dolinar ell_p0 is {dolinar_p0}")
    print(f"dolinar ell_p1 is {dolinar_p1}")
    print(f"dolinar D is D={D_recursive(1,PI_0, [dolinar_pi,dolinar_p0,dolinar_p1],objective_func)}")

    # print(f"best found l is l={l_best} and gives D={min_D}")
    fig = plt.figure()
    ax = Axes3D(fig)

    # ls = np.arange(-4., 4., T_STEP)
    ls = np.arange(-20., 20., T_STEP*100)

    ls0, ls1 = np.meshgrid(np.array(ls), np.array(ls))
    ls0_data = ls0.reshape((len(ls)*len(ls)))
    ls1_data = ls1.reshape((len(ls)*len(ls)))
    data = list(zip(ls0_data,ls1_data))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()*2) as pool:
        results = pool.map(run_D, data)

    # results = np.array(list(map(lambda d: run_D(d), data)))

    results.sort(key=lambda x: x[1])
    results.sort(key=lambda x: x[0])
    results = np.array(results)
    ells_0 = results[:,0]
    ells_1 = results[:,1]
    ds = results[:,2]
    index_min = np.argmin(ds)
    print(ells_0[index_min])
    print(ells_1[index_min])
    print(np.min(ds))
    # ds = np.log10(ds)
    ds = 1 - ds

    LS_0 = np.array(ells_0).reshape((len(ls),len(ls)))
    LS_1 = np.array(ells_1).reshape((len(ls),len(ls)))
    DS = np.array(ds).reshape((len(ls),len(ls)))

    surf = ax.plot_surface(LS_0, LS_1, DS, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == '__main__':
    # steps_2_3d_plot()
    # steps_1()
    steps_2_fixed_2()
    # steps_2_fixed_1()
    # print(f"optimal l gives D={D(PI_0,-s0,-s0,-s0,objective_func)}")
    # print(f"recursive gives D={D_recursive(1,PI_0, [-s0,-s0,-s0],objective_func)}")
    # print(f"iterative gives D={D_iterative(1,PI_0, [-s0,-s0,-s0],objective_func)}")