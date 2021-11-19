import numpy as np
import multiprocessing
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

s0=2
s1=-3
t=1e-15
pi0=0.3
T_STEP = 0.002

def lambda0(ell):
    return abs(s0+ell)**2
def lambda1(ell):
    return abs(s1+ell)**2
def lambda_bar(pi,ell):
    return pi*lambda0(ell)+(1-pi)*lambda1(ell)

def p0(pi,ell):
    return pi*lambda0(ell)/lambda_bar(pi,ell)
def p1(pi,ell):
    return (1-pi*lambda0(ell)*t)*pi/(1-pi*lambda_bar(pi,ell)*t)

def pi_11(pi, ell_pi, ell_p0, ell_p1): return p1(p1(pi,ell_pi),ell_p1)
def pi_01(pi, ell_pi, ell_p0, ell_p1): return p1(p0(pi,ell_pi),ell_p0)
def pi_10(pi, ell_pi, ell_p0, ell_p1): return p0(p1(pi,ell_pi),ell_p1)
def pi_00(pi, ell_pi, ell_p0, ell_p1): return p0(p0(pi,ell_pi),ell_p0)

def Pr_pi_11(pi, ell_pi, ell_p0, ell_p1):
    return lambda_bar(pi,ell_pi)*t*lambda_bar(pi,ell_p1)*t
def Pr_pi_01(pi, ell_pi, ell_p0, ell_p1):
    return (1-lambda_bar(pi,ell_pi)*t)*lambda_bar(pi,ell_p0)*t
def Pr_pi_10(pi, ell_pi, ell_p0, ell_p1):
    return lambda_bar(pi,ell_pi)*t*(1-lambda_bar(pi,ell_p1)*t)
def Pr_pi_00(pi, ell_pi, ell_p0, ell_p1):
    return (1-lambda_bar(pi,ell_pi)*t)*(1-lambda_bar(pi,ell_p0)*t)

def D(pi, ell_pi, ell_p0, ell_p1,g):
    return \
    g(pi_11(pi, ell_pi, ell_p0, ell_p1))*Pr_pi_11(pi, ell_pi, ell_p0, ell_p1)+ \
    g(pi_01(pi, ell_pi, ell_p0, ell_p1))*Pr_pi_01(pi, ell_pi, ell_p0, ell_p1)+ \
    g(pi_10(pi, ell_pi, ell_p0, ell_p1))*Pr_pi_10(pi, ell_pi, ell_p0, ell_p1)+ \
    g(pi_00(pi, ell_pi, ell_p0, ell_p1))*Pr_pi_00(pi, ell_pi, ell_p0, ell_p1)

def run_D(data):
    ell_pi = data[0]
    ell_p0 = data[1]
    ell_p1 = data[1]
    return (data[0], data[1], D(pi0, ell_pi, ell_p0, ell_p1,lambda x: get_renyi_entropy(x,0.5)))

def fisher_general(x,a): return (x*(1-x))**a
def fisher_general_der1(x,a):return a*(1-2*x)*((x*(1-x))**(a-1))

def entropy(x): return -x*np.log2(x)-(1-x)*np.log2(1-x)
def entropy_der1(x): return np.log2(1-x) - np.log2(x)

def get_renyi_entropy(x, alpha):return (np.log2(x**alpha +(1-x)**alpha))/(1-alpha)
def renyi_der1(x, alpha): return (alpha*x**(alpha-1)-alpha*(1-x)**(alpha-1))/((1-alpha)*np.log(2)*(x**alpha+(1-x)**alpha))

def main():
    print(f"optimal l gives D={D(pi0,-s0,-s0,-s0,lambda x: get_renyi_entropy(x,0.5))}")

    ls = np.arange(-3., 3., T_STEP)
    min_D = 1e10
    l_best = 1e10
    # for ell in ls:
    #     d = D(pi0,ell,-s0,-s0,lambda x: get_renyi_entropy(x,0.5))
    #     l_best = ell if d < min_D else l_best
    #     min_D = d if d < min_D else min_D

    # Ds = np.array(list(map(lambda ell: D(pi0,ell,-s0,-s0,lambda x: get_renyi_entropy(x,0.5)), ls)))

    # plt.plot(ls,Ds)
    # plt.show()

    # print(f"best found l is l={l_best} and gives D={min_D}")
    fig = plt.figure()
    ax = Axes3D(fig)

    ls0, ls1 = np.meshgrid(np.array(ls), np.array(ls))
    ls0_data = ls0.reshape((len(ls)*len(ls)))
    ls1_data = ls1.reshape((len(ls)*len(ls)))
    data = list(zip(ls0_data,ls1_data))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()*2) as pool:
    # with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(run_D, data)

    results.sort(key=lambda x: x[1])
    results.sort(key=lambda x: x[0])
    results = np.array(results)
    ells_0 = results[:,0]
    ells_1 = results[:,1]
    ds = results[:,2]
    ds = 1 - ds

    LS_0 = np.array(ells_0).reshape((len(ls),len(ls)))
    LS_1 = np.array(ells_1).reshape((len(ls),len(ls)))
    ZS = np.array(ds).reshape((len(ls),len(ls)))

    surf = ax.plot_surface(LS_0, LS_1, ZS, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == '__main__':
    main()