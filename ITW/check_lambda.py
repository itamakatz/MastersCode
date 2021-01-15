import math
import numpy as np

THRESHOLD = 1e-10 # threshold for floating point comparison

# region : Pure Functions

def get_l_star(pi_0_,s_0_,s_1_):
    pi_1_ = 1-pi_0_
    if(pi_0_ < 0.5): # depending on pi_0 if s_a,s_b are s_0,s_1 or s_1,s_0
        return ((s_0_*pi_0_)-(s_1_*pi_1_))/(pi_1_-pi_0_)
    else:
        return ((s_1_*pi_0_)-(s_0_*pi_1_))/(pi_1_-pi_0_)

def get_delta_0(pi_0_,s_0_,s_1_):
    pi_1_ = 1-pi_0_
    return pi_0_*pi_1_/(pi_0_-pi_1_)*(s_0_-s_1_)**2;

def get_lambda_i(s_i_,l_):
    return (s_i_ + l_)**2

def get_lambda_bar(pi_0_,lambda_0_,lambda_1_):
    pi_1_ = 1-pi_0_
    return lambda_0_*pi_0_ + lambda_1_*pi_1_
            
def get_pi_n(pi_0_,lambda_0_,lambda_bar_):
    return pi_0_*lambda_0_/lambda_bar_

def get_lambda_bar_original_1(pi_0_,pi_n_,delta_0_):
    pi_1_ = 1-pi_0_
    return delta_0_ * (math.sqrt(pi_n_*pi_1_) + math.sqrt((1-pi_n_)*pi_0_))**2
def get_lambda_bar_original_2(pi_0_,pi_n_,delta_0_):
    pi_1_ = 1-pi_0_
    return delta_0_ * (math.sqrt(pi_n_*pi_1_) - math.sqrt((1-pi_n_)*pi_0_))**2

def get_lambda_bar_factor_1(pi_0_,pi_n_,delta_0_):
    lambda_bar_original_1_ = get_lambda_bar_original_1(pi_0_,pi_n_,delta_0_)
    pi_1_ = 1-pi_0_
    return lambda_bar_original_1_*(pi_0_-pi_1_) / ((pi_n_-pi_0_)**2)
def get_lambda_bar_factor_2(pi_0_,pi_n_,delta_0_):
    lambda_bar_original_2_ = get_lambda_bar_original_2(pi_0_,pi_n_,delta_0_)
    pi_1_ = 1-pi_0_
    return lambda_bar_original_2_*(pi_0_-pi_1_) / ((pi_n_-pi_0_)**2)

# endregion : Pure Functions

s_0_arr = np.arange(-2,2,0.4)
s_1_arr = np.arange(-2,2,0.4)
pi_0_arr = np.arange(0,1,0.1)
l_arr = np.arange(-2,2,0.4)

print('Input Parameters Arrays:\n')
print('\t'+'s_0_arr' + ' = ' + str(s_0_arr))
print('\t'+'s_1_arr' + ' = ' + str(s_1_arr))
print('\t'+'pi_0_arr' + ' = ' + str(pi_0_arr))
print('\t'+'l_arr' + ' = ' + str(l_arr))

count_original_pass_1 = 0
count_original_pass_2 = 0
count_factor_pass_1 = 0
count_factor_pass_2 = 0
count_fail = 0

for s_0 in s_0_arr:
    for s_1 in s_1_arr:
        for pi_0 in pi_0_arr:
            for l in l_arr:

# print input parameters

                print('Input Parameters:')
                print('\t'+'s_0' + ' = ' + str(s_0))
                print('\t'+'s_1' + ' = ' + str(s_1))
                print('\t'+'pi_0' + ' = ' + str(pi_0))
                print('\t'+'l' + ' = ' + str(l))

# Calculate parameters

                delta_0 = get_delta_0(pi_0,s_0,s_1)
                lambda_0 = get_lambda_i(s_0,l)
                lambda_1 = get_lambda_i(s_1,l)
                lambda_bar = get_lambda_bar(pi_0,lambda_0,lambda_1)
                pi_n = get_pi_n(pi_0,lambda_0,lambda_bar)
                lambda_bar_original_1 = get_lambda_bar_original_1(pi_0,pi_n,delta_0)
                lambda_bar_original_2 = get_lambda_bar_original_2(pi_0,pi_n,delta_0)
                lambda_bar_factor_1 = get_lambda_bar_factor_1(pi_0,pi_n,delta_0)
                lambda_bar_factor_2 = get_lambda_bar_factor_2(pi_0,pi_n,delta_0)

# Print results

                print('Results:')
                print('\t'+'lambda_bar' + ' = ' + str(lambda_bar))
                print('\t'+'lambda_bar_original_1' + ' = ' + str(lambda_bar_original_1))
                print('\t'+'lambda_bar_original_2' + ' = ' + str(lambda_bar_original_2))
                print('\t'+'lambda_bar_factor_1' + ' = ' + str(lambda_bar_factor_1))
                print('\t'+'lambda_bar_factor_2' + ' = ' + str(lambda_bar_factor_2))

# Update counts

                failed_flag = True
                if(abs(lambda_bar_original_1 - lambda_bar) < THRESHOLD):
                    count_original_pass_1 = count_original_pass_1 + 1
                    failed_flag = False
                elif(abs(lambda_bar_original_2 - lambda_bar) < THRESHOLD):                    
                    count_original_pass_2 = count_original_pass_2 + 1
                    failed_flag = False
                if(abs(lambda_bar_factor_1 - lambda_bar) < THRESHOLD):
                    count_factor_pass_1 = count_factor_pass_1 + 1
                    failed_flag = False
                elif(abs(lambda_bar_factor_2 - lambda_bar) < THRESHOLD):
                    count_factor_pass_2 = count_factor_pass_2 + 1
                    failed_flag = False
                if(failed_flag):
                    count_fail = count_fail + 1


print('******************************************')

print('count_original_pass_1' + ' = ' + str(count_original_pass_1))
print('count_original_pass_2' + ' = ' + str(count_original_pass_2))
print('count_factor_pass_1' + ' = ' + str(count_factor_pass_1))
print('count_factor_pass_2' + ' = ' + str(count_factor_pass_2))
print('count_fail' + ' = ' + str(count_fail))