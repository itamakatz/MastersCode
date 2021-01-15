import math
import numpy as np

# region : Magic Numbers

THRESHOLD = 1e-3 #1e-10 # threshold for floating point comparison
PRINT = False

# end region : Magic Numbers

# region : Pure Functions

def get_l_star(pi_0_,s_0_,s_1_):
    pi_1_ = 1-pi_0_
    if(pi_0_ < 0.5): # depending on pi_0 if s_a,s_b are s_0,s_1 or s_1,s_0
        return ((s_0_*pi_0_)-(s_1_*pi_1_))/(pi_1_-pi_0_)
    else:
        return ((s_1_*pi_0_)-(s_0_*pi_1_))/(pi_1_-pi_0_)

def get_delta_0(pi_0_,s_0_,s_1_):
    pi_1_ = 1-pi_0_
    return pi_0_*pi_1_/(pi_0_-pi_1_)*(s_0_-s_1_)**2; # error!!!
    # return pi_0_*pi_1_/(pi_1_-pi_0_)*(s_0_-s_1_)**2;

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

def get_lambda_bar_1(pi_0_,pi_n_,delta_0_):
    pi_1_ = 1-pi_0_
    return delta_0_ * (math.sqrt(pi_n_*pi_1_) + math.sqrt((1-pi_n_)*pi_0_))**2 * (pi_0_-pi_1_) / ((pi_n_-pi_0_)**2)
def get_lambda_bar_2(pi_0_,pi_n_,delta_0_):
    pi_1_ = 1-pi_0_
    return delta_0_ * (math.sqrt(pi_n_*pi_1_) - math.sqrt((1-pi_n_)*pi_0_))**2 * (pi_0_-pi_1_) / ((pi_n_-pi_0_)**2)

def get_lambda_bar_yuval(pi_0_,pi_n_,delta_0_,s_0_,s_1_):
    pi_1_ = 1-pi_0_
    return (pi_0_ * pi_1_*((s_0_-s_1_)**2))/((math.sqrt(pi_0_*pi_n_)-math.sqrt(pi_1_*(1-pi_n_)))**2)

def get_relative_g(pi_0_,pi_n_):
    pi_1_ = 1-pi_0_
    return (2*math.sqrt(pi_0_*pi_1_*pi_n_*(1-pi_n_)) - 2*pi_0_*pi_1_)/(pi_1_-pi_0_) + pi_0_
def get_pi_minus_relative_g(pi_0_,pi_n_):
    pi_1_ = 1-pi_0_
    return (math.sqrt(pi_1_*pi_n_)-math.sqrt(pi_0_*(1-pi_n_)))**2/(pi_1_-pi_0_)

# endregion : Pure Functions

s_0_arr = np.arange(-2,2,0.4)
s_1_arr = np.arange(-2,2,0.4)
pi_0_arr = np.arange(0,1,0.1)
l_arr = np.arange(-2,2,0.4)

if(PRINT):
    print('Input Parameters Arrays:\n')
    print('\t'+'s_0_arr' + ' = ' + str(s_0_arr))
    print('\t'+'s_1_arr' + ' = ' + str(s_1_arr))
    print('\t'+'pi_0_arr' + ' = ' + str(pi_0_arr))
    print('\t'+'l_arr' + ' = ' + str(l_arr))

count_lambda_pass_1 = 0
count_lambda_pass_2 = 0
count_success_1 = 0
count_success_2 = 0
count_fail = 0

for s_0 in s_0_arr:
    for s_1 in s_1_arr:
        for pi_0 in pi_0_arr:
            for l in l_arr:

# print input parameters

                if(PRINT):
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
                lambda_bar_1 = get_lambda_bar_1(pi_0,pi_n,delta_0)
                lambda_bar_2 = get_lambda_bar_2(pi_0,pi_n,delta_0)
                relative_g = get_relative_g(pi_0,pi_n)
                pi_minus_relative_g = get_pi_minus_relative_g(pi_0,pi_n)

                lambda_bar_yuval = get_lambda_bar_yuval(pi_0,pi_n,delta_0,s_0,s_1)

# Print results

                if(PRINT):
                    print('Results:')
                    print('\t'+'lambda_bar' + ' = ' + str(lambda_bar))
                    print('\t'+'lambda_bar_1' + ' = ' + str(lambda_bar_1))
                    print('\t'+'lambda_bar_2' + ' = ' + str(lambda_bar_2))

# Compare and update counts

                # compare_from_1 = lambda_bar_1 * (pi_n - relative_g)
                # compare_from_2 = lambda_bar_2 * (pi_n - relative_g)

                # compare_from_1 = lambda_bar_1
                # compare_from_2 = lambda_bar_2

                # compare_from_1 = lambda_bar_original_1
                # compare_from_2 = lambda_bar_original_2

                compare_from_1 = lambda_bar
                compare_from_2 = lambda_bar

                # compare_from_1 = lambda_bar_1 * pi_minus_relative_g
                # compare_from_2 = lambda_bar_2 * pi_minus_relative_g

                # compare_with = delta_0 / (pi_0-pi_n)
                # compare_with = -delta_0
                compare_with = lambda_bar_yuval

                # compare_from_1 = pi_n - relative_g
                # compare_from_2 = compare_from_1

                # compare_with = pi_minus_relative_g

                if(PRINT):
                    pass
                    print('\t'+'delta_0' + ' = ' + str(delta_0))
                    print('\t'+'relative_g' + ' = ' + str(relative_g))
                    print('\t'+'compare_from_1' + ' = ' + str(compare_from_1))
                    print('\t'+'compare_from_2' + ' = ' + str(compare_from_2))
                    print('\t'+'compare_with' + ' = ' + str(compare_with))
                    print("");

                if(abs(lambda_bar_1 - lambda_bar) < THRESHOLD):
                    count_lambda_pass_1 = count_lambda_pass_1 + 1
                if(abs(lambda_bar_2 - lambda_bar) < THRESHOLD):                    
                    count_lambda_pass_2 = count_lambda_pass_2 + 1

                if(abs( compare_from_1 - compare_with) < THRESHOLD):
                    count_success_1 = count_success_1 + 1
                elif(abs( compare_from_2 - compare_with) < THRESHOLD):
                    count_success_2 = count_success_2 + 1
                elif(math.isnan(compare_from_1) or math.isnan(compare_from_2) or math.isnan(compare_with) or
                    math.isinf(compare_from_1) or math.isinf(compare_from_2) or math.isinf(compare_with)):
                    pass
                else:
                    count_fail = count_fail + 1


print('******************************************')

print('count_lambda_pass_1' + ' = ' + str(count_lambda_pass_1))
print('count_lambda_pass_2' + ' = ' + str(count_lambda_pass_2))
print('count_success_1' + ' = ' + str(count_success_1))
print('count_success_2' + ' = ' + str(count_success_2))
print('count_fail' + ' = ' + str(count_fail))