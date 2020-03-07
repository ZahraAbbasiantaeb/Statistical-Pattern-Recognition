from numpy.random import choice
import numpy as np
import matplotlib.pyplot as plt

def p_2(x):

    return x * (np.exp(-x**2/2))


def p_1(x):

    if(0<=x and x<=1):
        return x

    elif(1<=x and x<=2):
        return 2-x

    else:
        return 0


def plot_kernel_density(samples, range, h):

    Y = []

    sigma = 1

    N = len(samples)

    for point in range:

        count = 0

        for elem in samples:

            X = np.abs(point - elem)/h

            count += (1/(np.sqrt(2*np.pi))) * \
                     np.exp(-1 * np.linalg.norm(X) ** 2 / ( 2))

        Y.append(count)

    Y = np.array(Y)

    plt.plot(range, Y)


    return


def calc_H_star(samples):

    N = len(samples)

    sigma = np.sqrt(np.var(samples))

    return (1.06 * sigma) / (N ** 0.2)


def generate_sample(N, func, range):

    pdf = [func(x) for x in range]

    pdf = np.array(pdf)

    sum = np.sum(pdf)

    pdf /= sum

    draw = choice(range, N,
                    p=pdf)

    H = calc_H_star(draw)


    return draw, H


# Part A


range_1 = np.arange(-2, 2, 0.001).tolist()

Y_1 = [p_1(x) for x in range_1]

range_2 = np.arange(0, 10, 0.001).tolist()

Y_2 = [p_2(x) for x in range_2]

plt.plot(range_1, Y_1)

plt.title('pdf of p1')

plt.show()

plt.plot(range_2, Y_2)

plt.title('pdf of p2')

plt.show()



# Part B

N = 10

draw_1, H_1 = generate_sample(N, p_1, range_1)

plt.hist(draw_1)

plt.show()

draw_2,  H_2 = generate_sample(N, p_2, range_2)

plt.hist(draw_2)

plt.show()


# Part C

print(H_1)

for N in [10, 100, 1000]:

    print('N is: '+ str(N))

    draw_1,  H_1 = generate_sample(N, p_1, range_1)

    plt.hist(draw_1)

    plt.show()

    print(calc_H_star(draw_1))

    draw_2,  H_2 = generate_sample(N, p_2, range_2)

    plt.hist(draw_2)

    plt.show()

    print(calc_H_star(draw_2))


# Part D

range_1 = np.arange(-2, 4, 0.001).tolist()

for N in [10, 100, 1000]:

    draw_1, H_1 = generate_sample(N, p_1, range_1)

    titles = []

    for index in [3, 1, 0.3]:

        title = 'N = ' + str(N) + ' , H = ' + str(index)

        titles.append(title)

        plot_kernel_density(draw_1, range_1, H_1 * index)

    plt.legend(titles)

    plt.show()