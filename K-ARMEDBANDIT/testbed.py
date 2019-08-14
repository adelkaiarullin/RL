import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Incorrect arguments')
        exit()

    state = sys.argv[1]
    if state == 'stationary':
        stationary = True
    else:
        stationary = False

    steps = 100000
    eps = 0.1
    alpha = 0.1
    num_bandits = 10

    bandit_means = np.random.rand(num_bandits)
    estimates = np.zeros(num_bandits, dtype=np.float32)
    std = 1.0
    std_step = 1e-3
    max_est = bandit_means.argmax()
    
    counter = 0
    best_choice = 0
    acc = 0
    estimate_history = [[0.0] for _ in range(num_bandits)]
    acc_history = []

    while counter < steps:
        counter += 1
        experiment = np.random.multinomial(1, [1-eps, eps])
        i = experiment.argmax()
        if i == 1:
            best_choice = np.random.randint(0, num_bandits)
        else:
            best_choice = estimates.argmax()
        if max_est == best_choice:
            acc += 1

        if stationary is True:
            R = np.random.normal(loc = bandit_means[best_choice])
            estimates[best_choice] += (R - estimates[best_choice]) / (len(estimate_history[best_choice]))
        else:
            estimates[best_choice] *= 1 - alpha
            estimates[best_choice] += alpha * np.random.normal(loc = bandit_means[best_choice], scale=std)
            std += std_step

        acc_history.append(acc/counter)
        estimate_history[best_choice].append(estimates[best_choice])

    
    print('Estimates {} '.format(estimates))
    print('True value {}'.format(bandit_means))

    fig, axs = plt.subplots()
    axs.set_title('Distributions of estimates')
    axs.boxplot(estimate_history)
    fig, axs = plt.subplots()
    axs.set_title('Percent of right choice where eps is {}'.format(eps))
    axs.plot(range(len(acc_history)), acc_history)
    plt.show()
    
    

