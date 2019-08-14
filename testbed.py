import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    steps = 10000
    eps = 0.1
    alpha = 0.1
    num_bandits = 10

    bandits = np.random.rand(num_bandits)
    estimates = np.zeros(num_bandits, dtype=np.float32)
    max_est = estimates.argmax()
    
    counter = 0
    best_choice = 0
    acc = 0
    estimate_history = [[] for _ in range(num_bandits)]
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
        estimates[best_choice] *= 1 - alpha
        estimates[best_choice] += alpha * np.random.normal(loc = bandits[best_choice])

        acc_history.append(acc/counter)
        estimate_history[best_choice].append(estimates[best_choice])

    
    print('Estimates {} '.format(estimates))
    print('True value {}'.format(bandits))
    #fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
    fig, axs = plt.subplots()
    axs.set_title('Distributions of estimates')
    axs.boxplot(estimate_history)
    fig, axs = plt.subplots()
    axs.set_title('Percent of right choice')
    axs.plot(range(len(acc_history)), acc_history)
    plt.show()
    
    

