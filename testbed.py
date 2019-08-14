import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    steps = 10000
    eps = 0.1
    alpha = 0.1
    num_bandits = 5

    bandits = np.random.rand(num_bandits)
    estimates = np.zeros(num_bandits, dtype=np.float32)
    max_est = estimates.argmax()
    
    counter = 0
    best_choice = 0
    acc = 0

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

        if counter % 500 == 0:
            print('Estimates {} '.format(estimates))
            print('Accuracy {} '.format(acc/counter))
    
    print('True value {}'.format(bandits))
    
    

