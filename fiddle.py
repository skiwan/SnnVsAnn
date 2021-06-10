import pywt
import numpy as np
import matplotlib.pyplot as plt

def average_signal(data, stepsize):
    y = data.shape[0]
    x = data.shape[1]
    if x % stepsize != 0:
        print(f'stepsize {stepsize} not appropriate for input data lenght')
        return data
    prepared = data.reshape((y,int(x/stepsize),stepsize))
    return np.mean(prepared, axis=-1)

# sampling frequ 250 hz, 160 trials t3 - 7.5 s
# no data prep just cwt, morlet center is 0.8125 big T is 0.04
# scale is 1-250, 1000 time sample points 4/15 and 19/30 hz endgoal 22-1000 (22 frequency sample points) reduced to 22-200 by taking an average every 5


# prep random channel data
dt = 0.0250

scales = np.arange(4,15,0.5)
scales2 = np.arange(19,30,0.5)

#trials x timesteps x channels
whole_data = np.random.rand(256, 1000, 6)

cwt_trials = []
for t in range(whole_data.shape[0]):
    trial = whole_data[t]
    # make single channel accesable
    trial = np.swapaxes(trial,0,1)
    for c in range(trial.shape[0]):
        channel_data = trial[c]


channel_data = np.random.rand(1000)


coefs, freq = pywt.cwt(channel_data, scales, 'morl')
coefs2, freq = pywt.cwt(channel_data, scales2, 'morl')

coefs = average_signal(coefs, 5)
coefs2 = average_signal(coefs2, 5)

input = np.concatenate((coefs, coefs2), axis=0)

plt.imshow(input, cmap='Greys')
plt.show()

print('done')
