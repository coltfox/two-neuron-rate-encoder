import numpy as np
import matplotlib.pyplot as plt

from rate_encoder import RateEncoder, plot_spikes

T = 20
DT = 0.1
N_STEPS = int(T / DT)

timesteps = np.arange(0, T, DT)
inp = np.random.normal(loc=-0.5, size=N_STEPS)

# Separate positive and negative signals
pos_signal = inp.copy()
pos_signal[pos_signal < 0] = 0

neg_signal = inp.copy()
neg_signal[neg_signal > 0] = 0

# Encode each
encoder_pos = RateEncoder(n_neurons=1)
encoder_neg = RateEncoder(n_neurons=1, encoders=-1)

spikes_pos = encoder_pos.encode(pos_signal).reshape(-1, 1)
spikes_neg = encoder_neg.encode(neg_signal).reshape(-1, 1)

# Plot input signal
plt.figure()
plt.title("Input Signal")
plt.xlabel("t")
plt.ylabel("x")
plt.plot(timesteps, inp)

# Plot positive and negative portions of the input signal
plt.figure()
plt.title("Input Signal (+)")
plt.xlabel("t")
plt.ylabel("x")
plt.plot(timesteps, pos_signal)

plt.figure()
plt.title("Input Signal (-)")
plt.xlabel("t")
plt.ylabel("x")
plt.plot(timesteps, neg_signal)

# Plot spikes
plt.figure()
plt.title("Spikes Positive")
plt.xlabel("t")
plt.ylabel("S")
plot_spikes(timesteps, spikes_pos)

plt.figure()
plt.title("Spikes Negative")
plt.xlabel("t")
plt.ylabel("S")
plot_spikes(timesteps, spikes_neg)

plt.show()
