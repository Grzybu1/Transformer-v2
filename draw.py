import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("trained/results_loss.txt",
                    delimiter=",", dtype=float)

plt.plot(data)
for i in range(1, 10):
    plt.axvline(x = i*400, color = 'r', linestyle='dotted')
plt.yscale('log')
plt.xlabel('Numer partii')
plt.ylabel('Strata')
plt.show()