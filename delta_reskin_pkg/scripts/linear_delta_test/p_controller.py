import numpy as np
import matplotlib.pyplot as plt

errs = np.linspace(0.0099, -0.0099, 200)
last_err = 0
total_err = 0
time_elapsed = 1
pid = []
pid = 255 * np.sin(100 * errs * np.pi / 2)

plt.plot(errs, pid)
plt.show()
