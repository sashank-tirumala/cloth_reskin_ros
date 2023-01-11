import numpy as np
import matplotlib.pyplot as plt

errs = np.linspace(0.0099, -0.0099, 200)
# errs = np.sin(np.arange(-0.0099, 0.0099, 0.00001))
last_err = 0
total_err = 0
time_elapsed = 1
pid = []
p,i,d = 390, 0.25, 0.25

# for err in errs:
#     pid.append(p*err + i*total_err + d*(err- last_err)/time_elapsed)

#     # last_err = err
#     total_err += err
#     time_elapsed += 1

# for err in errs:
#     pid.append(np.cos(np.arccos(err)))

#     # last_err = err
#     total_err += err
#     time_elapsed += 1
pid = 255*np.sin(100*errs*np.pi/2)

plt.plot(errs, pid)
plt.show()