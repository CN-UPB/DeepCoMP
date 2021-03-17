# plot utility function
import matplotlib.pyplot as plt

from deepcomp.env.util.utility import log_utility

fig = plt.figure()
data_rate = list(range(100))
qoe = [log_utility(dr) for dr in data_rate]
plt.plot(data_rate, qoe)

plt.xlabel('Data Rate')
plt.ylabel('QoE')

plt.show()
