import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
# outfile = TemporaryFile()
file = "total_rewards.npz"


npzfile = np.load(file)
print(npzfile.files)
print(npzfile['arr_0'])

x = npzfile['arr_0']
y = npzfile['arr_1']

x= np.array(x)
y=np.array(y)

print(x)
print(y)

plt.plot(x,y)
plt.ylabel('average reward')
plt.xlabel('episode')
plt.show()