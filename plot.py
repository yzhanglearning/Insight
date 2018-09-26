
from inference import *

from time import time
val_size = 100000
for trn_size in [50, 100, 500, 1000, 5000, 10000, 20000, 50000]:
    print('#'*50)
    print(f'Experiment with training size {trn_size}')
    start = time()
    experiment(trn_size, val_size)
    t = time() - start
    print(f'Time cost: {t}')


import matplotlib.pyplot as plt

best_acc = [0.84558, 0.87324, 0.91232, 0.9203, 0.93174, 0.93584, 0.94032, 0.94616]
sizes = [50, 100, 500, 1000, 5000, 10000, 20000, 50000]
plt.plot(sizes, best_acc)
plt.title('Evolution of performance when increasing the training size')
plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.show()

plt.plot(sizes, best_acc)
plt.title('Evolution of performance when increasing the training size, Zoom on the [0-10000] size zone')
plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.xlim([0, 10000])
plt.show()

plt.plot(np.log(sizes)/np.log(10), best_acc)
plt.title('Evolution of performance when increasing the training size, with log scale for size')
plt.xlabel('Training size (log)')
plt.ylabel('Accuracy')
plt.show()
