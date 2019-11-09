import numpy as np
Y = np.array([1,0,0])

Y_pred1 = np.array([0.7,0.2,0.1]) #predict label0: 0.7...
# cross entropy
print(np.sum(-Y * np.log(Y_pred1)))
