import numpy as np

def add(x):
	np.random.shuffle(x)

x = np.arange(10)
add(x)
print(x)