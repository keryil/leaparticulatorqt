# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# <codecell>

% matplotlib
inline
import matplotlib.pyplot as plt
import numpy as np

# <codecell>

data_x = np.linspace(0, 100, 100)
true_a = 5.
true_b = 2.
data_y = data_x * true_a + true_b
data_y = map(lambda x: x + np.random.normal(0, 75.), data_y)
print data_x, data_y

# <codecell>

plt.scatter(data_x, data_y)

# <codecell>

import pymc as pm

alpha = pm.Uniform("alpha", -10, 10)
beta = pm.Uniform("beta", -10, 10)
y = pm.Normal("y", alpha * data_x + beta)

# <codecell>

alpha

