# Question 5

# a)
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y = 1/(1 + np.exp(-x))

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# b)
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y = 1/(1 + np.exp(-x))
df = y *(1-y)

plt.plot(x, df)
plt.xlabel("x")
plt.ylabel("f(x)")

plt.show()

# c)
# a)
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0, 4* np.pi, 0.1)
y = np.sin(np.sin(2*x))

plt.plot(x, y)
plt.show()

# b)
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np

x = sym.symbols('x')
fx = -x ** 3 - 2 * x ** 2 + 3 * x + 10
f4 = sym.lambdify(x, fx, 'numpy')
z = np.arange(-10, 10, 0.1)
result = f4(z)

plt.plot(z, result)
plt.show()


# c)
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y = np.exp(-0.8*x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x)")

plt.show()

# d)
import matplotlib.pyplot as plt
import numpy as np
import math

x = np.arange(-4*math.pi, 4* math.pi, 0.1)
y = x**2 * np.cos(np.cos(2*x)) - 2 * np.sin(np.sin(x - 3.14/3))

plt.plot(x, y)
plt.show()

# e)
import numpy as np
import matplotlib.pyplot as plt

def g(x):
    if (-np.pi <= x < 0):
        return 2 * np.cos(x + np.pi/6)

    if (0 <= x < np.pi):
        return x * np.exp(-0.4 * x*2)

    if (x <= -np.pi):
        z = x + (2 * np.pi)
        result = g(z)

    if (x > np.pi):
        z = x - (2 * np.pi)
        result = g(z)

    return result


# for the range between -2*pi and 2*pi
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

# Calculate y values using the function g(x)
y = [g(result) for result in x]

# Plot the function
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('g(x)')
plt.show()


# d)
# i)
import numpy as np
import matplotlib.pyplot as plt


def logistic(x):
    return (1 / (1 + np.exp(-x)))

def f(x):
    return np.sin(np.sin(2*x))

x = np.linspace(-10, 10, 100)
y=logistic(f(x))

plt.plot(x, y)
plt.show()
#
# ii)
import numpy as np
import matplotlib.pyplot as plt


def logistic(x):
    return (1 / (1 + np.exp(-x)))

def f(x):
    return -x**3 - 2 * x**2 + 3*x + 10

x = np.linspace(-10, 10, 100)
y= logistic(f(x))

plt.plot(x, y)
plt.show()
#
# iii)
import numpy as np
import matplotlib.pyplot as plt


def logistic(x):
    return (1 / (1 + np.exp(-x)))

def f(x):
    return np.exp(-0.8*x)

x = np.linspace(-10, 10, 100)
y= logistic(f(x))

plt.plot(x, y)
plt.show()

# iv)
import numpy as np
import matplotlib.pyplot as plt

def logistic(x):
    return 1 / (1 + np.exp(-x))

def f(x):
    return x**2 * np.cos(np.cos(2*x)) - 2 * np.sin(np.sin(x - 3.14/3))

x = np.linspace(-10, 10, 100)
y= logistic(f(x))

plt.plot(x, y)
plt.show()

# v)
import numpy as np
import matplotlib.pyplot as plt

xVal=[]
for i in np.arange(-np.pi, np.pi):
    xVal.append(i * 0.1)

yValues=[]


def g(x):
    if -np.pi <= x < 0:
        return (1 / (1 + np.exp(-x))) * 2 * np.cos(x + np.pi/6)

    if 0 <= x < np.pi:
        return (1 / (1 + np.exp(-x))) * x * np.exp(-0.4 * x*2)

    if x <= -np.pi:
        value = x + (2 * np.pi)
        result = g(value)

    if x > np.pi:
        value = x - (2 * np.pi)
        result = g(value)

    return result

x = np.arange(-4 * np.pi, 4 * np.pi, 0.01)
y = [g(result) for result in x]

#Plot the graph
plt.plot(x, y, color="green")
plt.xlabel('x')
plt.ylabel('g(x)')
plt.show()

