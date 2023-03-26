# Question 3

# a)
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(5 *-3.14, 7 * 3.14, 100)

def f(x):
    return x * np.cos(x/2)

plt.ylabel("f(x)")
plt.title("")
plt.plot(x, f(x))
plt.show()

# b)
from math import pi, sin
import math

def taylorCosSum(x, values):
    # Initialize to 1
    result = 1

    for i in range(1, values + 1):
        numerator = (-1) ** i * x ** (2 * i)
        denominator = math.factorial(2 * i)
        result += numerator/denominator

    return result

x = math.pi/2
values = 5
print(taylorCosSum(x, values))

#
# c)
import sympy
import matplotlib.pyplot as plt
import numpy as np

# Define the function
def f(x):
  return sympy.cos(x)

# series for the first 60 terms
x = sympy.Symbol('x')
taylorSeries = sympy.series(f(x), x0=sympy.pi/2, n=60).removeO()

range = np.linspace(-10, 10, 1000)

# Get the taylor series for the given range
yValues = [taylorSeries.evalf(subs={x: xValues}) for xValues in range]

plt.title("Taylor Series")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(range, yValues)
plt.show()

# d)
import numpy as np
import math

def cos(x):
    #taylor seies for x = pi/2
    n = 5  #Number of terms
    result = 1

    for i in range(n):
        result += (-1)**i * x**(2*i) / factorial(2*i)
    return result

def factorial(n):
    #Get the factorial
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

#approximate value at pi/3
x = math.pi/3
approx = cos(x)
print("Approximation : ", approx)

#deviation
actualValue = x * cos(math.pi/6)
approximate = cos(x)
absolError = abs(actualValue - approximate)
print("Deviation: ", absolError)

result = absolError/ actualValue
print("Deviation of the approximation from its actual value is: ", result)