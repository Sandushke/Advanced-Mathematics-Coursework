#Question 1
# a)
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    if 0 > x >= -np.pi:
        return x ** 2 + 1

    if np.pi >= x >= 0:
        return x * np.exp(-x)

    if x < -np.pi:
        value = x+(2*np.pi)
        result = f(value)

    if x > np.pi:
        value = x - (2 * np.pi)
        result = f(value)

    return result

# Create a sequence between -4.pi to +4.pi
xValues = np.linspace(-4*np.pi, 4*np.pi, 1000)
period = 2 * np.pi + xValues

yValues = [f(x) for x in period]

# Plot the function
plt.title("Periodic function f(x)")
plt.xlabel=("y")
plt.ylabel =("x")
plt.plot(period, yValues)
plt.show()

# b)
import numpy as np
import sympy as sym

x = sym.symbols("x")
z = sym.symbols("z")

seq = np.empty(6, dtype=object)

func1 = x**2 +1
func2 = x*sym.exp(-1*x)

#equation for a0
a0 = (1 / (2 * sym.pi)) * (func1.integrate((x, -np.pi, 0)) + func2.integrate((x,0,np.pi)))
#equation for an
an = (1 / sym.pi) * (sym.integrate((func1 * sym.cos(z * x)), (x, -1 * np.pi, 0)) + sym.integrate((func2 * sym.cos(z * x)), (x, 0, np.pi)))
#equation for bn
bn = (1 / sym.pi) * (sym.integrate((func1 * sym.sin(z * x)), (x, -1 * np.pi, 0)) + sym.integrate((func2 * sym.sin(z * x)), (x, 0, np.pi)))

#Print the results
print("Result of a0: ", a0)
print("Result of an: ", an)
print("Result of bn: ", bn)

seq[0] = a0 / 2

#Print the an series
answer = 1
for a in range(1, 3):
    an = (1 / sym.pi) * (sym.integrate((func1 * sym.cos(a * x)), (x, -1 * np.pi, 0)) + sym.integrate((func2 * sym.cos(a * x)), (x, 0, np.pi)))
    seq[answer] = an
    answer+=1

#Print the bn series
answer = 3
for b in range(3, 6):
    bn = (1 / sym.pi) * (sym.integrate((func1 * sym.sin(b * x)), (x, -1 * np.pi, 0)) + sym.integrate((func2 * sym.sin(b * x)), (x, 0, np.pi)))
    seq[answer] = bn
    answer +=1

print("")
print("The fourier series of 1st six terms are: ", seq, end="")

# c)
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

x = sym.symbols("x")
z = sym.symbols("z")

seq = np.empty(150, dtype=object)
xRange = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
y = np.zeros([151, 1000])

func1 = x ** 2 +1
func2 = x * sym.exp(-x)

#equation for a0
a0 = (1 / (2 * sym.pi)) * (func1.integrate((x, -np.pi, 0)) + func2.integrate((x, 0, np.pi)))
#equation for anu
an = (1 / sym.pi) * (sym.integrate((func1 * sym.cos(z * x)), (x, -1 * np.pi, 0)) + sym.integrate((func2 * sym.cos(z * x)), (x, 0, np.pi)))
#equation for bn
bn = (1 / sym.pi) * (sym.integrate((func1 * sym.sin(z * x)), (x, -1 * np.pi, 0)) + sym.integrate((func2 * sym.sin(z * x)), (x, 0, np.pi)))

seq[0] = a0
f1 = sym.lambdify(x, seq[0], 'numpy')
y[0, :] = f1(xRange)

#Iterate over the harmonies(0th to 150th)
for i in range(1, 150):
    seq[i] = seq[i - 1] + an.subs(z, i) * sym.cos(i * x) + bn.subs(z, i) * sym.sin(i * x)
    #print (n+1, ":", ms[z])
    f1 = sym.lambdify(x, seq[i], 'numpy')
    y[i, :] = f1(xRange)


#Fit the ranges within the graph
plt.plot(xRange, y[1, :])
plt.plot(xRange, y[4, :])
plt.plot(xRange, y[149, :])
plt.plot(xRange, y[150, :])

#Plot the graph
plt.title("Harmonic graph")
plt.legend(["5", "10", "100", "func"])
plt.show()

# d)
import math

#To get the RMSE between f(x) and 1st harmonic value
absolvalue = [y[1, :]]
pred = [yValues]
meanSquareError = np.square(np.subtract(absolvalue, pred)).mean()
#Root mean square error
rootSquareMeanError = math.sqrt(meanSquareError)
print("")
print("")
print("RMSE of 1st harmonic :", rootSquareMeanError)


#To get the RMSE between f(x) and 5th harmonic value
absolvalue = [y[4, :]]
meanSquareError = np.square(np.subtract(absolvalue, pred)).mean()
#Root mean square error
rootSquareMeanError = math.sqrt(meanSquareError)
print("RMSE of 5th harmonic:", rootSquareMeanError)


#To get the RMSE between f(x) and 150th harmonic value
absolvalue = [y[149, :]]
meanSquareError = np.square(np.subtract(absolvalue, pred)).mean()
#Root mean square error
rootSquareMeanError = math.sqrt(meanSquareError)
print("RMSE of 150th harmonic:", rootSquareMeanError)


