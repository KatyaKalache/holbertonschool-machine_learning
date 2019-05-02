#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure()
st = fig.suptitle('All in One', fontsize='x-large')
grid = plt.GridSpec(3, 2, wspace=0.5, hspace=0.5)

plt.subplot(grid[0, 0])
plt.plot(y0, color='red')
plt.subplot(grid[0,1])
plt.scatter(x1, y1, color='m')
plt.xlabel('Height (in)', fontsize='x-small')
plt.ylabel('Weight (lbs)', fontsize='x-small')
plt.title('Men\'s Height vs Weight', fontsize='x-small')
plt.subplot(grid[1,0])
plt.plot(x2, y2)
plt.yscale(value='log')
plt.xlim(0, 28650)
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of C-14', fontsize='x-small')
plt.subplot(grid[1,1])
plt.plot(x3, y31, color='red', linestyle='--', label='C-14')
plt.plot(x3, y32, color='green', label='Ra-226')
plt.ylim(0, 1)
plt.xlim(0, 20000)
plt.legend()
plt.title('Exponential Decay of Radioactive Elements', fontsize="x-small")
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.subplot(grid[2,:2])
bin = np.arange(0, max(student_grades), 10)
plt.title('Project A', fontsize='x-small')
plt.hist(student_grades, bins=bin, edgecolor='black')
ybin = np.arange(0, len(student_grades), 10)
plt.yticks(ybin)
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.show()
