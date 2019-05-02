#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
bin = np.arange(0, max(student_grades), 10)
plt.hist(student_grades, bins=bin, edgecolor='black')
ybin = np.arange(0, len(student_grades), 5)
plt.yticks(ybin)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
