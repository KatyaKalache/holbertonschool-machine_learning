#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
apples = np.array(fruit[0])
bananas = np.array(fruit[1])
oranges = np.array(fruit[2])
peaches = np.array(fruit[3])
N = np.arange(3)
p0 = plt.bar(N, apples, width=0.5, color='red')
p1 = plt.bar(N, bananas, width=0.5, bottom=apples, color='yellow')
p2 = plt.bar(N, oranges, width=0.5, bottom=apples+bananas, color='#ff8000')
p3 = plt.bar(N, peaches, width=0.5, bottom=apples+bananas+oranges,
             color='#ffe5b4')
labels = ['apples', 'bananas', 'oranges', 'peaches']
plt.xticks(N, ('Farrah', 'Fred', 'Felicia'))
plt.yticks(np.arange(0, 90, 10))
plt.ylabel('Quantity of Fruit')
plt.legend((p0[0], p1[0], p2[0], p3[0]),
           (labels[0], labels[1], labels[2], labels[3]))
plt.title('Number of Fruit per Person')
plt.show()
