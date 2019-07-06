#!/usr/bin/env python3

import tensorflow as tf
early_stopping = __import__('7-early_stopping').early_stopping

if __name__ == '__main__':
    print(early_stopping(1.0, 1.1, 0.05, 16, 8))
    print(early_stopping(1.0, 1.05, 0.05, 16, 8))
    print(early_stopping(1.0, 1.05, 0.05, 16, 15))
