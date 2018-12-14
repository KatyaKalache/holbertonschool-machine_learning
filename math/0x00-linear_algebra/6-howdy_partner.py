#!/usr/bin/env python3
# concatenates two arrays


def cat_arrays(arr1, arr2):
    """Appending one array to another"""
    res = []
    for i in arr1:
        res.append(i)
    for j in arr2:
        res.append(j)
    return (res)
