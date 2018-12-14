#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    if arr1 is None or arr2 is None:
        return None
    res = []
    if len(arr1) == len(arr2):
        i = 0
        while i <= len(arr1) - 1:
            res.append(arr1[i]+arr2[i])
            i += 1
        return (res)
    else:
        return (None)
