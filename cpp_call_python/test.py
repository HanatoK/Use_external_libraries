#!/usr/bin/env python3
import math

def test(person):
    return "Hello, " + person;

def calc_xxx(X):
    print(X)
    s = 0
    for i in range(0, len(X)):
        s += math.sin(X[i])
    return s
