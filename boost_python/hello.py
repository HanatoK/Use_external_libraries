#!/usr/bin/env python3
import hello_ext

if __name__ == '__main__':
    print(hello_ext.greet())
    X = [1,2,3,4]
    ext_obj = hello_ext.World("xxx")
    print(ext_obj.func(X))