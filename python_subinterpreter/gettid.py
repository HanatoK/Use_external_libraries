#!/usr/bin/env python3
import threading

def gettid():
    return threading.get_native_id()
