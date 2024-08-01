# test_imports.py
import numpy
import pmdarima
print("Numpy version:", numpy.__version__)
print("Pmdarima version:", pmdarima.__version__)

import site
print(site.getsitepackages())
