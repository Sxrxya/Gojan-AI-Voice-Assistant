import sys
import traceback

with open("debug_output.txt", "w") as f:
    try:
        import numpy
        f.write("Numpy imported successfully.\n")
    except Exception as e:
        f.write("Numpy import failed:\n")
        traceback.print_exc(file=f)
