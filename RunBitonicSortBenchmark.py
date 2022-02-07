from sys import argv
import subprocess
import numpy as np

import matplotlib.pyplot as plt


def getData(n: int):
    return np.random.randint(0, 100, n)


def main(argv) -> None:
    if (len(argv) != 2):
        print("Usage: {0} [BenchProgram]".format(argv[0]))
    prog = argv[1]

    min_size: int = 1 << 0
    max_size: int = 1 << 22

    data = getData(2 * max_size - min_size)

    data_str = ""
    i = 0
    l = min_size
    while l <= max_size:
        data_str += "{0} ".format(l) + " ".join(str(e) for e in data[i:i+l]) + "\n"
        i += l
        l *= 2
    r = subprocess.run(prog, input=data_str, shell=True, text=True, capture_output=True)
    print(r.stdout)

if __name__ == '__main__':
    main(argv)
