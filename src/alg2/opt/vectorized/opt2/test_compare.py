import numpy as np


class colors:  # You may need to change color settings
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'


# <----------- GLOBALS --------------------->
ALGOS = ["./out.txt"]

THRESHOLD = .05
PASS = True


# <------------FUNCTIONS-------------------->
def get_num_lines(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
        f.close()
    return count


def compare(PASS):
    for j, algo in enumerate(ALGOS):
        num_lines = get_num_lines(algo)
        num_lines = num_lines
        print("\nAlgorithm 2 opt2 (vectorized): ", end="\n")
        with open(algo, "r") as file:
            for i in range(int(num_lines / 5)):
                input_size = file.readline().split(" ")
                print(
                    f"Dimensions N: {input_size[0]}, M: {input_size[1]}, d: {input_size[2]} K: {input_size[3][:-1]} ---> ",
                    end="")
                file.readline()
                c_impl = np.array(file.readline().split(" "), dtype=np.float64)
                file.readline()
                python_impl = np.array(file.readline().split(" "), dtype=np.float64)
                abs_diff = np.abs(c_impl - python_impl)
                diff_comp = np.allclose(python_impl, c_impl, 1e-07, 1e-06)
                if np.all(diff_comp):
                    print(colors.GREEN + "PASS" + colors.ENDC)
                else:
                    PASS = False
                    print(colors.RED + "FAIL" + colors.ENDC)

    return PASS


if __name__ == '__main__':
    PASS = compare(PASS)
    if (PASS):
        print(colors.YELLOW + "All tests passed!" + colors.ENDC)
    else:
        print(colors.YELLOW + "All/some tests failed! Check output" + colors.ENDC)
