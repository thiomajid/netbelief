import numpy as np


def main():
    mask = np.random.binomial(n=1, p=0.15, size=(4, 8))
    print(mask[:, :, None, None])
    print("Hello from netbelief!")


if __name__ == "__main__":
    main()
