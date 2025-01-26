import matplotlib.pyplot as plt
import numpy as np


def main():
    # https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    n = 10000
    i = np.arange(0, n)
    theta = 4 * np.pi * i / (1 + 5**0.5)
    cos_phi = 1 - 2 * i / n
    phi = np.arccos(cos_phi)
    sin_phi = np.sin(phi)
    x, y, z = np.cos(theta) * sin_phi, np.sin(theta) * sin_phi, cos_phi

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, marker="o")
    plt.show()


if __name__ == "__main__":
    main()
