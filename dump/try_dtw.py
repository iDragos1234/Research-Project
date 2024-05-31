import numpy as np
import matplotlib.pyplot as plt
import dtw


def swap(a, b) -> None:
    temp = a; a = b; b = temp; return


class Seq:
    def __init__(self, xs, ys=None, f=None) -> None:
        assert (ys is None) ^ (f is None)
        
        self.xs = xs
        self.ys = ys if f is None else f(xs)

        self.ps = np.array(list(zip(self.xs, self.ys)))


seqA = Seq(
    xs=np.linspace(1, 2.5, num=40),
    f=lambda xs: np.sin(xs) + np.random.uniform(-0.25, 0.25, size=40) / 10.0,
)

seqB = Seq(
    xs=np.linspace(1, 2.5, num=30),
    f=lambda xs: np.sin(xs) - 1 + np.random.uniform(-0.25, 0.25, size=30) / 10.0,
)


swap(seqA, seqB)


alignment = dtw.dtw(
    seqA.ps, seqB.ps,
    step_pattern='symmetric2',
)


plt.figure('DTW in action')
for i, j in zip(alignment.index1, alignment.index2):
    plt.plot(
        [seqA.xs[i], seqB.xs[j]], 
        [seqA.ys[i], seqB.ys[j]],
        'c'
    )


plt.scatter(seqA.xs, seqA.ys, marker='.')
plt.scatter(seqB.xs, seqB.ys, marker='.')
plt.plot(seqA.xs, seqA.ys, seqB.xs, seqB.ys)
plt.show()
