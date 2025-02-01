import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _dtw_mk_ijpairlist(K, ei, ej):
    """ Backtrace from the matrix  K using the path endpoint indices [ei, ej] to obtain a list of matched point index pairs along the path.
    """
    ijpairlist = []
    i, j = ei, ej
    for _ in range(K.shape[0] + K.shape[1]):
        ijpairlist.append([i, j])
        if K[i, j] == 0: break
        i, j = i - K[i, j] // 10000, j - K[i, j] % 10000
    ijpairlist.reverse()
    return ijpairlist


def _dtw_walk_path(a, b, getlocaldis, istep=1, jstep=1,oba=False,obb=False,oea=False,oeb=False):
    """
    Uses a dynamic programming approach to find the optimal (minimum cumulative distance) matching path between sequences `a` and `b` under given constraints.
The function `getlocaldis(i, j)` represents the direct local distance between point `i` in sequence `a` and point `j` in sequence `b`.

Returns:
    mindis (float): The cumulative distance of the optimal matching path.
    ei, ej (int, int): The endpoint indices of the path in sequences `a` and `b`, respectively.
    D (numpy.ndarray): The cumulative distance matrix.
    K (numpy.ndarray): The backtracking matrix.

Backtracking matrix `K`:
    - `K[i, j] // 10000` gives the row backtracking step.
    - `K[i, j] % 10000` gives the column backtracking step.
    - A boundary value of `0` indicates the end of backtracking.

Path constraint conditions:
    - `istep`, `jstep`: The maximum step size allowed in the path for sequences `a` and `b`, respectively (default is 1, maintaining standard DTW continuity).
    - `oba`, `obb`, `oea`, `oeb`: Flags indicating whether to relax the start or end point constraints for sequences `a` or `b`.
      Default is `False`, maintaining the original DTW boundary condition (i.e., enforcing start-to-start and end-to-end matching).
    """
    na,nb=len(a),len(b)
    def Init_cumulative_and_trace_matrix(A, B, oba, obb):
        na, nb = len(A), len(B)
        D = np.zeros((na, nb))  # The cumulative distance matrix to be computed.
        K = np.zeros((na, nb)).astype('i')   # The backtracking matrix, storing backtracking directions as tuples.
        D[0, 0] = getlocaldis(0, 0)
        for i in range(1,na):
            if oba:
                D[i, 0] = getlocaldis(i, 0)

            else:
                D[i, 0] = D[i - 1, 0] + getlocaldis(i, 0)
                K[1:, 0] = 10000

        for j in range(1,nb):
            if obb:
                D[0, j] = getlocaldis(0, j)

            else:
                D[0, j] = D[0, j - 1] + getlocaldis(0, j)
                K[0, 1:] = 1
        return D, K

    D,K=Init_cumulative_and_trace_matrix(a,b,oba,obb)
    for i in range(1, na):
        for j in range(1, nb):
            si, sj = i - istep, j - jstep  # The forward-point positions in the cumulative distance matrix.
            if si < 0: si = 0  #
            if sj < 0: sj = 0
            d = D[si:i + 1, sj:j + 1]  # The submatrix of the cumulative distance matrix from the current point to the forward-point positions.
            k = np.argmin(d.ravel()[:-1])
            dm, dn = d.shape
            mi, mj = k // dn, k % dn
            ki = dm - 1 - mi  # The step size for movement in the row direction along the path.
            kj = dn - 1 - mj  # The step size for movement in the column direction along the path.
            D[i, j] = d[mi, mj] + getlocaldis(i, j)  # Set the current cumulative distance.
            K[i, j] = ki * 10000 + kj  # Encode the current path direction.
    ei, ej = na - 1, nb - 1  # end point
    if oea:
        ei = np.argmin(D[:, -1])
    if oeb:
        ej = np.argmin(D[-1, :])
    return D[ei, ej], ei, ej, D, K


def _dtw_mk_getlocaldis_func(disfunc, a, b, cache=True):
    """
    Returns a cached function for computing the distance between any two points in two sequences. Given two sequences of lengths na and nb, and a distance function disfunc(p0, p1), this function caches previously computed distances to avoid redundant calculations, improving efficiency.
    """

    def getlocaldis(i, j):
        if Dlocal[i, j] == np.inf:
            Dlocal[i, j] = disfunc(a[i], b[j])
        return Dlocal[i, j]

    na, nb = len(a), len(b)
    Dlocal = np.zeros((na, nb)) + np.inf  # The cached local distance matrix, where `Dlocal[i, j]` represents the direct distance between `a[i]` and `b[j]`.
    getlocaldis.Dlocal = Dlocal

    return getlocaldis
def vision_D(D,K,ijpairlist):
    ilist, jlist = zip(*ijpairlist)
    plt.figure(figsize=np.array(D.shape[::-1]) / max(D.shape) * 5, dpi=128)
    plt.imshow(D, origin='lower', cmap='binary')
    plt.colorbar()
    plt.plot(jlist, ilist, '.-r')
    plt.title('cumsum D & path')
    plt.xlabel('b')
    plt.ylabel('a')
    plt.tight_layout()
    plt.show()
    pass
def dtwf(a, b,w,oba=False,obb=False,oea=False,oeb=False):
    na, nb = len(a), len(b)
    kstep = 1

    def disAbs(a, b):
        if len(a) == 1:
            return abs(a[0] - b[0])
        ssuumm = 0

        the_cov = (abs(np.cov(a) - np.cov(b))) ** 0.5

        print(the_cov)

        for i in range(len(a)):
            ssuumm = abs(a[i] - b[i]) + ssuumm
        finally_sum = w * ssuumm + (1-w) * the_cov
        return finally_sum

    getlocaldis = _dtw_mk_getlocaldis_func(disAbs, a, b, cache=True)
    mindis_ei_ej_D_K = _dtw_walk_path(a, b, getlocaldis, istep=kstep,jstep=1,oba=oba,obb=obb,oea=oea,oeb=oeb)

    mindis, ei, ej, D, K = mindis_ei_ej_D_K

    ijpairlist = _dtw_mk_ijpairlist(K, ei, ej)

    t_ijpair = []
    t2_ijpair = []
    for i in range(len(ijpairlist)):
        t_ijpair.append(ijpairlist[i][0])
        t2_ijpair.append(ijpairlist[i][1])
    return ijpairlist, a, b, D[ei, ej],ei,ej,D,K


if __name__ == '__main__':
    a = [(1, 2),(3, 4),(5, 6),(6, 2),(3, 4), (5, 6), (6, 7), (1, 2), (2, 3), (1, 1), (1, 2), (3, 4), (4, 5), (1, 2)]
    b=[  (1,3), (2,4), (3,6), (2, 3),(5,6),  (3, 4), (5, 6), (6, 2), (3, 4), (5, 6), (6, 7), (1, 2), (2, 3), (1, 1),(1, 2),(3, 4),(4, 5),(1, 2),(1,6), (3,5),(6,7),(2,5),(6,3)]
    ijpairlist1, a1, b1, D1, ei, ej ,D,K= dtwf(a, b,0.3,oba=False,obb=False,oea=False,oeb=False)
    vision_D(D,K,ijpairlist1)
    print(ijpairlist1)
    print(D1)


