import numpy as np
import scipy


def solve_committor(epsilon, xy):

    Ndata = xy.shape[0]

    beta = 3

    # The numpy-type potential function
    def Vpot(xy):
        xa = -3
        ya = 3
        xb = 0
        yb = 4.5
        x = xy[:, 0]
        y = xy[:, 1]
        f = (1 - x) ** 2 + (y - 0.25 * x**2) ** 2 + 1
        g1 = 1 - np.exp(-0.125 * ((x - xa) ** 2 + (y - ya) ** 2))
        g2 = 1 - np.exp(-0.25 * (((x - xb) ** 2 + (y - yb) ** 2)))
        g3 = 1.2 - np.exp(-2 * ((x + 0) ** 2 + (y - 2) ** 2))
        g4 = 1 + np.exp(-2 * (x + 1.5) ** 2 - (y - 3.5) ** 2 - (x + 1) * (y - 3.5))
        v = f * g1 * g2 * g3 * g4
        return v

    # define sets A and B
    minimum_a = np.array([-3, 3])
    minimum_b = np.array([0, 4.5])

    ra = 0.5
    rb = 0.5

    # find indices of points in A and B and in OmegaAB
    # Remove training points lying in the sets A and B
    da2 = np.sum((xy - np.outer(np.ones((Ndata,)), minimum_a)) ** 2, axis=1)
    db2 = np.sum((xy - np.outer(np.ones((Ndata,)), minimum_b)) ** 2, axis=1)
    indA = np.argwhere(da2 < ra**2).ravel()
    indB = np.argwhere(db2 < rb**2).ravel()
    ind = np.arange(Ndata)
    ind_OmegaAB = np.setdiff1d(ind, np.union1d(indA, indB))

    # the construction of the generator matrix
    def make_generator(xy, eps_par, mu):
        r2 = 9 * eps_par  # radius squared of the neighborhood (removed semicolon)

        # construct the similarity matrix
        N = np.size(xy, axis=0)
        e0 = np.ones((N,))

        # Use lil_matrix for efficient construction with indexing
        K = scipy.sparse.lil_matrix((N, N), dtype=float)

        for j in range(N):
            d2 = np.sum((xy - np.outer(e0, xy[j, :])) ** 2, axis=1)
            # Use ravel() to convert to 1D array of indices
            ind = np.argwhere(d2 - r2 < 0).ravel()
            K[j, ind] = np.exp(-d2[ind] / eps_par)

        # Convert to CSR format for efficient arithmetic operations
        K = K.tocsr()

        # kernel density estimation
        c = 2 * np.pi * eps_par
        rho = np.sum(K, axis=1).A.ravel() / (N * c)  # .A converts to ndarray

        # right-normalize K
        D = scipy.sparse.diags(np.sqrt(mu) / rho, 0)
        K = K @ D

        # left-normalize K
        rho = np.sum(K, axis=1).A.ravel()  # row sums of K
        D = scipy.sparse.diags(1 / rho, 0)
        P = D @ K

        # define the generator matrix
        L = (P - scipy.sparse.eye(N, N, k=0)) / eps_par

        return L

    # compute the invariant measure (the target measure)
    mu = np.exp(-beta * Vpot(xy))
    # define the kernel bandwidth parameter: k(x,y) = exp(-||x-y||^2 / eps_par)
    eps_par = epsilon
    # construct the generator matrix
    L = make_generator(xy, eps_par, mu)
    q = np.zeros((Ndata,))
    q[indB] = 1.0
    rhs = -np.sum(L[:, indB], axis=1)
    L_OmegaAB = L[ind_OmegaAB, :][:, ind_OmegaAB]
    # find the committor
    q[ind_OmegaAB] = scipy.sparse.linalg.spsolve(L_OmegaAB, rhs[ind_OmegaAB])

    return q


def solve_committor_knn(k, eps_par, xy):

    Ndata = xy.shape[0]

    beta = 3

    # The numpy-type potential function
    def Vpot(xy):
        xa = -3
        ya = 3
        xb = 0
        yb = 4.5
        x = xy[:, 0]
        y = xy[:, 1]
        f = (1 - x) ** 2 + (y - 0.25 * x**2) ** 2 + 1
        g1 = 1 - np.exp(-0.125 * ((x - xa) ** 2 + (y - ya) ** 2))
        g2 = 1 - np.exp(-0.25 * (((x - xb) ** 2 + (y - yb) ** 2)))
        g3 = 1.2 - np.exp(-2 * ((x + 0) ** 2 + (y - 2) ** 2))
        g4 = 1 + np.exp(-2 * (x + 1.5) ** 2 - (y - 3.5) ** 2 - (x + 1) * (y - 3.5))
        v = f * g1 * g2 * g3 * g4
        return v

    # define sets A and B
    minimum_a = np.array([-3, 3])
    minimum_b = np.array([0, 4.5])

    ra = 0.5
    rb = 0.5

    # find indices of points in A and B and in OmegaAB
    # Remove training points lying in the sets A and B
    da2 = np.sum((xy - np.outer(np.ones((Ndata,)), minimum_a)) ** 2, axis=1)
    db2 = np.sum((xy - np.outer(np.ones((Ndata,)), minimum_b)) ** 2, axis=1)
    indA = np.argwhere(da2 < ra**2).ravel()
    indB = np.argwhere(db2 < rb**2).ravel()
    ind = np.arange(Ndata)
    ind_OmegaAB = np.setdiff1d(ind, np.union1d(indA, indB))

    # the construction of the generator matrix
    def make_generator(xy, eps_par, k, mu):

        # construct the similarity matrix
        N = np.size(xy, axis=0)
        e0 = np.ones((N,))

        # Use lil_matrix for efficient construction with indexing
        K = scipy.sparse.lil_matrix((N, N), dtype=float)

        for j in range(N):
            d2 = np.sum((xy - np.outer(e0, xy[j, :])) ** 2, axis=1)
            # reserve K largest elements
            neighbors = np.argpartition(d2, k - 1)[:k]
            K[j, neighbors] = np.exp(-d2[neighbors] / eps_par)

        # Convert to CSR format for efficient arithmetic operations
        K = K.tocsr()

        # need to symmetrize
        K = (K + K.T) / 2

        # kernel density estimation
        c = 2 * np.pi * eps_par
        rho = np.sum(K, axis=1).A.ravel() / (N * c)  # .A converts to ndarray

        # right-normalize K
        D = scipy.sparse.diags(np.sqrt(mu) / rho, 0)
        K = K @ D

        # left-normalize K
        rho = np.sum(K, axis=1).A.ravel()  # row sums of K
        D = scipy.sparse.diags(1 / rho, 0)
        P = D @ K

        # define the generator matrix
        L = (P - scipy.sparse.eye(N, N, k=0)) / eps_par

        return L

    # compute the invariant measure (the target measure)
    mu = np.exp(-beta * Vpot(xy))
    # define the kernel bandwidth parameter: k(x,y) = exp(-||x-y||^2 / eps_par)
    # construct the generator matrix
    L = make_generator(xy, eps_par, k, mu)
    q = np.zeros((Ndata,))
    q[indB] = 1.0
    rhs = -np.sum(L[:, indB], axis=1)
    L_OmegaAB = L[ind_OmegaAB, :][:, ind_OmegaAB]
    # find the committor
    q[ind_OmegaAB] = scipy.sparse.linalg.spsolve(L_OmegaAB, rhs[ind_OmegaAB])

    return q
