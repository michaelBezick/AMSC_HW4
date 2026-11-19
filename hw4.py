import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from functions import solve_committor, solve_committor_knn

# import dataset
dd = np.load("./FEM_pts_comm.npz")
pts = dd["pts"]  # FEM mesh points
q = dd["comm"]  # committor values at them

question_1 = True
question_2 = True
question_3 = True

epsilon_grid = np.logspace(-3,0, 20)

if question_1:
    # epsilon_grid = np.logspace(-4, 0, 30).tolist()

    errors = []

    for epsilon in tqdm(epsilon_grid):
        q_approx = solve_committor(epsilon, pts)
        abs_error = np.abs(q - q_approx)
        mean_abs_error = np.mean(abs_error)
        errors.append(mean_abs_error)

    print(errors)
    print(epsilon_grid)
    plt.figure()
    plt.title("MAE versus Epsilon")
    plt.plot(epsilon_grid, errors)
    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("MAE")
    plt.savefig("mae_vs_epsilon.pdf")

if question_2:
    # form K_\epsilon

    def get_K_epsilon(eps_par, xy):
        r2 = 9 * eps_par
        N = xy.shape[0]
        e0 = np.ones((N,))

        K = scipy.sparse.lil_matrix((N, N), dtype=float)

        for j in range(N):
            d2 = np.sum((xy - np.outer(e0, xy[j, :])) ** 2, axis=1)
            # Use ravel() to convert to 1D array of indices
            ind = np.argwhere(d2 - r2 < 0).ravel()
            K[j, ind] = np.exp(-d2[ind] / eps_par)

        # Convert to CSR format for efficient arithmetic operations
        K = K.tocsr()

        return K

    # epsilon_grid = np.logspace(-3, 1, 20).tolist()

    S_values = np.array(
        [
            float(get_K_epsilon(eps, pts).sum()) / pts.shape[0] ** 2
            for eps in epsilon_grid
        ]
    )

    log_S_values = np.array([np.log(s) for s in S_values])
    log_eps_values = np.array([np.log(eps) for eps in epsilon_grid])

    # calculate slopes
    dy = log_S_values[1:] - log_S_values[:-1]
    dx = log_eps_values[1:] - log_eps_values[:-1]
    slopes = dy / dx
    best_slope = np.argmax(slopes)
    best_log_eps = (log_eps_values[best_slope] + log_eps_values[best_slope + 1]) / 2
    best_eps = np.exp(best_log_eps)

    plt.figure()
    plt.plot(log_eps_values, log_S_values, marker='o')
    plt.axvline(x=best_log_eps, color="red", label=rf"$\epsilon^*={best_eps:.3g}$")
    plt.ylabel("log S(eps)")
    plt.xlabel("log eps")
    plt.legend()
    plt.savefig("question2.pdf")

    print(f"Best eps: {best_eps}")

    # epsilon_grid = np.logspace(-3, 1, 20).tolist()

    errors = []

    for epsilon in tqdm(epsilon_grid):
        q_approx = solve_committor(epsilon, pts)
        abs_error = np.abs(q - q_approx)
        mean_abs_error = np.mean(abs_error)
        errors.append(mean_abs_error)

    plt.figure()
    plt.title("MAE versus Epsilon")
    plt.plot(epsilon_grid, errors, marker='o')
    plt.axvline(x=best_eps, color="red", label=rf"$\epsilon^*={best_eps:.3g}$")


    errors = np.asarray(errors)
    if np.isfinite(errors).any():
        i_min = np.argmin(np.where(np.isfinite(errors), errors, np.inf))
        error_minimizer = epsilon_grid[i_min]
        plt.axvline(x=error_minimizer, color="green",
                    label=rf"MAE minimum at $\epsilon={error_minimizer:.3g}$")

    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig("question2_2.pdf")

    errors = np.asarray(errors)
    i_min = np.nanargmin(errors)

    print(f"Ksum predicted epsilon: {best_eps:.6g}")
    print(f"Smallest observed MAE: {errors[i_min]:.6g}")
    print(f"Epsilon at smallest MAE: {epsilon_grid[i_min]:.6g}")

if question_3:

    k_list = [2 ** p for p in range(5,10)]

    
    for i, k in enumerate(k_list):
        print(f"Iteration {i}/{len(k_list)}. Processing k: {k}")

        def get_K_epsilon_knn(k, eps_par, xy):
            N = xy.shape[0]
            e0 = np.ones((N,))

            K = scipy.sparse.lil_matrix((N, N), dtype=float)

            for j in range(N):

                d2 = np.sum((xy - np.outer(e0, xy[j, :])) ** 2, axis=1)
                # reserve K largest elements
                neighbors = np.argpartition(d2, k - 1)[:k]
                K[j, neighbors] = np.exp(-d2[neighbors] / eps_par)

            # Convert to CSR format for efficient arithmetic operations
            K = K.tocsr()

            return K

        # epsilon_grid = np.logspace(-3, 1, 20).tolist()

        S_values = np.array(
            [
                float(get_K_epsilon_knn(k, eps, pts).sum()) / pts.shape[0] ** 2
                for eps in epsilon_grid
            ]
        )

        log_S_values = np.array([np.log(s) for s in S_values])
        log_eps_values = np.array([np.log(eps) for eps in epsilon_grid])

        # calculate slopes
        dy = log_S_values[1:] - log_S_values[:-1]
        dx = log_eps_values[1:] - log_eps_values[:-1]
        slopes = dy / dx
        best_slope = np.argmax(slopes)
        best_log_eps = (log_eps_values[best_slope] + log_eps_values[best_slope + 1]) / 2
        best_eps = np.exp(best_log_eps)

        plt.figure()
        plt.plot(log_eps_values, log_S_values, marker='o')
        plt.axvline(x=best_log_eps, color="red", label=rf"$\epsilon^*={best_eps:.3g}$")
        plt.ylabel("log S(eps)")
        plt.xlabel("log eps")
        plt.title(f"k: {k}")
        plt.legend()
        plt.savefig(f"./question3/question3_k_{k}.pdf")

        print(f"Best eps: {best_eps}")

        # epsilon_grid = np.logspace(-3, -1, 20).tolist()

        errors = []

        for epsilon in tqdm(epsilon_grid):
            q_approx = solve_committor_knn(k, epsilon, pts)
            abs_error = np.abs(q - q_approx)
            mean_abs_error = np.mean(abs_error)
            errors.append(mean_abs_error)


        plt.figure()
        plt.title(f"MAE versus Epsilon, k={k}")
        plt.plot(epsilon_grid, errors, marker='o')
        plt.axvline(x=best_eps, color="red", label=rf"k sum $\epsilon^*={best_eps:.3g}$")


        errors = np.asarray(errors)
        if np.isfinite(errors).any():
            i_min = np.argmin(np.where(np.isfinite(errors), errors, np.inf))
            error_minimizer = epsilon_grid[i_min]
            plt.axvline(x=error_minimizer, color="green",
                        label=rf"MAE minimum at $\epsilon={error_minimizer:.3g}$")


        plt.xscale("log")
        plt.xlabel("Epsilon")
        plt.ylabel("MAE")
        plt.legend()
        plt.savefig(f"./question3/question3_2_k_{k}.pdf")

        errors = np.asarray(errors)
        i_min = np.nanargmin(errors)

        print(f"Ksum predicted epsilon: {best_eps:.6g}")
        print(f"Smallest observed MAE: {errors[i_min]:.6g}")
        print(f"Epsilon at smallest MAE: {epsilon_grid[i_min]:.6g}")
