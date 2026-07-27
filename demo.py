"""Minimal Dataset 2 comparison between LINGIC and HSIC."""

import numpy as np
from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma

from LINGIC import gauss_poly_HSIC_twice_pvalue


def main():
    # Fixed experiment configuration.
    num_samples = 120
    num_sources = 3
    student_df = 3
    repetitions = 200
    num_permutations = 199
    alpha = 0.05
    seed_start = 1000

    method_names = ("LINGIC", "HSIC")
    rejection_counts = {name: [0, 0] for name in method_names}

    # DGP: H0 uses independent sources; H1 uses shared sources.
    for condition, under_null in enumerate((True, False)):
        for repetition in range(repetitions):
            data_seed = seed_start + repetition
            np.random.seed(data_seed)

            sources_x = np.random.standard_t(
                student_df,
                size=(num_samples, num_sources),
            )
            if under_null:
                sources_y = np.random.standard_t(
                    student_df,
                    size=(num_samples, num_sources),
                )
            else:
                sources_y = sources_x

            weights_x = np.random.uniform(-3, 3, size=num_sources)
            weights_y = np.random.uniform(-3, 3, size=num_sources)
            x = sources_x @ weights_x
            y = sources_y @ weights_y
            xy = np.vstack([x, y]).T

            # Whiten only the shared-source alternative samples.
            if not under_null:
                xy = xy - xy.mean(axis=0)
                eigenvalues, eigenvectors = np.linalg.eigh(
                    np.cov(xy, rowvar=False)
                )
                whitening_matrix = (
                    eigenvectors
                    @ np.diag(1.0 / np.sqrt(eigenvalues))
                    @ eigenvectors.T
                )
                xy = xy @ whitening_matrix

            x = xy[:, 0].reshape(-1, 1)
            y = xy[:, 1].reshape(-1, 1)

            np.random.seed(10_000_000 + data_seed)
            pvalues = (
                gauss_poly_HSIC_twice_pvalue(
                    x,
                    y,
                    N_PERM=num_permutations,
                ),
                hsic_test_gamma(x, y, bw_method="mdbs")[1],
            )
            for name, pvalue in zip(method_names, pvalues):
                rejection_counts[name][condition] += pvalue <= alpha

    print(f"{'Method':<20} {'Type-I':>8} {'Power':>8}")
    print("-" * 38)
    for name in method_names:
        type_i, power = (
            count / repetitions for count in rejection_counts[name]
        )
        print(f"{name:<20} {type_i:>8.3f} {power:>8.3f}")


if __name__ == "__main__":
    main()
