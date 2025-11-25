import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, Matern # gp only

params = {
    "pce": [
        {
            "order": list(range(1, 6)),
            "mindex_type": ["total_order", "hyperbolic"],
            "fit_type": ["linear", "ElasticNetCV"],
            "fit_params": [
                {
                    "alphas": np.logspace(-8, 4, 20),
                    "max_iter": 100000,
                    "tol": 5e-2,
                }
            ],
        },
        {
            "order": list(range(1, 6)),
            "mindex_type": ["total_order", "hyperbolic"],
            "fit_type": ["LassoCV"],
            "fit_params": [
                {
                    "alphas": np.logspace(-8, 4, 20),
                    "max_iter": 500000,
                    "tol": 2.5e-2,
                }
            ],
        }
    ],

    "mlp": {
        "hidden_layer_sizes": [
            (64,) * 2,
            (64,) * 4,
            (128,) * 2,
            (128,) * 6,
            (256,) * 4,
            (512,) * 2,
            (512,) * 4,
            (1024,) * 1,
        ],
        "solver": ["sgd", "adam"],
        "activation": ["relu"],
        "max_iter": [10000],
        "batch_size": ["auto"],
        "learning_rate": ["invscaling", "adaptive"],
        # "alpha": [1e-3, 1e-4],
        # "tol": [1e-3],
        # "random_state": [0],
    },

    "rf": {
        "n_estimators": [200, 500, 1000],
        "max_features": ["sqrt", "log2", 1],
        "max_depth": [2, 5, 10, 15],
    },

    "gpr": {
        "kernel": [
            1.0 * RBF(0.1) + 0.01**2 * WhiteKernel(0.01),
            1.0 * RBF(0.1) + 0.01**2 * WhiteKernel(0.01) + 1.0 * DotProduct(0.1),
            1.0 * Matern(length_scale=0.1, nu=1.5) + 0.1**2 * WhiteKernel(0.1),
        ],
        "alpha": [1e-10],
        "optimizer": ["fmin_l_bfgs_b"],
        "n_restarts_optimizer": [10],
        "random_state": [0, 99],
    }
}
