def main():
    import os
    import logging
    import datetime
    import numpy as np
    from sklearn.datasets import load_boston
    from sklearn.metrics import mean_absolute_error
    from gplearn_MLAA.genetic import SymbolicRegressor
    from sklearn.model_selection import train_test_split

    # load data
    seed = 0
    boston = load_boston()
    X_boston, y_boston = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, test_size=0.3, random_state=seed)
    # print baseline
    print("The baseline: {:.3f}".format(mean_absolute_error(y_test, np.repeat(y_test.mean(), len(y_test)))))

    # setup logger
    name = "_boston_"
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "logFiles/" + str(datetime.datetime.now().date()) + name + "log.txt")
    logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(name)s,%(message)s')

    edda_params = {"deme_size": 50, "p_gsgp_demes": 0.50, "maturation": 5, "p_mutation": 1.0, "gsm_ms": -1.0}

    est_gp = SymbolicRegressor(population_size=200,
                               generations=10, edda_params=edda_params, stopping_criteria=0.0,
                               edv_stopping_criteria=0.0, n_semantic_neighbors=0,
                               p_crossover=0.0, p_subtree_mutation=0.0,
                               p_gs_crossover=0.1, p_gs_mutation=0.90, gsm_ms=-1, semantical_computation=False,
                               p_hoist_mutation=0.0, p_point_mutation=0.0,
                               parsimony_coefficient=0.0, val_set=0.2,
                               verbose=1, n_jobs=1, log=True, random_state=seed)
    # print GS-GP
    est_gp.fit(X_train, y_train)
    if not est_gp.semantical_computation:
        print(est_gp._program)
        print("Generalization ability: {:.3f}".format(mean_absolute_error(y_test, est_gp.predict(X_test))))


if __name__ == "__main__":
    main()