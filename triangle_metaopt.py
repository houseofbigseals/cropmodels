import numpy as np
import growing_4 as grow
import pickle
import time
from multiprocessing import Pool
import traceback
import sys




if __name__ == "__main__":
    # dict of parameters for model from experiment
    # dt is 20 minutes and it is our new time constant
    # parameters = {'a':0.0014, 'b':0.175, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}

    # make data as list of dicts
    p1 = {'a': 0.0014, 'b': 0.087, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1}
    p2 = {'a': 0.00055, 'b': 0.087, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
    p3 = {'a': 0.0014, 'b': 0.087, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
    p4 = {'a': 0.00055, 'b': 0.087, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1}
    p5 = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1}
    p6 = {'a': 0.00055, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
    p7 = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
    p8 = {'a': 0.00055, 'b': 0.175, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1}

    data = [p1, p2, p3, p4, p5, p6, p7, p8]
    # start work
    # growing triangle method metaoptimization
    j = 0
    for parameters in data:

        # create model of crop
        plant = grow.plant_model(**parameters)
        # fix a cose big boss say this
        a = 1
        betas = np.arange(1, 2, 1)
        gammas = np.arange(2, 3, 1)
        errors = []
        # kwargs = {"x_start": 20, "y_start": 20, "show": False,
        #           "alpha": 1, "beta": 0.5, "gamma": 2.9, "max_iteration_number": 600}
        line_data = []
        # create dicts with parameters
        print("CURRENT MODEL IS {} \n".format(parameters))
        for b in betas:
            for g in gammas:
                # just save all error data for science
                new_arg = {"x_start": 20, "y_start": 20, "show": False,
                           "alpha": 1, "beta": b, "gamma": g, "max_iteration_number": 600}
                print("hello, Im doing that {}".format(new_arg))
                try:
                    er = np.sum(plant.find_triangle_minimum(**new_arg))
                except Exception:
                    # traceback.print_exception(*sys.exc_info())
                    error = np.inf
                else:
                    error = er
                print("error is {}, beta is {}, gamma is {}".format(error, b, g))
                # line_data.append(new_arg)

        # def stupid_wrapper(dict):
        #     print("hello, Im doing that {}".format(dict))
        #     try:
        #         er = np.sum(plant.find_triangle_minimum(**dict))
        #     except Exception:
        #         traceback.print_exception(*sys.exc_info())
        #         error = np.inf
        #     else:
        #         error = er
        #     return error

        # map it all
        # start = time.time()
        # pool = Pool(2)
        # errors = pool.map(stupid_wrapper, line_data)

        # save results
        j = j+1
        # end = time.time()
        # print("time is {}".format(end - start))
        # result = {'model': parameters, 'results':{'data': line_data, 'errors': errors},
        #                                                                     'time': (end - start)}
        # with open('noized_triangle_metaoptimize_{}.pickle'.format(j), 'wb') as f:
        #     pickle.dump(result, f)
        # with open('noized_triangle_metaoptimize_double_{}.pickle'.format(j), 'wb') as f:
        #     pickle.dump(result, f)