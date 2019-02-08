import numpy as np
import growing_4 as grow
import pickle
import time
import pylab as pl

def grad_metaopt(model):
    start = time.time()
    # create model of crop
    plant = grow.plant_model(**model)
    # fix a cose big boss say this

    gammas = np.arange(0.001, 1, 0.05)
    print(len(gammas))
    errors = []
    i = 0
    for g in gammas:
        try:
            error = np.sum(plant.find_gradient_minimum(max_iteration_number=600,x_start=20, y_start=20, show=False, gamma=g))
        except Exception:
            error = 1000000
        finally:
            errors.append(error)
        # just save all error data for future science
        print('iteration number {} gamma is {} error is {}'.format(i, g, error))
        i += 1
    # save results
    end = time.time()
    print('time is {}'.format(end - start))
    fig = pl.figure()
    pl.plot(gammas, np.array(errors), '-or')
    pl.ylabel('error')
    pl.xlabel('gamma')
    pl.title(model)
    pl.grid()
    pl.show()
    # results = {'data': parameters, 'errors': errors, 'time': (end - start)}
    # with open('gradient_metaoptimize_{}.pickle'.format(j), 'wb') as f:
    #     pickle.dump(results, f)
    # with open('gradient_metaoptimize_double{}.pickle'.format(j), 'wb') as f:
    #     pickle.dump(results, f)


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
    # data = [p1, p2, p3, p4]
    # start work
    # growing triangle method metaoptimization
    for model in data:
        grad_metaopt(model)




