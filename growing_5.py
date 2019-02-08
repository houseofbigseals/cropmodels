# author: cratch_rider
# growing parameter optimization mark II
from numba import jit
import numpy as np
import pylab as pl
import matplotlib.patches
import matplotlib.path
import random


def draw_triangle(axes, p1, p2, p3_, color_='r'):
    polygon_1 = matplotlib.patches.Polygon([p1, p2, p3_], fill=False, color=color_)
    axes.add_patch(polygon_1)



class plant_model:
    '''
    it realizes all parameters and methods of plantation
    '''


    def __init__(self, **kwargs):
        # dict of parameters for model from standart experiment
        # dt is 20 minutes and it is our new time constant
        # that is parameters for default
        default_params = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
        # param will be our dict of parameters
        self.param = {}
        for kw in default_params.keys():
            if kw in kwargs.keys():
                self.param.update({kw: kwargs[kw]})
            else:
                self.param.update({kw: default_params[kw]})

    @jit
    def noise_scalar_calc_functional(self, x1, x2, t):
        '''
        calculates one value of functional F from point (x1, x2, t) with noise
        '''
        a = self.param['a']
        b = self.param['b']
        k = self.param['k']
        e1 = self.param['e1']
        e2 = self.param['e2']

        f = e1 * (x1 - (a * t + k * np.sin(b * t))) * (x1 - (a * t + k * np.sin(b * t))) + \
            e2 * (x2 - (a * t + k * np.sin(b * t))) * (x2 - (a * t + k * np.sin(b * t)))
        psy = random.uniform(-0.01 * f, 0.01 * f)
        f = f + psy
        return f

    @jit
    def noise_vector_calc_fuctional(self, X1, X2, t):
        '''
        for calculating grid in moment t- calculates vector of values F
        from vector of points(x1, x2) and time point t with noise
        '''
        N = len(X1)

        A = ((np.ones(N)) * self.param['a']).reshape((np.shape(X1)))
        B1 = ((np.ones(N)) * self.param['b']).reshape((np.shape(X1)))
        B2 = B1
        E1 = ((np.ones(N)) * self.param['e1']).reshape((np.shape(X1)))
        E2 = ((np.ones(N)) * self.param['e2']).reshape((np.shape(X1)))
        K = ((np.ones(N)) * self.param['k']).reshape((np.shape(X1)))
        T = ((np.ones(N)) * (t + self.param['dt'])).reshape((np.shape(X1)))
        R1 = (X1 - (A * T + K * np.sin(B1 * T)))
        R2 = (X2 - (A * T + K * np.sin(B2 * T)))
        F = E1 * R1 * R1 + E2 * R2 * R2
        Max = np.mean(F)
        psy = np.random.uniform(-0.01 * Max, 0.01 * Max, N)
        PSY = psy.reshape(np.shape(X1))
        F = F + PSY
        return F

    @jit
    def vector_calc_fuctional(self, X1, X2, t):
        '''
        for calculating grid in moment t- calculates vector of values F
        from vector of points(x1, x2) and time point t
        '''
        N = len(X1)

        A = ((np.ones(N)) * self.param['a']).reshape((np.shape(X1)))
        B1 = ((np.ones(N)) * self.param['b']).reshape((np.shape(X1)))
        B2 = B1
        E1 = ((np.ones(N)) * self.param['e1']).reshape((np.shape(X1)))
        E2 = ((np.ones(N)) * self.param['e2']).reshape((np.shape(X1)))
        K = ((np.ones(N)) * self.param['k']).reshape((np.shape(X1)))
        T = ((np.ones(N)) * (t + self.param['dt'])).reshape((np.shape(X1)))
        R1 = (X1 - (A * T + K * np.sin(B1 * T)))
        R2 = (X2 - (A * T + K * np.sin(B2 * T)))
        F = E1 * R1 * R1 + E2 * R2 * R2
        return F

    @jit
    def scalar_calc_functional(self, x1, x2, t):
        '''
        calculates one value of functional F from point (x1, x2, t)
        '''
        a = self.param['a']
        b = self.param['b']
        k = self.param['k']
        e1 = self.param['e1']
        e2 = self.param['e2']

        f = e1 * (x1 - (a * t + k * np.sin(b * t))) * (x1 - (a * t + k * np.sin(b * t))) + \
            e2 * (x2 - (a * t + k * np.sin(b * t))) * (x2 - (a * t + k * np.sin(b * t)))
        return f


    def find_gradient_minimum(self, max_iteration_number=100, x_start=20, y_start=20, show=True, gamma=0.09):
        '''
        find minimum of functional on our model parameters
        with simple gradient method
        '''
        # Make data.
        # X is E W/m^2
        # Y is T Celsius
        E = np.arange(-100, 200, 0.25)
        T = np.arange(-30, 40, 0.05)
        X, Y = np.meshgrid(E, T)
        d1x, d2x = np.shape(X)
        d1y, d2y = np.shape(Y)
        X_ = X.reshape((d1x * d2x, 1))
        Y_ = Y.reshape((d1y * d2y, 1))
        t = 1  # start time
        # Z_ = self.vector_calc_fuctional(X_, Y_, t)
        z_min = np.array((0,))
        y_min = np.array((0,))
        x_min = np.array((0,))
        x_grad = np.array([x_start])
        y_grad = np.array([y_start])
        error = np.array((0,))
        # gradient parameters
        dx1 = 0.1
        dx2 = 0.1
        for i in range(0, max_iteration_number):
            # main cycle for gradient iterations
            ZZ_ = self.vector_calc_fuctional(X_, Y_, t)
            # find min of ZZ_
            n_min = np.argmin(ZZ_)
            z_min = np.append(z_min, np.min(ZZ_))
            x_min = np.append(x_min, X_[n_min])
            y_min = np.append(y_min, Y_[n_min])
            # gradient step
            xg, yg = self.gradient_step(x_grad[i], y_grad[i], t, dx1, dx2, gamma)
            x_grad = np.append(x_grad, xg)
            y_grad = np.append(y_grad, yg)
            # squared error calculation
            Fmin = np.min(ZZ_)
            Fnow = self.scalar_calc_functional(xg, yg, t)
            er = (Fmin - Fnow)
            error = np.append(error, er)
            if (i % 5 == 0 and show):
                # plot all steps
                Z = ZZ_.reshape((len(T), len(E)))
                fig = pl.figure()
                cs = pl.contour(X, Y, Z, 20)
                pl.clabel(cs, fmt='%.1f', colors="black")
                # add a color bar which maps values to colors.
                fig.colorbar(cs, shrink=0.5, aspect=5)
                # plotting min point trajectory
                pl.plot(x_min, y_min, "-b")
                # plotting gradient steps trajectory
                pl.plot(x_grad, y_grad, "-or")
                # plotting where is min point now
                pl.plot(X_[n_min], Y_[n_min], "sk")
                pl.grid()
                pl.show()
            # print(x_grad[i], y_grad[i], t)
            # print(x_min[i],  y_min[i])
            t = t + 3 * self.param['dt']
        # at the end of all iterations:
        if (show):
            Z = ZZ_.reshape((len(T), len(E)))
            fig = pl.figure()
            cs = pl.contour(X, Y, Z, 20)
            pl.clabel(cs, fmt='%.1f', colors="black")
            # Add a color bar which maps values to colors.
            fig.colorbar(cs, shrink=0.5, aspect=5)
            pl.plot(x_min, y_min, "-b")
            pl.plot(x_grad, y_grad, "-or")
            pl.plot(X_[n_min], Y_[n_min], "sk")
            pl.grid()
            pl.savefig("growing.png")
            pl.show()
        return error


    def gradient_step(self, x1, x2, t, dx1, dx2, gamma):

        # dt = 20 minuts and it is one
        dt = self.param['dt']
        f0 = self.scalar_calc_functional(x1, x2, t)
        f1 = self.scalar_calc_functional(x1 + dx1, x2, t + dt)
        f2 = self.scalar_calc_functional(x1, x2 + dx2, t + 2 * dt)
        x1_next = x1 - gamma * ((f1 - f0) / np.float(dx1))
        x2_next = x2 - gamma * ((f2 - f0) / np.float(dx2))
        return x1_next, x2_next



    def conj_gradient_step(self, start, x1, x2, dx1, dx2, df1, df2, S1_last, S2_last, lam, t):
        # x1 x2 - point to step
        # dx1, dx2 - steps to find gradient
        # df1, df2 - last gradient components
        # S_last - last S component for conjugate step
        dt = self.param['dt']

        if (start):
            # it means that it is first iteration
            f0 = self.scalar_calc_functional(x1, x2, t)
            f1 = self.scalar_calc_functional(x1 + dx1, x2, t + dt)
            f2 = self.scalar_calc_functional(x1, x2 + dx2, t + 2 * dt)
            df1_next = ((f1 - f0) / np.float(dx1))
            df2_next = ((f2 - f0) / np.float(dx2))
            S1_next = -1 * df1_next
            S2_next = -1 * df2_next
            # find min l
            l = np.arange(0.000001, 1, 0.0005)  # what do this numbers mean?
            x1vec = np.ones(np.shape(l)[0]) * x1
            x2vec = np.ones(np.shape(l)[0]) * x2
            S1vec = np.ones(np.shape(l)[0]) * S1_next
            S2vec = np.ones(np.shape(l)[0]) * S2_next
            func = self.vector_calc_fuctional(x1vec + l * S1vec, x2vec + l * S2vec, t + 2 * dt)
            n_min = np.argmin(func)
            lam = l[n_min]
            print('lambda min is {}'.format(lam))
            x1_next = x1 + lam * S1_next
            x2_next = x2 + lam * S2_next
            return x1_next, x2_next, df1_next, df2_next, S1_next, S2_next
        else:
            f0 = self.scalar_calc_functional(x1, x2, t)
            f1 = self.scalar_calc_functional(x1 + dx1, x2, t + dt)
            f2 = self.scalar_calc_functional(x1, x2 + dx2, t + 2 * dt)

            df1_next = ((f1 - f0) / np.float(dx1))
            df2_next = ((f2 - f0) / np.float(dx2))
            w = (df1_next * df1_next + df2_next * df2_next) / np.float((df1 * df1 + df2 * df2))
            S1_next = -1 * df1_next + w * S1_last
            S2_next = -1 * df2_next + w * S2_last
            # find min l
            l = np.arange(0.000001, 1, 0.0005)
            x1vec = np.ones(np.shape(l)[0]) * x1
            x2vec = np.ones(np.shape(l)[0]) * x2
            S1vec = np.ones(np.shape(l)[0]) * S1_next
            S2vec = np.ones(np.shape(l)[0]) * S2_next
            func = self.vector_calc_fuctional(x1vec + l * S1vec, x2vec + l * S2vec, t + 2 * dt)
            n_min = np.argmin(func)
            lam = l[n_min]
            print('lambda min is {}'.format(lam))
            x1_next = x1 + lam * S1_next
            x2_next = x2 + lam * S2_next
            return x1_next, x2_next, df1_next, df2_next, S1_next, S2_next


    def find_conj_gradient_minimum(self, max_iteration_number=100, x_start=20, y_start=20, show=True):
        '''
        find minimum of functional on our model parameters with
        conjugate gradient method
        '''
        # we calling step function depending on type_
        # Make data.
        # X is E W/m^2
        # Y is T Celsius
        E = np.arange(-100, 200, 0.25)
        T = np.arange(-30, 40, 0.05)
        X, Y = np.meshgrid(E, T)
        d1x, d2x = np.shape(X)
        d1y, d2y = np.shape(Y)
        X_ = X.reshape((d1x * d2x, 1))
        Y_ = Y.reshape((d1y * d2y, 1))
        t = 1  # start time
        z_min = np.array((0,))
        y_min = np.array((0,))
        x_min = np.array((0,))
        x_grad = np.array([x_start])
        y_grad = np.array([y_start])
        error = np.array((0,))

        # conjugate gradient parameters
        dx1 = 0.1
        dx2 = 0.1
        lam = 0.001
        S1 = 0
        S2 = 0
        df1 = 0
        df2 = 0
        for i in range(0, max_iteration_number):
            # main cycle for gradient iterations
            ZZ_ = self.vector_calc_fuctional(X_, Y_, t)
            # find min of ZZ_
            n_min = np.argmin(ZZ_)
            z_min = np.append(z_min, np.min(ZZ_))
            x_min = np.append(x_min, X_[n_min])
            y_min = np.append(y_min, Y_[n_min])
            # gradient step
            xg, yg, df1_n, df2_n, S1_n, S2_n = self.conj_gradient_step((i == 0), x_grad[i], \
                                                                       y_grad[i], dx1, dx2, df1, df2, S1, S2, lam, t)
            x_grad = np.append(x_grad, xg)
            y_grad = np.append(y_grad, yg)
            # remember new parameters
            S1 = S1_n
            S2 = S2_n
            df1 = df1_n
            df2 = df2_n
            # error
            F_now = self.scalar_calc_functional(xg, yg, t)
            F_min = np.min(ZZ_)
            er = (F_now - F_min)
            error = np.append(error, er)
            if (i % 10 == 0 and show):
                print(i)
                # plot all steps
                Z = ZZ_.reshape((len(T), len(E)))
                fig = pl.figure()
                cs = pl.contour(X, Y, Z, 20)
                pl.clabel(cs, fmt='%.1f', colors="black")
                # add a color bar which maps values to colors.
                fig.colorbar(cs, shrink=0.5, aspect=5)
                # plotting min point trajectory
                pl.plot(x_min, y_min, "-b")
                # plotting gradient steps trajectory
                pl.plot(x_grad, y_grad, "-or")
                # plotting where is min point now
                pl.plot(X_[n_min], Y_[n_min], "sk")
                pl.grid()
                pl.show()
            # print(x_grad[i], y_grad[i], t)
            # print(x_min[i],  y_min[i])
            t = t + 5 * self.param['dt']

        # at the end of all iterations:
        if (show):
            Z = ZZ_.reshape((len(T), len(E)))
            fig = pl.figure()
            cs = pl.contour(X, Y, Z, 20)
            pl.clabel(cs, fmt='%.1f', colors="black")
            # Add a color bar which maps values to colors.
            fig.colorbar(cs, shrink=0.5, aspect=5)
            pl.plot(x_min, y_min, "-b")
            pl.plot(x_grad, y_grad, "-or")
            pl.plot(X_[n_min], Y_[n_min], "sk")

            pl.grid()
            pl.savefig("growing.png")
            pl.show()
        return error

    def find_triangle_minimum(self, alpha, beta, gamma, max_iteration_number=40, x_start=20,
                              y_start=20, show=True):
        '''
        find minimum of functional on our model parameters with
        triangle deformation method
        '''
        # Make data.
        # X is E W/m^2
        # Y is T Celsius
        E = np.arange(-100, 200, 0.25)
        T = np.arange(-30, 40, 0.05)
        X, Y = np.meshgrid(E, T)
        d1x, d2x = np.shape(X)
        d1y, d2y = np.shape(Y)
        X_ = X.reshape((d1x * d2x, 1))
        Y_ = Y.reshape((d1y * d2y, 1))
        t = 1  # start time
        # Z_ = self.vector_calc_fuctional(X_, Y_, t)
        z_min = np.array((0,))
        y_min = np.array((0,))
        x_min = np.array((0,))
        # np array for quadratic errors of each iteration
        error = np.array((0,))
        # set start triangle
        triangles = []
        x01 = np.array((x_start, y_start))
        x02 = np.array((x_start + 2, y_start))
        x03 = np.array((x_start, y_start + 2))
        triangles.append([x01, x02, x03])

        for i in range(0, max_iteration_number):
            print("step number {}".format(i))
            # main cycle for optimization iterations
            print("vector calc")
            ZZ_ = self.noise_vector_calc_fuctional(X_, Y_, t)
            print("end vector calc")
            # find min of ZZ_
            n_min = np.argmin(ZZ_)
            z_min = np.append(z_min, np.min(ZZ_))
            x_min = np.append(x_min, X_[n_min])
            y_min = np.append(y_min, Y_[n_min])
            # triangle step
            print("triangle step")
            x1, x2, x3, x_mid = self.triangle_step(triangles[i][0],
                                                   triangles[i][1], triangles[i][2], t, alpha, beta, gamma)
            print("triangle step done")
            triangles.append([np.array(x1), np.array(x2), np.array(x3)])
            print("scalar calc")
            F_now = self.scalar_calc_functional(x_mid[0], x_mid[1], t)
            print(" end scalar calc")
            # F_min = self.scalar_calc_functional(X_[n_min], Y_[n_min], t)
            F_min = np.min(ZZ_)
            er = (F_now - F_min)
            error = np.append(error, er)
            if (i % 2 == 0 and show == True):

                # plot all steps

                Z = ZZ_.reshape((len(T), len(E)))

                fig = pl.figure()

                cs = pl.contour(X, Y, Z, 20)
                pl.clabel(cs, fmt='%.1f', colors="black")
                # add a color bar which maps values to colors.
                fig.colorbar(cs, shrink=0.5, aspect=5)
                # plotting min point trajectory

                pl.plot(x_min, y_min, "-b")
                # plotting triangles
                axes = pl.gca()

                for tr in triangles:
                    draw_triangle(axes, tr[0], tr[1], tr[2])
                # draw last triangle

                draw_triangle(axes, triangles[i + 1][0], triangles[i + 1][1], triangles[i + 1][2], color_='g')
                # plotting where is min point now
                pl.plot(X_[n_min], Y_[n_min], "sk")
                pl.grid()
                pl.show()

            print(triangles[i][0], triangles[i][1], triangles[i][2], t)
            print(x_min[i], y_min[i])
            print(F_now, F_min)
            t = t + 5 * self.param['dt']
        # at the end of all iterations:
        if (show == True):
            Z = ZZ_.reshape((len(T), len(E)))
            fig = pl.figure()
            cs = pl.contour(X, Y, Z, 20)
            pl.clabel(cs, fmt='%.1f', colors="black")
            # Add a color bar which maps values to colors.
            fig.colorbar(cs, shrink=0.5, aspect=5)
            pl.plot(x_min, y_min, "-b")
            # plotting triangles
            axes = pl.gca()
            for tr in triangles:
                draw_triangle(axes, tr[0], tr[1], tr[2])
            pl.plot(X_[n_min], Y_[n_min], "sk")
            pl.grid()
            pl.savefig("triangles.png")
            pl.show()
        return error


    def triangle_step(self, p1, p2, p3, t, alpha, beta, gamma):
        # print("triangle step")
        dt = self.param['dt']
        f1 = self.noise_scalar_calc_functional(p1[0], p1[1], t)
        f2 = self.noise_scalar_calc_functional(p2[0], p2[1], t + dt)
        f3 = self.noise_scalar_calc_functional(p3[0], p3[1], t + 2 * dt)
        # lets make them numpy vectors
        x1 = np.array(p1)
        x2 = np.array(p2)
        x3 = np.array(p3)
        xs = [x1, x2, x3]
        n_min = np.argmin([f1, f2, f3])
        # print("n min is {}".format(n_min))
        n_max = np.argmax([f1, f2, f3])
        # print("n max is {}".format(n_max))
        num = [0, 1, 2]
        num.remove(n_min)
        # print(num)
        num.remove(n_max)
        n_av = num[0]
        # print(num)
        # print("n av is {}".format(n_av))
        x_min = xs[n_min]
        x_max = xs[n_max]
        # find middlepoint of triangle
        x_mid = 0.5 * (x1 + x2 + x3 - x_max)
        # reflection
        x5 = x_mid + alpha * (x_mid - x_max)
        # stretching
        f5 = self.noise_scalar_calc_functional(x5[0], x5[1], t + 3 * dt)
        if (f5 <= min([f1, f2, f3])):
            x6 = x_mid + gamma * (x5 - x_mid)
            f6 = self.noise_scalar_calc_functional(x6[0], x6[1], t + 4 * dt)
            if (f6 < min([f1, f2, f3])):
                return x6, xs[n_min], xs[n_av], x_mid
            else:
                return x5, xs[n_min], xs[n_av], x_mid
        # compression
        if (f5 > [f1, f2, f3][n_min] and f5 > [f1, f2, f3][n_av] and f5 < [f1, f2, f3][n_max]):
            x6 = x_mid + beta * (x_max - x_mid)
            return x6, xs[n_min], xs[n_av], x_mid
        # reduction
        if (f5 >= [f1, f2, f3][n_max]):
            x11 = x_min + 0.5 * (x1 - x_min)
            x21 = x_min + 0.5 * (x2 - x_min)
            x31 = x_min + 0.5 * (x3 - x_min)
            return x11, x21, x31, x_mid

        return x5, xs[n_min], xs[n_av], x_mid


    def find_constant_for_all(self, max_iteration_number=40):
        # we need to find F(every E and T) - F(E_min, T_min) = F(every E and T)
        # at all steps and summ it for each point (E, T)
        E = np.arange(-100, 200, 0.25)
        T = np.arange(-30, 40, 0.05)
        X, Y = np.meshgrid(E, T)
        d1x, d2x = np.shape(X)
        d1y, d2y = np.shape(Y)
        X_ = X.reshape((d1x * d2x, 1))
        Y_ = Y.reshape((d1y * d2y, 1))
        t = 1  # start time
        # Z_ = self.vector_calc_fuctional(X_, Y_, t)
        z_min = np.array((0,))
        y_min = np.array((0,))
        x_min = np.array((0,))
        # np array for quadratic errors of each iteration
        error = np.zeros((np.len(E), np.len(T)))
        for i in range(0, max_iteration_number):
            pass


def main():
    # dict of parameters for model from experiment
    # dt is 20 minutes and it is our new time constant
    parameters = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1}
    plant = plant_model(**parameters)
    print("Start it")
    # error = plant.find_gradient_minimum(x_start = 5, y_start = 5, show = False)
    # error = plant.find_conj_gradient_minimum(x_start=5, y_start=5, show=True)
    # plant.find_newton_minimum(x_start = 3, y_start = 3)
    # plant.find_triangle_minimum(x_start = 5, y_start = 5)
    error = plant.find_triangle_minimum(x_start = 20, y_start = 20, show = False,
                                        alpha = 1, beta = 0.5, gamma = 2.9, max_iteration_number = 600)

    print(error)
    # show squared errors by steps
    fig = pl.figure()
    pl.plot(range(0, np.shape(error)[0], 1), error[:], "-or")
    pl.grid()
    pl.show()

if __name__ == "__main__":
    main()





