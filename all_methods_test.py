import numpy as np
from growing_4 import plant_model
import pylab as pl
import pandas as pd


# it is global constant search time, depending on search task
# all steps before this time is in search error area, after -in yaw error area
SEARCH_TIME = 100 # 72*dt it is about 24 hours
ITERATIONS = 100

if __name__ == "__main__":
    # global SEARCH_TIME
    p1 = {'a': 0.0014, 'b': 0.087, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1, 'noise': 0.01}
    p2 = {'a': 0.00055, 'b': 0.087, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1, 'noise': 0.01}
    p3 = {'a': 0.0014, 'b': 0.087, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1, 'noise': 0.01}
    p4 = {'a': 0.00055, 'b': 0.087, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1, 'noise': 0.01}
    p5 = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1, 'noise': 0.01}
    p6 = {'a': 0.00055, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1, 'noise': 0.01}
    p7 = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1, 'noise': 0.01}
    p8 = {'a': 0.00055, 'b': 0.175, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1, 'noise': 0.01}

    p9 = {'a': 0.0014, 'b': 0.087, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1, 'noise': 0.1}
    p10 = {'a': 0.00055, 'b': 0.087, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1, 'noise': 0.1}
    p11 = {'a': 0.0014, 'b': 0.087, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1, 'noise': 0.1}
    p12 = {'a': 0.00055, 'b': 0.087, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1, 'noise': 0.1}
    p13 = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1, 'noise': 0.1}
    p14 = {'a': 0.00055, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1, 'noise': 0.1}
    p15 = {'a': 0.0014, 'b': 0.175, 'e1': 1, 'e2': 10, 'k': 0.15, 'dt': 1, 'noise': 0.1}
    p16 = {'a': 0.00055, 'b': 0.175, 'e1': 1, 'e2': 1, 'k': 0.15, 'dt': 1, 'noise': 0.1}


    models = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16]
    # models = [p11]
    models = [p3]
    for model in models:
        plant = plant_model(**model)

        sum_error_Nelder = np.zeros(600)
        find_error_Nelder = sum_error_Nelder[0:int(SEARCH_TIME/5)]
        yaw_error_Nelder = sum_error_Nelder[int(SEARCH_TIME/5):]

        sum_error_grad = np.zeros(600)
        find_error_grad = sum_error_grad[0:int(SEARCH_TIME/3)]
        yaw_error_grad = sum_error_grad[int(SEARCH_TIME/3):]

        sum_error_congrad = np.zeros(600)
        find_error_congrad = sum_error_congrad[0:int(SEARCH_TIME/5)]
        yaw_error_congrad = sum_error_congrad[int(SEARCH_TIME/5):]

        sum_error_combined = np.zeros(600)
        find_error_combined = sum_error_combined[0:int(SEARCH_TIME/5)]
        yaw_error_combined = sum_error_combined[int(SEARCH_TIME/5):]

        kwargs_Nelder = {"x_start": 20, "y_start": 20, "show": False,
                         "alpha": 1, "beta": 1, "gamma": 3, "max_iteration_number": 600}
        kwargs_grad = {"x_start": 20, "y_start": 20, "show": False,
                       "gamma": 0.001, "max_iteration_number": 600}
        kwargs_congrad = {"x_start": 20, "y_start": 20, "show": False, "max_iteration_number": 600}
        # N_transition is a number of steps, after them we go to gradient method from Nelder
        # it depends on SEARCH_TIME, and its about SEARCH_TIME/time_of_one_Nelder_iteration
        kwargs_combined = {"g_grad": 0.001, "a": 1, "b": 1, "g": 3, "N_transition": 20,
                           "max_iteration_number": 600, "x_start": 20, "y_start": 20, "show": False}

        for i in range(0, ITERATIONS):

            print("Iteration {}".format(i))
            # Nelder-Mid
            try:
                ern = plant.find_triangle_minimum(**kwargs_Nelder)
            except Exception:
                sum_error_Nelder += 100000*np.ones((600))
            else:
                sum_error_Nelder += ern/ITERATIONS

            find_error_Nelder += sum_error_Nelder[0:int(SEARCH_TIME/5)]/ITERATIONS
            yaw_error_Nelder += sum_error_Nelder[int(SEARCH_TIME/5):]/ITERATIONS
            # Gradient
            try:
                erg = plant.find_gradient_minimum(**kwargs_grad)
            except Exception:
                sum_error_grad += 1*np.ones((600))
            else:
                sum_error_grad += erg/ITERATIONS
            find_error_grad += sum_error_grad[0:int(SEARCH_TIME/3)]/ITERATIONS
            yaw_error_grad += sum_error_grad[int(SEARCH_TIME/3):]/ITERATIONS
            # # Conjugate gradient
            # try:
            #     ercg = plant.find_conj_gradient_minimum(**kwargs_congrad)
            # except Exception:
            #     sum_error_congrad += 100000*np.ones((600))
            # else:
            #     sum_error_congrad += ercg
            # find_error_congrad += sum_error_congrad[0:int(SEARCH_TIME/5)]
            # yaw_error_congrad += sum_error_congrad[int(SEARCH_TIME/5):]
            # Combined method
            try:
                erc = plant.combined_method(**kwargs_combined)
            except Exception:
                sum_error_combined += 1 * np.ones((600))
            else:
                sum_error_combined += erc/ITERATIONS
            find_error_combined += sum_error_combined[0:int(SEARCH_TIME/5)]/ITERATIONS
            yaw_error_combined += sum_error_combined[int(SEARCH_TIME/5):]/ITERATIONS

        print(model)

        print("Nelder error: summ {}, search {}, yaw {}"
              .format(np.sum(sum_error_Nelder), np.sum(find_error_Nelder), np.sum(yaw_error_Nelder)))
        print("Grad : summ error is : {}, search : {}, yaw : {}"
              .format(np.sum(sum_error_grad), np.sum(find_error_grad), np.sum(yaw_error_grad)))
        print("Combined : summ error is : {}, search : {}, yaw : {}"
              .format(np.sum(sum_error_combined), np.sum(find_error_combined), np.sum(yaw_error_combined)))
        # print("Conjugate grad : summ error is : {}, search : {}, yaw : {}"
        #       .format(np.sum(sum_error_congrad)/100, np.sum(find_error_congrad)/100, np.sum(yaw_error_congrad)/100))
        # show squared errors by steps
        fig = pl.figure()
        # print((np.arange(100,100 + np.shape(sum_error_Nelder[100::10])[0], 1)*10))
        # print(np.shape(sum_error_Nelder[100::10])[0])
        # print(np.shape(sum_error_Nelder[100::10]))
        pl.plot((np.arange(0, np.shape(sum_error_Nelder[:20:2])[0], 1))*2, sum_error_Nelder[:20:2], "or",
                label = "Nelder-Mead", linewidth=3, markersize = 8)
        pl.plot(((np.arange(0, np.shape(sum_error_Nelder[20::10])[0], 1))*10)+20, sum_error_Nelder[20::10], "or"
                , linewidth=3, markersize = 8)

        final_errors_Nelder = np.concatenate((sum_error_Nelder[:20:2], sum_error_Nelder[20::10]))
        final_intervals_Nelder = np.concatenate((((np.arange(0, np.shape(sum_error_Nelder[:20:2])[0], 1))*2,
                    ((np.arange(0, np.shape(sum_error_Nelder[20::10])[0], 1))*10)+20)))

        pl.plot(np.arange(0, np.shape(sum_error_grad[::10])[0], 1)*10, sum_error_grad[::10], "vb",
                label = "Gradient", linewidth=3, markersize = 8)

        final_errors_Gradient = sum_error_grad[::10]
        final_intervals_Gradient = np.arange(0, np.shape(sum_error_grad[::10])[0], 1)*10

        pl.plot(np.arange(0, np.shape(sum_error_combined[:20:3])[0], 1)*3, sum_error_combined[:20:3],
                "sc", label = "Combined method", linewidth=3, markersize = 8)
        pl.plot(np.arange(0, np.shape(sum_error_combined[20::15])[0], 1)*15+20, sum_error_combined[20::15],
                "sc", linewidth=3, markersize = 8)

        final_errors_Combined = np.concatenate(
            (sum_error_combined[:20:3], sum_error_combined[20::15])
        )
        final_intervals_Combined = np.concatenate(
            (np.arange(0, np.shape(sum_error_combined[:20:3])[0], 1) * 3,
             np.arange(0, np.shape(sum_error_combined[20::15])[0], 1) * 15 + 20)
        )

        pl.plot(range(0, np.shape(sum_error_Nelder)[0], 1), sum_error_Nelder, "-r",
                linewidth=1, markersize = 6)
        pl.plot(range(0, np.shape(sum_error_grad)[0], 1), sum_error_grad, "-b"
                , linewidth=1, markersize = 6)
        pl.plot(range(0, np.shape(sum_error_combined)[0], 1), sum_error_combined,
                "-c", linewidth=1, markersize = 6)
        # pl.plot(range(0, np.shape(sum_error_congrad)[0], 1), sum_error_congrad[:], "-^k",
        #         label="conjugate gradient summ error")
        pl.xlabel("Iterations")
        pl.ylabel("Integral error")
        pl.title("Integral error for 600 iterations with 10% noise level")
        pl.legend()
        pl.grid()

        # convert data to xlsx

        # Create a Pandas dataframe from some data.
        # df = pd.DataFrame({'Y_values': [10, 20, 30, 20, 15, 30, 45], 'X_values': [1, 2, 3, 4, 5, 6, 7]})
        # }
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('Integral_error_for_600_iterations_with_1_noise_level.xlsx', engine='xlsxwriter')

        df_Nelder_Mead = pd.DataFrame(
            {'Iterations':final_intervals_Nelder,'Integral error':final_errors_Nelder}
        )
        df_Gradient = pd.DataFrame(
            {'Iterations':final_intervals_Gradient,'Integral error':final_errors_Gradient}
        )
        df_Combined = pd.DataFrame(
            {'Iterations': final_intervals_Combined, 'Integral error': final_errors_Combined}
        )
        df_Nelder_Mead_full = pd.DataFrame(
            {'Iterations':np.arange(0, np.shape(sum_error_Nelder)[0], 1),
             'Integral error':sum_error_Nelder}
        )
        df_Gradient_full = pd.DataFrame(
            {'Iterations': np.arange(0, np.shape(sum_error_grad)[0], 1),
             'Integral error': sum_error_grad}
        )
        #
        df_Combined_full = pd.DataFrame(
            {'Iterations': np.arange(0, np.shape(sum_error_combined)[0], 1),
             'Integral error': sum_error_combined}
        )
        # Convert the dataframe to an XlsxWriter Excel object.
        df_Nelder_Mead.to_excel(writer, sheet_name='Nelder_Mead')
        df_Gradient.to_excel(writer, sheet_name='Gradient')
        df_Combined.to_excel(writer, sheet_name='Combined')

        df_Nelder_Mead_full.to_excel(writer, sheet_name='Nelder_Mead_full')
        df_Gradient_full.to_excel(writer, sheet_name='Gradient_full')
        df_Combined_full.to_excel(writer, sheet_name='Combined_full')
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

        pl.savefig("test_rasterization.eps", dpi=500)
        pl.savefig("test_rasterization.pdf", dpi=500)
        pl.show()
        print("End of calculation")