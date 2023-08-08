import time
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import src.conf as C
from src.p2p import Graph
from src.utils import load_conf, fixed_seed, exp_details, save, load

""" 
TODO :
 --> Scale data between 1 and -1: https://apmonitor.com/do/index.php/Main/LSTMNetwork
 --> Review LSTM params: https://github.com/srivatsan88/End-to-End-Time-Series/blob/master/Multivariate_Time_Series_Modeling_using_LSTM.ipynb
 --> RMSE ::     print "%.2f RMSE" % (math.sqrt(mean_squared_error(true_vals, predictions)))
"""

if __name__ == '__main__':
    # colors = ['black', 'blue', 'orange', 'red']
    # data = {
    #     "1H": load("CL_1H_500_974.pkl")[0],
    #     "30M": load("CL_30min_500_974.pkl")[0],
    #     "15M": load("CL_15min_500_974.pkl")[0],
    #     "5M": load("CL_5min_500_974.pkl")[0],
    # }
    # i = 0
    # for k, v in data.items():
    #     dd = v['train']['val_rmse'][:100]
    #     dd.append(v['test']['rmse'])
    #     plt.plot(range(len(dd)), dd, label=k, color=colors[i])
    #     # plt.bar(i, dd, label=k, color=colors[i])
    #     i = i + 1
    # plt.legend(loc="best", shadow=True)
    # plt.show()
    t = time.time()
    args = load_conf(use_cpu=True)
    # Configuration ------------------>
    cluster_id = 0  # 0    5484672
    season = 'summer'  # 'summer'
    args.model = "RNN"
    args.epochs = 5  # 5
    args.batch_size = 128
    C.TIME_ABSTRACTION = "15min"
    C.RECORD_PER_HOUR = 4
    resample = False if C.TIME_ABSTRACTION is None else True

    # Details ------------------------>
    fixed_seed(True)
    exp_details(args)

    # Centralized Training ----------->
    results = Graph.centralized_training(args, cluster_id, season, resample, predict=False)
    save(f"CL_{C.ML_ENGINE}_{C.TIME_ABSTRACTION}_cluster_{cluster_id}_{season}_{args.epochs}", results)
    # save(f"CL_{C.TIME_ABSTRACTION}_{args.epochs}", results)
    # train_log, predictions, n_steps_predictions, homes_logs, meta_logs = results
    ""
    # Plots -------------------------->
    # plot predictions
    # info = Map({'xlabel': "Time period", 'ylabel': 'Temperature'})
    # plots.plot_predictions(predictions, info)
    # plots.plot_predictions(n_steps_predictions, info)
    # # plot box_plot
    # box_plot(train_log, homes_logs, showfliers=False)
    # # plot scatter_plot
    # scatter_plot(homes_logs, meta_logs)

    # END ---------------------------->
    print(f"END in {round(time.time() - t, 2)}s")
