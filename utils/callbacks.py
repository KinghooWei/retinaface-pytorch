import os

import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt


class LossHistory():
    def __init__(self, save_path, form):
        # import datetime
        # curr_time = datetime.datetime.now()
        # time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        # self.log_dir    = log_dir
        self.form = form
        # self.time_str   = time_str
        self.save_path  = save_path
        self.losses     = []

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def append_loss(self, loss):
        self.losses.append(loss)
        with open(os.path.join(self.save_path, self.form + "-loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='%s loss'%self.form)
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth %s loss'%self.form)
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, self.form + "-loss.png"))

        plt.cla()
        plt.close("all")
