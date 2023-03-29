from matplotlib import pyplot as plt

fig_num = 0
def new_plot():
    global fig_num
    fig_num += 1
    plt.figure(fig_num)