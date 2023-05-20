import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


results_l0 = pd.read_csv("results_figure1.csv")
results_lasso = pd.read_csv('./lasso/lasso_results_figure1.csv')


fig, axs = plt.subplots(3)
# fig.suptitle('Comparison')

axs[0].set_ylim([0, 1.01])
axs[0].set_ylabel('A')
axs[0].plot(results_l0.n, results_l0.A, color='blue', marker='o', markerfacecolor='none', markersize=2, linewidth=0.7)
axs[0].plot(results_lasso.n, results_lasso.A, color='red', marker='s', markerfacecolor='none', markersize=3, linewidth=0.7)
axs[0].legend(['Exact', 'Lasso'], loc='lower right')

axs[1].set_ylim([0, 0.01])
axs[1].set_ylabel('F')
axs[1].plot(results_l0.n, results_l0.F, color='blue', marker='o', markerfacecolor='none', markersize=2, linewidth=0.7)
axs[1].plot(results_lasso.n, results_lasso.F, color='red', marker='s', markerfacecolor='none', markersize=3, linewidth=0.7)
axs[1].legend(['Exact', 'Lasso'])

axs[2].set_ylabel('Time')
axs[2].plot(results_l0.n, results_l0.Time, color='blue', marker='o', markerfacecolor='none', markersize=2, linewidth=0.7)
axs[2].plot(results_lasso.n, results_lasso.time, color='red', marker='s', markerfacecolor='none', markersize=3, linewidth=0.7)
axs[2].legend(['Exact', 'Lasso'])

fig.show()
fig.savefig("comparison1.pdf", format="pdf", bbox_inches="tight")


## Results of general w
results_l0 = pd.read_csv("results_general_w.csv")
results_lasso = pd.read_csv('./lasso/lasso_results_general_w.csv')

fig, axs = plt.subplots(3)
# fig.suptitle('Comparison')

axs[0].set_ylim([0, 1.01])
axs[0].set_ylabel('A')
axs[0].plot(results_l0.n, results_l0.A, color='blue', marker='o', markerfacecolor='none', markersize=2, linewidth=0.7)
axs[0].plot(results_lasso.n, results_lasso.A, color='red', marker='s', markerfacecolor='none', markersize=3, linewidth=0.7)
axs[0].legend(['Exact', 'Lasso'], loc='lower right')

axs[1].set_ylim([0, 0.01])
axs[1].set_ylabel('F')
axs[1].plot(results_l0.n, results_l0.F, color='blue', marker='o', markerfacecolor='none', markersize=2, linewidth=0.7)
axs[1].plot(results_lasso.n, results_lasso.F, color='red', marker='s', markerfacecolor='none', markersize=3, linewidth=0.7)
axs[1].legend(['Exact', 'Lasso'])

axs[2].set_ylabel('Time')
axs[2].plot(results_l0.n, results_l0.Time, color='blue', marker='o', markerfacecolor='none', markersize=2, linewidth=0.7)
axs[2].plot(results_lasso.n, results_lasso.time, color='red', marker='s', markerfacecolor='none', markersize=3, linewidth=0.7)
axs[2].legend(['Exact', 'Lasso'])

fig.show()
fig.savefig("comparison2.pdf", format="pdf", bbox_inches="tight")
