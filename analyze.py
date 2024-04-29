import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_line(data, x_label, y_label, title, file_name):
    for x, mean, std, label in data:
        n = len(x)
        mean = mean[:n]
        std = std[:n]
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.1)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(file_name, bbox_inches="tight")
    plt.clf()

def steps_vs_psi(data):
    for h in [0, 1, 2, 4, 8]:
        for tolerance in [0, 1, 10, 100, 1000]:
            _data = [(range(1_000), *data[(gamma, h, tolerance)], f"$\\gamma={gamma}$") for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]]
            plot_line(_data, "Steps", "$\\psi$", f"$h = {h}$, $\\tau = {tolerance}$", f"figs/steps_vs_psi/vary_gamma/steps_vs_psi_h_{h}_tau_{tolerance}.pdf")
            _data = [(range(100), *data[(gamma, h, tolerance)], f"$\\gamma={gamma}$") for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]]
            plot_line(_data, "Steps", "$\\psi$", f"$h = {h}$, $\\tau = {tolerance}$", f"figs/steps_vs_psi/vary_gamma/steps_vs_psi_h_{h}_tau_{tolerance}_zoom_in.pdf")
        for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _data = [(range(1_000), *data[(gamma, h, tolerance)], f"$\\tau={tolerance}$") for tolerance in [0, 1, 10, 100, 1000]]
            plot_line(_data, "Steps", "$\\psi$", f"$h = {h}$, $\\gamma = {gamma}$", f"figs/steps_vs_psi/vary_tau/steps_vs_psi_h_{h}_gamma_{gamma}.pdf")
            _data = [(range(100), *data[(gamma, h, tolerance)], f"$\\tau={tolerance}$") for tolerance in [0, 1, 10, 100, 1000]]
            plot_line(_data, "Steps", "$\\psi$", f"$h = {h}$, $\\gamma = {gamma}$", f"figs/steps_vs_psi/vary_tau/steps_vs_psi_h_{h}_gamma_{gamma}_zoom_in.pdf.pdf")
    for tolerance in [0, 1, 10, 100, 1000]:
        for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _data = [(range(1_000), *data[(gamma, h, tolerance)], f"$h={h}$") for h in [0, 1, 2, 4, 8]]
            plot_line(_data, "Steps", "$\\psi$", f"$\\tau = {tolerance}$, $\\gamma = {gamma}$", f"figs/steps_vs_psi/vary_h/steps_vs_psi_tau_{tolerance}_gamma_{gamma}.pdf")
            _data = [(range(100), *data[(gamma, h, tolerance)], f"$h={h}$") for h in [0, 1, 2, 4, 8]]
            plot_line(_data, "Steps", "$\\psi$", f"$\\tau = {tolerance}$, $\\gamma = {gamma}$", f"figs/steps_vs_psi/vary_h/steps_vs_psi_tau_{tolerance}_gamma_{gamma}_zoom_in.pdf")

def gamma_vs_psi(data):
    for h in [0, 1, 2, 4, 8]:
        _data = []
        for tolerance in [0, 1, 10, 100, 1000]:
            mean = []
            std = []
            for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
                means, stds = data[(gamma, h, tolerance)]
                mean.append(means[-1])
                std.append(stds[-1])
            mean = np.array(mean)
            std = np.array(std)
            label = f"$\\tau={tolerance}$"
            _data.append(([0.0, 0.25, 0.5, 0.75, 1.0], mean, std, label))
            plot_line(_data, "$\\gamma$", "$\\psi$", f"$h = {h}$", f"figs/gamma_vs_psi/vary_tau/gamma_vs_psi_h_{h}.pdf")
    for tolerance in [0, 1, 10, 100, 1000]:
        _data = []
        for h in [0, 1, 2, 4, 8]:
            mean = []
            std = []
            for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
                means, stds = data[(gamma, h, tolerance)]
                mean.append(means[-1])
                std.append(stds[-1])
            mean = np.array(mean)
            std = np.array(std)
            label = f"$h={h}$"
            _data.append(([0.0, 0.25, 0.5, 0.75, 1.0], mean, std, label))
            plot_line(_data, "$\\gamma$", "$\\psi$", f"$\\tau = {tolerance}$", f"figs/gamma_vs_psi/vary_h/gamma_vs_psi_tau_{tolerance}.pdf")

data = {}
for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
    for h in [0, 1, 2, 4, 8]:
        for tolerance in [0, 1, 10, 100, 1000]:
            file_name_prefix = f"exps/csvs/exp_gamma_{gamma}_h_{h}_tolerance_{tolerance}_seed_"
            dfs = []
            for seed in range(30):
                df = pd.read_csv(file_name_prefix + f"{seed}.csv")["psi"].reindex(range(1_000), fill_value=1.0)
                dfs.append(df)
            mean_df = np.mean(dfs, axis=0)
            std_df = np.std(dfs, axis=0)
            data[(gamma, h, tolerance)] = (mean_df, std_df)

steps_vs_psi(data)

gamma_vs_psi(data)


















