import seaborn as sns

strategies_palette = {
    "myopic": "skyblue",
    "take_loss": "orangered",
    "random": "springgreen",
}


def plot_final_rewards(data):
    g = sns.displot(
        data=data,
        x="total_reward",
        hue="strategy",
        kind="hist",
        palette=strategies_palette,
        stat="density",
        common_norm=False,
    )
    g.set(xlabel="Final total reward", title=f"Strategy final total reward comparison")
    g._legend.remove()
    return g


def plot_avg_reward_per_step(data):
    g = sns.relplot(
        data=data,
        x="step",
        y="reward",
        col="strategy",
        hue="strategy",
        height=4,
        aspect=0.9,
        kind="line",
        palette=strategies_palette,
    )
    for ax in g.axes.flat:
        ax.set_xticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8])  # set new labels
        ax.set_xticklabels(fontsize=10, labels=[str(i) for i in range(1, 9)])
    g._legend.remove()
    return g
