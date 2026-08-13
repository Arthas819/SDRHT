"""

 This file plots all result curves used by Experiment_*.py.

 """

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

'''
    Error rate curves visualization. 
'''
def plot_error_rate_curves(
    df_mean,
    dataset,
    d,
    figure_path,
    x_label='# observation (m)',
    x_ticks=None,
    x_tick_labels=None,
):
    _plot_metric_curves(
        df_mean,
        dataset,
        d,
        figure_path,
        y_label='Error Rate',
        x_label=x_label,
        x_ticks=x_ticks,
        x_tick_labels=x_tick_labels,
        color_scale=1.0,
    )


def plot_f1_score_curves(df_mean, dataset, d, figure_path, x_label='# observation (m)', x_ticks=None):
    _plot_metric_curves(
        df_mean,
        dataset,
        d,
        figure_path,
        y_label='F1-score',
        x_label=x_label,
        x_ticks=x_ticks,
        color_scale=0.65,
    )


def _plot_metric_curves(
    df_mean,
    dataset,
    d,
    figure_path,
    y_label,
    x_label,
    x_ticks=None,
    x_tick_labels=None,
    color_scale=1.0,
):
    df_plot = df_mean.sort_index()

    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    for method_idx, method in enumerate(df_plot.columns):
        style = _method_style(method_idx, color_scale=color_scale)
        ax.plot(
            df_plot.index,
            df_plot[method],
            label=_method_label(method),
            **style,
        )
    ax.set_xlabel(x_label, fontfamily='Times New Roman', fontsize=25)
    ax.set_ylabel(y_label, fontfamily='Times New Roman', fontsize=25)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        labels = x_tick_labels if x_tick_labels is not None else [str(x_tick) for x_tick in x_ticks]
        ax.set_xticklabels(labels)
    ax.tick_params(axis='both', labelsize=22, width=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
    handles, labels = ax.get_legend_handles_labels()
    n_legend_cols = 2 if len(labels) >= 6 else 1
    legend = ax.legend(
        handles,
        labels,
        loc='best',
        ncol=n_legend_cols,
        prop={'family': 'Times New Roman', 'size': 21},
        frameon=True,
        facecolor='white',
        framealpha=0.8,
        edgecolor='#d0d0d0',
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.8,
    )
    legend.set_zorder(20)
    ax.grid(True, linestyle='-', linewidth=0.6, alpha=0.18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def _method_style(method_idx, color_scale=1.0):
    colors = [
        '#d62728',
        '#0072b2',
        '#56b4e9',
        '#009e73',
        '#e69f00',
        '#7a5195',
        '#cc79a7',
        '#000000',
        '#f781bf',
        '#a65628',
        '#4daf4a',
        '#984ea3',
    ]
    linestyles = ['-', '--', '-.', ':']
    markers = ['h', 'p', 's', 'D', 'o', 'v', '^', 'X', '*', '<', '>', '8']
    base_style = {
        'linewidth': 2.6,
        'markersize': 9,
        'markeredgewidth': 1.0,
    }
    linestyle = linestyles[method_idx % len(linestyles)]
    if method_idx == 4:
        linestyle = '--'

    return {
        **base_style,
        'color': _scale_hex_color(colors[method_idx % len(colors)], color_scale),
        'linestyle': linestyle,
        'marker': markers[method_idx % len(markers)],
        'zorder': 10 if method_idx == 0 else 2,
    }


def _scale_hex_color(color, scale):
    if scale >= 0.999:
        return color
    color = color.lstrip('#')
    rgb = [int(color[idx:idx + 2], 16) for idx in (0, 2, 4)]
    rgb = [max(0, min(255, int(channel * scale))) for channel in rgb]
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def _method_label(method):
    tokens = str(method).strip().replace('-', '_').split('_')
    return '-'.join(_format_method_token(token) for token in tokens if token)


def _format_method_token(token):
    canonical_tokens = {
        'sdro': 'SDRO',
        'hycnn': 'HyCNN',
        'icnn': 'ICNN',
        'fdro': 'FDRO',
        'wdro': 'WDRO',
        'gmm': 'GMM',
        'lr': 'LR',
        'svm': 'SVM',
        '3nn': '3NN',
    }
    return canonical_tokens.get(token.lower(), token)
