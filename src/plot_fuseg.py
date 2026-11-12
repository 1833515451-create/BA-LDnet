from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    files = [args.source / 'FUSeg' / f'{name}_per_image.csv'
             for name in ('baseline', 'boundary_aux')]
    a, b = [pd.read_csv(p).set_index('image') for p in files]
    assert a.index.is_unique and b.index.is_unique
    assert set(a.index) == set(b.index) and len(a) == 986
    b = b.loc[a.index]
    assert a.tiny_dice.notna().equals(b.tiny_dice.notna())
    cols = ['dice', 'precision', 'recall']
    assert np.isfinite(a[cols].to_numpy()).all()
    assert np.isfinite(b[cols].to_numpy()).all()
    assert ((a[cols] >= 0) & (a[cols] <= 1)).all().all()
    assert ((b[cols] >= 0) & (b[cols] <= 1)).all().all()
    tiny = a.tiny_dice.notna()
    assert tiny.sum() == 156
    assert round(a.dice.mean()*100, 2) == 77.25
    assert round(b.dice.mean()*100, 2) == 78.13
    assert (a.dice < .5).sum() == 79 and (b.dice < .5).sum() == 73
    source = pd.DataFrame(index=a.index)
    source['tiny'] = tiny
    for c in cols:
        source[f'mean_teacher_{c}_percent'] = a[c]*100
        source[f'ba_ldnet_{c}_percent'] = b[c]*100
        source[f'delta_{c}_pp'] = (b[c]-a[c])*100
    source.to_csv(out/'paired_source_data.csv', float_format='%.12g')
    delta = (b[cols]-a[cols])*100
    # Numerical equality tolerance is far below the stored metric precision.
    eps = 1e-6
    p, r, d = delta.precision, delta.recall, delta.dice
    quads = {
        'both_increased': int(((p > eps) & (r > eps)).sum()),
        'precision_up_recall_down': int(((p > eps) & (r < -eps)).sum()),
        'precision_down_recall_up': int(((p < -eps) & (r > eps)).sum()),
        'both_decreased': int(((p < -eps) & (r < -eps)).sum()),
        'at_least_one_unchanged': int(((p.abs() <= eps) | (r.abs() <= eps)).sum()),
    }
    assert sum(quads.values()) == len(a)
    stats = {
        'n': len(a), 'n_tiny': int(tiny.sum()), 'seed': 2026,
        'label_fraction': .2, 'equality_tolerance_pp': eps,
        'dice_increased': int((d > eps).sum()),
        'dice_decreased': int((d < -eps).sum()),
        'dice_unchanged': int((d.abs() <= eps).sum()),
        'precision_recall_quadrants': quads,
        'mean_teacher_means_percent': (a[cols].mean()*100).to_dict(),
        'ba_ldnet_means_percent': (b[cols].mean()*100).to_dict(),
        'mean_differences_pp': delta.mean().to_dict(),
        'tiny_mean_dice_difference_pp': float(d[tiny].mean()),
        'failure_recovered': int(((a.dice < .5) & (b.dice >= .5)).sum()),
        'failure_new': int(((a.dice >= .5) & (b.dice < .5)).sum()),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (out/'statistics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')

    # Quantitative grid: overall paired overlap and within-image P/R changes.
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 8,
        'axes.labelsize': 9, 'axes.titlesize': 10,
        'axes.linewidth': .7, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'legend.frameon': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.95))
    fig.subplots_adjust(left=.085, right=.98, bottom=.19, top=.79, wspace=.35)
    colors = ['#6B7780', '#B85C26']
    for ax in axes:
        ax.set_box_aspect(1)
        ax.tick_params(length=3, width=.6)
    x, y = a.dice*100, b.dice*100
    axes[0].plot([0,100], [0,100], linestyle='--', color='#333333', lw=.8, zorder=1)
    for mask, color, marker, size, alpha in [
        (~tiny, colors[0], 'o', 11, .43),
        (tiny, colors[1], '^', 22, .85),
    ]:
        axes[0].scatter(x[mask], y[mask], s=size, c=color, marker=marker,
                        alpha=alpha, linewidths=.25, edgecolors='white', zorder=2)
        axes[1].scatter(r[mask], p[mask], s=size, c=color, marker=marker,
                        alpha=alpha, linewidths=.25, edgecolors='white', zorder=2)
    axes[0].set(xlim=(-2,102), ylim=(-2,102),
                xticks=range(0,101,20), yticks=range(0,101,20),
                xlabel='Mean Teacher Dice (%)', ylabel='BA-LDNet Dice (%)')
    axes[0].set_title('a  Paired image-level Dice', loc='left', pad=9)
    axes[0].text(.04,.96, 'Higher: 553\nLower: 425\nUnchanged: 8',
                 transform=axes[0].transAxes, va='top', fontsize=8,
                 bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
    axes[1].axhline(0, lw=.7, color='#444444', zorder=1)
    axes[1].axvline(0, lw=.7, color='#444444', zorder=1)
    axes[1].set(xlim=(-33,66), ylim=(-33,66),
                xticks=[-20,0,20,40,60], yticks=[-20,0,20,40,60],
                xlabel=r'$\Delta$Recall (percentage points)',
                ylabel=r'$\Delta$Precision (percentage points)')
    axes[1].set_title('b  Paired precision-recall changes', loc='left', pad=9)
    for xy, text, ha, va in [
        ((-.0, .98), 'Precision up\nRecall down\nn = 282', 'left', 'top'),
        ((.98, .98), 'Both up\nn = 137', 'right', 'top'),
        ((.02, .02), 'Both down\nn = 71', 'left', 'bottom'),
        ((.98, .02), 'Precision down\nRecall up\nn = 169', 'right', 'bottom'),
    ]:
        axes[1].text(*xy, text, transform=axes[1].transAxes, ha=ha, va=va,
                     fontsize=7.5, color='#333333',
                     bbox={'facecolor':'white','edgecolor':'none','alpha':.87,'pad':1})
    handles = [Line2D([],[],marker=m,linestyle='',color=c,markersize=5,label=l)
               for m,c,l in [('o',colors[0],'Non-Tiny (n = 830)'),
                              ('^',colors[1],'Tiny (n = 156)')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52,.96),
               ncol=2, handletextpad=.4, columnspacing=2, fontsize=9)
    fig.text(.52,.035, 'FUSeg (n = 986)  |  20% labeling  |  Pre-specified primary run (seed 2026)',
             ha='center', fontsize=8)
    # Explicit limits include every observation; no outlier clipping or jitter.
    assert r.min() > -33 and r.max() < 66 and p.min() > -33 and p.max() < 66
    assert sum(len(c.get_offsets()) for c in axes[0].collections) == 986
    assert sum(len(c.get_offsets()) for c in axes[1].collections) == 986
    fig.canvas.draw()
    for ax in axes:
        for text in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            extent = text.get_window_extent(fig.canvas.get_renderer())
            assert extent.x0 >= 0 and extent.y0 >= 0
            assert extent.x1 <= fig.bbox.width and extent.y1 <= fig.bbox.height
    for suffix in ['pdf','svg']:
        kwargs = {'dpi':600} if suffix in ['png','tiff'] else {}
        fig.savefig(out/f'FUSeg_paired_analysis.{suffix}', **kwargs)
    plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
