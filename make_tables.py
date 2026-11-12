"""Turn the output of ai4ts_fixes.run_all() into the paper's tables, figures and
in-text numbers.

    python3 make_tables.py                 # reads results/ai4ts_*.csv
    python3 make_tables.py path/to/ai4ts   # or another prefix

Reads (only the first is required; missing ones leave TODO placeholders):
    <prefix>_results.csv    run_all: scores per horizon and model
    <prefix>_dm.csv         run_all: Diebold-Mariano tests
    <prefix>_dropped.csv    run_all: daytime hours missing / outage per year
    <prefix>_alignment.csv  run_all: best UTC offset per season, NASA shift
    <prefix>_monthly.csv    run_all: monthly RMSE, 2021 vs 2022
    <prefix>_lstm.csv       score_lstm() rows for every horizon, concatenated

Writes tables/*.tex, tables/numbers.tex and figures/*.pdf. paper.tex inputs each
file if it exists and shows a red TODO if it doesn't, so don't hand-edit the
generated files; rerun this script instead.

Figures are one AAAI column wide (3.3125 in) with Times at 8 pt and embedded
TrueType fonts. Color follows the model and line style follows the setup, so
they still read in grayscale.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLES, FIGURES = HERE / 'tables', HERE / 'figures'
ORACLE = ' + target-hour irradiance'
BASELINES = ['persistence', 'day-ago persistence', 'smart persistence', 'CLIPER']
LEARNED = ['linear regression', 'XGBoost', 'MLP', 'LSTM', 'GRU']
NEURAL = ['MLP', 'LSTM', 'GRU']  # scored by score_lstm(); sequence models on fewer hours
LABEL = {'persistence': 'Persistence', 'day-ago persistence': 'Day-ago persistence',
         'smart persistence': 'Smart persistence', 'CLIPER': 'CLIPER',
         'linear regression': 'Linear regression', 'XGBoost': 'XGBoost',
         'MLP': 'MLP', 'LSTM': 'LSTM', 'GRU': 'GRU'}
HNAME = {1: 'One', 3: 'Three', 6: 'Six', 24: 'TwentyFour'}  # macro-safe horizon names
SHORT = {'smart persistence': 'Smart', 'CLIPER': 'CLIPER', 'linear regression': 'LR',
         'XGBoost': 'XGB', 'MLP': 'MLP', 'LSTM': 'LSTM', 'GRU': 'GRU'}
REF, REF_COL = 'CLIPER', 'skill_vs_cliper'  # skill reference (Yang et al. 2020)

COLUMN_WIDTH = 3.3125
INK, MUTED, GRID = '#222222', '#555555', '#dddddd'
COLOR = {'CLIPER': '#52514e', 'linear regression': '#2a78d6',
         'XGBoost': '#eb6834', 'LSTM': '#1baf7a'}
MARKER = {'CLIPER': 's', 'linear regression': 'o', 'XGBoost': '^', 'LSTM': 'D'}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def read(prefix, name):
    path = Path(f'{prefix}_{name}.csv')
    return pd.read_csv(path) if path.exists() else None


def fmt_p(p):
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def fmt_offset(x):
    """UTC offset in hours: -5 -> UTC$-$5, -5.25 -> UTC$-$5.25."""
    x = float(x)
    return 'UTC' + fmt_signed(x, 0 if x.is_integer() else 2)


def fmt_signed(x, digits=1):
    s = f'{x:.{digits}f}'
    if float(s) == 0:  # no "-0.0"
        s = s.lstrip('-')
    return s.replace('-', '$-$')


def write(path, text):
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    print('wrote', path.relative_to(HERE))


# ---------------------------------------------------------------- tables

def main_table(res, horizons):
    """table*: nRMSE and skill (vs. CLIPER) per horizon; forecasts, then the target-hour setup."""
    col = ' & '.join(f'\\multicolumn{{2}}{{c}}{{$h={h}$\\,h}}' for h in horizons)
    rule = ' '.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(horizons)))
    sub = ' & '.join(['nRMSE & Skill'] * len(horizons))
    ncol = 1 + 2 * len(horizons)
    lines = ['\\begin{tabular}{l' + 'rr' * len(horizons) + '}', '\\toprule',
             f'Model & {col} \\\\', rule, f' & {sub} \\\\', '\\midrule',
             f'\\multicolumn{{{ncol}}}{{l}}{{\\emph{{Forecasts: issue-time information only}}}} \\\\']
    present = set(res['model'])
    causal = [m for m in BASELINES + LEARNED if m in present]
    best = {h: res[(res.h == h) & res.model.isin(causal)]['nRMSE_%'].min() for h in horizons}
    oracle = [m for m in LEARNED if m + ORACLE in present]
    for block, models, tag in [('causal', causal, ''), ('oracle', oracle, ORACLE)]:
        if block == 'oracle':
            lines += ['\\midrule', f'\\multicolumn{{{ncol}}}{{l}}'
                      '{\\emph{With target-hour irradiance (not a forecast)}} \\\\']
        for m in models:
            if block == 'causal' and m == LEARNED[0]:
                lines.append(f'\\cmidrule(lr){{1-{ncol}}}')
            cells = []
            for h in horizons:
                r = res[(res.h == h) & (res.model == m + tag)]
                if r.empty:
                    cells += ['--', '--']
                    continue
                n, sk = r['nRMSE_%'].iloc[0], 100 * r[REF_COL].iloc[0]
                n_txt = f'{n:.2f}'
                if block == 'causal' and np.isclose(n, best[h]):
                    n_txt = f'\\textbf{{{n_txt}}}'
                cells += [n_txt, '--' if m == REF else fmt_signed(sk)]
            name = LABEL[m] + ('$^\\dagger$' if m in ('LSTM', 'GRU') else '')
            lines.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write(TABLES / 'main_results.tex', '\n'.join(lines) + '\n')


def dm_table(dm, nn, horizons):
    """DM statistic per comparison and horizon, with its p-value on the line below."""
    rows = []
    for a, b in [('XGBoost', 'CLIPER'), ('linear regression', 'CLIPER'),
                 ('XGBoost', 'linear regression'), ('CLIPER', 'smart persistence')]:
        cells = [dm[(dm.h == h) & (dm.A == a) & (dm.B == b)] for h in horizons]
        if all(c.empty for c in cells):
            continue
        rows.append((f'{SHORT[a]} vs.\\ {SHORT[b]}',
                     [None if r.empty else (r.DM_stat.iloc[0], r.p_value.iloc[0]) for r in cells]))
    if nn is not None and 'DM_vs_cliper' in nn:
        for m in [m for m in NEURAL if m in set(nn.model)]:
            cells = [nn[(nn.h == h) & (nn.model == m)] for h in horizons]
            dag = '$^\\dagger$' if m != 'MLP' else ''
            rows.append((f'{m} vs.\\ CLIPER{dag}',
                         [None if r.empty else (r.DM_vs_cliper.iloc[0], r.p_vs_cliper.iloc[0]) for r in cells]))
    lines = ['\\begin{tabular}{l' + 'r' * len(horizons) + '}', '\\toprule',
             'Comparison & ' + ' & '.join(f'$h={h}$' for h in horizons) + ' \\\\', '\\midrule']
    for i, (name, cells) in enumerate(rows):
        lines.append(name + ' & ' + ' & '.join('--' if c is None else fmt_signed(c[0], 2)
                                               for c in cells) + ' \\\\')
        lines.append(' & ' + ' & '.join('' if c is None else f'\\scriptsize ({fmt_p(c[1])})'
                                        for c in cells) + ' \\\\' + ('[2pt]' if i < len(rows) - 1 else ''))
    lines += ['\\bottomrule', '\\end{tabular}']
    write(TABLES / 'dm.tex', '\n'.join(lines) + '\n')


def data_table(dropped):
    role = lambda y: 'train' if y <= 2020 else ('val' if y == 2021 else 'test')  # noqa: E731
    lines = ['\\begin{tabular}{llrrr}', '\\toprule',
             'Year & Split & Daytime h & Missing & Outage \\\\', '\\midrule']
    for _, r in dropped.iterrows():
        lines.append(f'{int(r.year)} & {role(int(r.year))} & {int(r.daytime):,} & '
                     f'{int(r.missing):,} & {int(r.outage):,} \\\\')
    t = dropped[['daytime', 'missing', 'outage']].sum()
    lines += ['\\midrule', f'All & & {int(t.daytime):,} & {int(t.missing):,} & {int(t.outage):,} \\\\',
              '\\bottomrule', '\\end{tabular}']
    write(TABLES / 'data_summary.tex', '\n'.join(lines).replace(',', '{,}') + '\n')


# ---------------------------------------------------------------- figures

def horizon_figure(res, horizons):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))
    x = np.arange(len(horizons))
    handles = []
    for m in ['CLIPER', 'linear regression', 'XGBoost', 'LSTM']:
        for tag, ls, suffix in [('', '-', ''), (ORACLE, '--', ' + target-hour irr.')]:
            ys = [res[(res.h == h) & (res.model == m + tag)]['nRMSE_%'] for h in horizons]
            if all(s.empty for s in ys):
                continue
            ys = [s.iloc[0] if not s.empty else np.nan for s in ys]
            (line,) = ax.plot(x, ys, ls=ls, lw=1.6, color=COLOR[m], marker=MARKER[m], ms=4.5,
                              mfc=COLOR[m] if tag == '' else 'white', mec=COLOR[m], mew=1.2,
                              label=LABEL[m] + suffix, zorder=3)
            handles.append(line)
    ax.set_xticks(x, [f'{h} h' for h in horizons])
    ax.set_xlabel('Forecast horizon')
    ax.set_ylabel('nRMSE (% of capacity)')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.tick_params(length=2.5, width=0.6)
    ax.legend(handles=handles, ncol=2, frameon=False, loc='lower center',
              bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / 'nrmse_by_horizon.pdf')
    plt.close(fig)
    print('wrote figures/nrmse_by_horizon.pdf')


def monthly_figure(monthly):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.8))
    cols = [c for c in monthly.columns if c != 'month']
    w = 0.38
    for i, (c, color) in enumerate(zip(cols, ['#2a78d6', '#eb6834'])):
        ax.bar(monthly['month'] + (i - 0.5) * (w + 0.02), monthly[c], width=w, color=color,
               label=c, zorder=3)
    ax.set_xticks(range(1, 13), ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.set_xlabel('Month')
    ax.set_ylabel('RMSE (kW)')
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.tick_params(length=2.5, width=0.6)
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / 'monthly_rmse.pdf')
    plt.close(fig)
    print('wrote figures/monthly_rmse.pdf')


# ---------------------------------------------------------------- in-text numbers

def numbers(res, align, dropped, horizons):
    defs = {}
    for h in horizons:
        k = HNAME.get(h)
        if k is None:
            continue
        for m in ['smart persistence', 'CLIPER', 'linear regression', 'XGBoost'] + NEURAL:
            for tag, t in [('', ''), (ORACLE, 'Oracle')]:
                r = res[(res.h == h) & (res.model == m + tag)]
                if r.empty:
                    continue
                name = SHORT[m]
                defs[f'NRMSE{t}{name}{k}'] = f'{r["nRMSE_%"].iloc[0]:.1f}'
                if 'R2' in r:
                    defs[f'RSq{t}{name}{k}'] = f'{r["R2"].iloc[0]:.3f}'
                if m != REF and REF_COL in r:
                    defs[f'Skill{t}{name}{k}'] = fmt_signed(100 * r[REF_COL].iloc[0])
                if m != 'smart persistence':
                    defs[f'SkillSmart{t}{name}{k}'] = fmt_signed(100 * r['skill_vs_smart'].iloc[0])
    r = res[(res.h == horizons[0]) & (res.model == 'smart persistence')]
    defs['TestHours'] = f'{int(r.n_hours.iloc[0]):,}'.replace(',', '{,}')
    if align is not None:
        clock = str(align['plant_clock'].iloc[0])
        defs['PlantClock'] = ('US Eastern time with daylight saving' if '/' in clock
                              else fmt_offset(clock))
        defs['AlignWinter'] = fmt_offset(align['best_utc_offset'].iloc[0])
        defs['AlignSummer'] = fmt_offset(align['best_utc_offset'].iloc[1])
        if 'best_nasa_shift_h' in align:
            defs['NasaShift'] = str(int(align['best_nasa_shift_h'].iloc[0]))
    if dropped is not None:
        t = dropped[['daytime', 'missing', 'outage']].sum()
        defs['MissingHours'] = f'{int(t.missing):,}'.replace(',', '{,}')
        defs['OutageHours'] = f'{int(t.outage):,}'.replace(',', '{,}')
        defs['DroppedShare'] = f'{100 * (t.missing + t.outage) / t.daytime:.1f}'
    body = '% generated by make_tables.py; do not edit\n' + ''.join(
        f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(defs.items()))
    write(TABLES / 'numbers.tex', body)


def main(prefix):
    res = read(prefix, 'results')
    if res is None:
        sys.exit(f'{prefix}_results.csv not found')
    dm, dropped = read(prefix, 'dm'), read(prefix, 'dropped')
    align, monthly, lstm = read(prefix, 'alignment'), read(prefix, 'monthly'), read(prefix, 'lstm')
    if lstm is not None:  # neural-model rows join the table; their reference rows don't
        nn = lstm[lstm.model.isin(NEURAL)]
        res = pd.concat([res, nn[[c for c in res.columns if c in nn]]], ignore_index=True)
    horizons = sorted(res.h.unique())
    FIGURES.mkdir(exist_ok=True)
    main_table(res, horizons)
    horizon_figure(res, horizons)
    if dm is not None:
        dm_table(dm, lstm, horizons)
    if dropped is not None:
        data_table(dropped)
    if monthly is not None:
        monthly_figure(monthly)
    numbers(res, align, dropped, horizons)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / 'results' / 'ai4ts'))
