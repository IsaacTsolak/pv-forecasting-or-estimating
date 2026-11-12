"""Recurrent and neural models for the paper, on the same data, features,
splits and scoring as ai4ts_fixes.run_all().

    python3 lstm_experiments.py select '/path/BS_*.csv'            # designs on 2021 validation
    python3 lstm_experiments.py final  '/path/BS_*.csv' lstm24_res  # chosen design, 3 seeds, 2022 test

Designs (all predict mean power over hour t+h from data up to the end of hour t):
    lstm24      LSTM over a 24-hour window, predicts power / capacity
    lstm72      the same over a 72-hour window (three days of history)
    gru24       GRU instead of LSTM
    lstm24_res  LSTM that predicts the correction to smart persistence
    mlp         feed-forward network on the issue-hour features only (no sequence)

Design choice uses the 2021 validation year only; the 2022 test year is touched
once, by `final`. Training uses hours with the sun up; validation and test
are scored on daytime hours (sun > 5 deg), as for the other models.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai4ts_fixes as fx  # noqa: E402

HERE = Path(__file__).resolve().parent
HORIZONS = (1, 3, 6, 24)
DESIGNS = {
    'lstm24':     dict(cell='lstm', seq_len=24, residual=False),
    'lstm72':     dict(cell='lstm', seq_len=72, residual=False),
    'gru24':      dict(cell='gru', seq_len=24, residual=False),
    'lstm24_res': dict(cell='lstm', seq_len=24, residual=True),
    'mlp':        dict(cell='mlp', seq_len=1, residual=False),
}
HIDDEN, DROPOUT, LR, BATCH, MAX_EPOCHS, PATIENCE = 64, 0.1, 1e-3, 256, 80, 10
torch.set_num_threads(int(__import__("os").environ.get("TORCH_THREADS", 8)))


class Net(nn.Module):
    def __init__(self, n_feat, cell):
        super().__init__()
        self.cell = cell
        if cell == 'mlp':
            self.body = nn.Sequential(nn.Linear(n_feat, 128), nn.ReLU(), nn.Dropout(DROPOUT),
                                      nn.Linear(128, HIDDEN), nn.ReLU())
        else:
            rnn = nn.LSTM if cell == 'lstm' else nn.GRU
            self.body = rnn(n_feat, HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        if self.cell == 'mlp':
            z = self.body(x[:, -1, :])
        else:
            z = self.body(x)[0][:, -1, :]  # hidden state at the issue hour
        return self.head(z).squeeze(-1)


def arrays(hourly, h, utc_offset, design):
    """Windows, targets and reference (smart persistence / capacity) per split."""
    out = fx.lstm_arrays(hourly, h, utc_offset, seq_len=design['seq_len'])
    res = {}
    for split, (X, y, rows) in out.items():
        ref = fx.baselines(rows)['smart persistence'] / fx.CAPACITY_KW
        sun = rows['cosz_target'].values > (0 if split == 'train' else fx.MIN_COS_ZENITH)
        res[split] = (X[sun], y[sun], ref[sun], rows[sun])
    return res


def train(data, design, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, rtr, _ = data['train']
    Xva, yva, rva, _ = data['val']
    off_tr = rtr if design['residual'] else 0 * rtr
    off_va = rva if design['residual'] else 0 * rva
    Xtr_t, ttr = torch.tensor(Xtr), torch.tensor((ytr - off_tr).astype('float32'))
    Xva_t = torch.tensor(Xva)
    model = Net(Xtr.shape[2], design['cell'])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, wait, epoch = np.inf, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr_t[b]), ttr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_rmse = rmse_kw(predict(model, Xva_t, off_va), yva)
        if val_rmse < best - 1e-3:
            best, wait = val_rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, epoch + 1


def predict(model, X, offset):
    model.eval()
    with torch.no_grad():
        X = X if torch.is_tensor(X) else torch.tensor(X)
        p = np.concatenate([model(X[i:i + 4096]).numpy() for i in range(0, len(X), 4096)])
    return np.clip(p + offset, 0, 1.1)


def rmse_kw(pred_scaled, y_scaled):
    return float(np.sqrt(np.mean((pred_scaled - y_scaled) ** 2)) * fx.CAPACITY_KW)


def select(pattern, utc_offset, out_dir, names=None, seed=0):
    """Validation scores for the given designs (all of them by default)."""
    hourly = fx.load_plant_hourly(pattern)
    rows = []
    names = names or list(DESIGNS)
    for name in names:
        design = DESIGNS[name]
        for h in HORIZONS:
            t0 = time.time()
            data = arrays(hourly, h, utc_offset, design)
            model, val_rmse, epochs = train(data, design, seed)
            _, yva, rva, _ = data['val']
            rows.append({'design': name, 'h': h, 'val_RMSE_kW': round(val_rmse, 1),
                         'val_smart_RMSE_kW': round(rmse_kw(rva, yva), 1),
                         'val_hours': len(yva), 'epochs': epochs, 'seconds': round(time.time() - t0)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df['val_skill'] = 1 - df['val_RMSE_kW'] / df['val_smart_RMSE_kW']
    df.to_csv(out_dir / f'lstm_selection_{"_".join(names)}.csv', index=False)
    return df


def summary(out_dir):
    df = pd.concat(pd.read_csv(f) for f in sorted(out_dir.glob('lstm_selection_*.csv')))
    pivot = df.pivot(index='design', columns='h', values='val_skill')
    pivot['mean'] = pivot.mean(axis=1)
    print('VALIDATION (2021) SKILL OVER SMART PERSISTENCE\n' + pivot.round(3).to_string())
    df.to_csv(out_dir / 'lstm_selection.csv', index=False)
    return pivot


LABELS = {'lstm24': 'LSTM', 'lstm72': 'LSTM', 'lstm24_res': 'LSTM', 'gru24': 'GRU', 'mlp': 'MLP'}


def final(pattern, utc_offset, out_dir, design_name, seeds=(0, 1, 2)):
    """Chosen design on the 2022 test year: one row per horizon with the mean of
    the seeds' scores, and the DM tests of the seed with the median RMSE.
    Smart persistence and CLIPER are rescored on exactly the same hours."""
    design, label = DESIGNS[design_name], LABELS[design_name]
    hourly = fx.load_plant_hourly(pattern)
    per_seed, rows = [], []
    for h in HORIZONS:
        data = arrays(hourly, h, utc_offset, design)
        lstm_test = fx.lstm_arrays(hourly, h, utc_offset, seq_len=design['seq_len'])['test']
        cliper_ref = fx.cliper_params(fx.split(fx.build_dataset(hourly, h, utc_offset))[0])
        Xall = torch.tensor(lstm_test[0])
        ref_all = np.asarray(fx.baselines(lstm_test[2])['smart persistence']) / fx.CAPACITY_KW
        off = ref_all if design['residual'] else 0 * ref_all
        scored = []
        for seed in seeds:
            model, _, epochs = train(data, design, seed)
            # score_lstm applies the daytime filter itself, so pass every test window
            r = fx.score_lstm(h, lstm_test, predict(model, Xall, off), name=label, cliper_ref=cliper_ref)
            r['seed'], r['epochs'], r['design'] = seed, epochs, design_name
            scored.append(r)
            per_seed.append(r)
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.iloc[0].items()}, flush=True)
        lstm_rows = pd.concat(scored)
        main = lstm_rows[lstm_rows.model == label].sort_values('RMSE_kW')
        median = main.iloc[len(main) // 2]
        row = {'h': h, 'model': label, 'design': design_name}
        for c in ['RMSE_kW', 'MAE_kW', 'nRMSE_%', 'R2', 'skill_vs_smart', 'skill_vs_cliper']:
            row[c] = main[c].mean()
        row['nRMSE_sd'] = main['nRMSE_%'].std()
        row['n_hours'] = int(median['n_hours'])
        for c in ['DM_vs_smart', 'p_vs_smart', 'DM_vs_cliper', 'p_vs_cliper', 'p_value']:
            row[c] = median[c]
        rows.append(row)
        first = scored[0]
        rows += first[first.model != label].assign(design=design_name).to_dict('records')
    pd.concat(per_seed).to_csv(out_dir / f'lstm_seeds_{design_name}.csv', index=False)
    out = pd.DataFrame(rows)
    print('\nTEST (2022)\n' + out.round(3).to_string(index=False))
    return out


if __name__ == '__main__':
    # python3 lstm_experiments.py select PATTERN [design ...]   (one process per design runs in parallel)
    # python3 lstm_experiments.py summary
    # python3 lstm_experiments.py final PATTERN design
    stage = sys.argv[1]
    out_dir = HERE / 'results'
    out_dir.mkdir(exist_ok=True)
    if stage == 'select':
        select(sys.argv[2], -5.0, out_dir, sys.argv[3:] or None)
    elif stage == 'summary':
        summary(out_dir)
    else:
        name = sys.argv[3]
        final(sys.argv[2], -5.0, out_dir, name).to_csv(out_dir / f'lstm_final_{name}.csv', index=False)
