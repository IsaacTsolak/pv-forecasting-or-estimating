"""Forecasting setup, baselines and evaluation for the paper.

    python3 -c "import ai4ts_fixes as fx; fx.run_all('data/BS_*.csv', 'data/POWER_*_LST.csv')"

What it does:
  * Every input is known when the forecast is issued (end of hour t); the target
    is mean kW over hour t+h.
  * The same models are also run with irradiance measured during the target
    hour added (the target-hour setup), to measure how much accuracy that
    input accounts for.
  * Hours are kept on a complete hourly grid, so shift()-based lags never jump
    across a gap, and gaps are not interpolated into the evaluation.
  * Baselines: persistence, day-ago persistence, smart (clear-sky) persistence
    and CLIPER; skill is reported against CLIPER.
  * Scored on daytime hours only (sun more than 5 degrees up), with nRMSE.
  * Diebold-Mariano test with a Newey-West variance on daily loss differentials.
  * make_sequences() and lstm_arrays() build windows for the neural models in
    lstm_experiments.py; score_lstm() scores them on the same footing.

run_all() saves ai4ts_results.csv, _dm.csv, _dropped.csv, _alignment.csv and
_monthly.csv under out_prefix; make_tables.py turns them into the paper's
tables and figures.
"""
import glob

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # only for running without xgboost installed
    XGBRegressor = None
    from sklearn.ensemble import HistGradientBoostingRegressor

LAT, LON = 37.76, -84.85                  # plant location (nearest NASA POWER grid point)
CAPACITY_KW = 10_000                      # 10 MW plant (max observed 10,212 kW)
MIN_COS_ZENITH = np.cos(np.radians(85))   # score only when the sun is > 5 deg up
ORACLE_COLS = ['POAI_target', 'GHI_target']


# ---------------------------------------------------------------- data loading

def load_plant_hourly(pattern):
    """Minute CSVs -> complete hourly grid (hour-start labels). Gaps stay NaN."""
    cols = ['Timestamp', 'kW', 'POAI', 'GHI', 'TmpC']
    raw = pd.concat([pd.read_csv(f, usecols=cols, parse_dates=['Timestamp'])
                     for f in sorted(glob.glob(pattern))])
    raw = raw.drop_duplicates('Timestamp').set_index('Timestamp').sort_index()
    hourly = raw.resample('h').mean()
    counts = raw[['kW', 'POAI']].resample('h').count()
    for c in ['kW', 'POAI', 'GHI']:
        hourly[c] = hourly[c].clip(lower=0)
    hourly['valid'] = (counts['kW'] >= 45) & (counts['POAI'] >= 45)
    # plant off or curtailed while the sun is out: kept out of training and scoring
    hourly['outage'] = (hourly['POAI'] > 100) & (hourly['kW'] < 0.02 * CAPACITY_KW)
    return hourly


def load_nasa(pattern):
    """NASA POWER hourly CSVs, header length detected instead of skiprows=14."""
    frames = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            skip = next(i for i, line in enumerate(fh) if '-END HEADER-' in line) + 1
        frames.append(pd.read_csv(f, skiprows=skip))
    nasa = pd.concat(frames)
    nasa.index = pd.to_datetime(dict(year=nasa.YEAR, month=nasa.MO, day=nasa.DY, hour=nasa.HR))
    nasa = nasa.drop(columns=['YEAR', 'MO', 'DY', 'HR']).replace(-999, np.nan)
    return nasa[~nasa.index.duplicated()].sort_index()


# ---------------------------------------------------------------- solar geometry

def _to_utc(index, utc_offset):
    idx = pd.DatetimeIndex(index)
    if isinstance(utc_offset, str):  # e.g. 'America/New_York' if the plant clock follows DST
        idx = idx.tz_localize(utc_offset, ambiguous='NaT', nonexistent='shift_forward')
        return idx.tz_convert('UTC').tz_localize(None)
    return idx - pd.Timedelta(hours=utc_offset)


def solar_geometry(index, utc_offset):
    """cos(solar zenith) at mid-hour and Haurwitz clear-sky GHI (W/m2).

    NOAA solar-position equations; `index` holds hour-start timestamps on a
    clock that is UTC+utc_offset (or a tz name). pvlib gives the same numbers.
    """
    t = _to_utc(index, utc_offset) + pd.Timedelta(minutes=30)
    jc = (np.asarray(t.to_julian_date()) - 2451545.0) / 36525.0
    rad = np.radians
    l0 = (280.46646 + jc * (36000.76983 + 0.0003032 * jc)) % 360
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    c = (np.sin(rad(m)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(rad(2 * m)) * (0.019993 - 0.000101 * jc)
         + np.sin(rad(3 * m)) * 0.000289)
    omega = 125.04 - 1934.136 * jc
    app_long = l0 + c - 0.00569 - 0.00478 * np.sin(rad(omega))
    obliq = (23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
             + 0.00256 * np.cos(rad(omega)))
    decl = np.arcsin(np.sin(rad(obliq)) * np.sin(rad(app_long)))
    y = np.tan(rad(obliq / 2)) ** 2
    eot = 4 * np.degrees(y * np.sin(2 * rad(l0)) - 2 * e * np.sin(rad(m))
                         + 4 * e * y * np.sin(rad(m)) * np.cos(2 * rad(l0))
                         - 0.5 * y * y * np.sin(4 * rad(l0))
                         - 1.25 * e * e * np.sin(2 * rad(m)))  # equation of time, minutes
    minutes_utc = np.asarray(t.hour * 60 + t.minute, dtype=float)
    hour_angle = rad((minutes_utc + eot + 4 * LON) / 4 - 180)
    cosz = (np.sin(rad(LAT)) * np.sin(decl)
            + np.cos(rad(LAT)) * np.cos(decl) * np.cos(hour_angle))
    cosz = np.clip(cosz, 0, 1)
    ghi_cs = np.where(cosz > 0, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 1e-3)), 0.0)
    return pd.DataFrame({'cosz': cosz, 'ghi_cs': ghi_cs}, index=index)


def check_time_alignment(hourly, nasa=None):
    """Which UTC offset lines plant GHI up with the sun, in winter and in summer,
    and how many hours NASA POWER (LST = longitude band, likely UTC-6 here) is
    shifted from the plant. Same offset in both seasons -> plant clock ignores DST."""
    offsets = np.arange(-7, -2.75, 0.25)
    rows = []
    for name, months in [('winter (Dec-Feb)', [12, 1, 2]), ('summer (May-Aug)', [5, 6, 7, 8])]:
        sub = hourly[hourly.index.month.isin(months) & hourly['valid']].dropna(subset=['GHI'])
        corr = [np.corrcoef(sub['GHI'], solar_geometry(sub.index, o)['ghi_cs'])[0, 1]
                for o in offsets]
        row = {'period': name, 'best_utc_offset': float(offsets[int(np.argmax(corr))])}
        if nasa is not None:
            lags = list(range(-3, 4))
            nc = [sub['GHI'].corr(nasa['ALLSKY_SFC_SW_DWN'].shift(k).reindex(sub.index))
                  for k in lags]
            row['best_nasa_shift_h'] = lags[int(np.nanargmax(nc))]
        rows.append(row)
    out = pd.DataFrame(rows)
    print('\nTIME ALIGNMENT\n' + out.to_string(index=False))
    return out


# ---------------------------------------------------------------- features

def build_dataset(hourly, h, utc_offset, nasa=None, nasa_shift=0, dropna=True):
    """One row per issue time t (hour [t, t+1) has just ended); y = mean kW over hour t+h.

    All features are known at issue time except ORACLE_COLS, the irradiance
    measured during the target hour (the target-hour setup).
    """
    assert 1 <= h <= 24, 'day-ago feature needs h <= 24'
    d = hourly.copy()
    geo = solar_geometry(d.index, utc_offset)
    d['cosz'], d['ghi_cs'] = geo['cosz'], geo['ghi_cs']
    with np.errstate(divide='ignore', invalid='ignore'):  # clear-sky index, 0 at night
        d['kt'] = np.where(d['ghi_cs'] > 50, d['GHI'] / d['ghi_cs'], 0.0).clip(0, 1.5)
    tgt = d.shift(-h)                       # complete grid, so shift(-h) == h hours later
    target_time = d.index + pd.Timedelta(hours=h)

    X = pd.DataFrame(index=d.index)
    for lag in range(3):
        X[f'kW_lag{lag}'] = d['kW'].shift(lag)
    for c in ['POAI', 'GHI', 'kt']:
        X[f'{c}_lag0'] = d[c]
        X[f'{c}_lag1'] = d[c].shift(1)
    X['TmpC_lag0'] = d['TmpC']
    X['cosz_lag0'], X['ghi_cs_lag0'] = d['cosz'], d['ghi_cs']
    X['kW_sameHourYesterday'] = d['kW'].shift(24 - h)       # kW(t+h-24)
    X['cosz_target'], X['ghi_cs_target'] = tgt['cosz'], tgt['ghi_cs']
    X['hour_sin'] = np.sin(2 * np.pi * target_time.hour / 24)
    X['hour_cos'] = np.cos(2 * np.pi * target_time.hour / 24)
    X['doy_sin'] = np.sin(2 * np.pi * target_time.dayofyear / 365.25)
    X['doy_cos'] = np.cos(2 * np.pi * target_time.dayofyear / 365.25)
    if nasa is not None:  # not available in real time, so report it as an ablation
        w = nasa[['PS', 'RH2M', 'WS10M', 'ALLSKY_SFC_UV_INDEX']].shift(nasa_shift)
        w = w.reindex(d.index).fillna({'ALLSKY_SFC_UV_INDEX': 0})
        for col in w.columns:
            X[f'nasa_{col}'] = w[col]
    X['POAI_target'], X['GHI_target'] = tgt['POAI'], tgt['GHI']

    X['y'] = tgt['kW']
    X['target_time'] = target_time
    X['ok'] = (d['valid'] & d['valid'].shift(-h, fill_value=False)
               & ~d['outage'].shift(-h, fill_value=True)
               & X.drop(columns=['target_time']).notna().all(axis=1))
    return X[X['ok']].drop(columns='ok') if dropna else X


def feature_columns(data, oracle=False):
    drop = {'y', 'target_time', 'ok'} | (set() if oracle else set(ORACLE_COLS))
    return [c for c in data.columns if c not in drop]


# ---------------------------------------------------------------- models and scoring

def baselines(data):
    p = data['kW_lag0'].values
    day_ago = data['kW_sameHourYesterday'].values
    # clear-sky persistence: keep the current clear-sky index, roll the sun forward
    ratio = data['ghi_cs_target'].values / np.maximum(data['ghi_cs_lag0'].values, 1e-6)
    smart = np.where(data['cosz_lag0'].values > 0.1, p * ratio, day_ago)
    return {'persistence': p,
            'day-ago persistence': day_ago,
            'smart persistence': np.clip(smart, 0, 1.1 * CAPACITY_KW)}


def cliper_params(train):
    """Climatology and lag-h autocorrelation of the clear-sky power index
    k = P / GHI_cs, from training hours with the sun well up at both ends."""
    ok = (train['cosz_lag0'] > 0.1) & (train['cosz_target'] > 0.1)
    k_now = train.loc[ok, 'kW_lag0'] / train.loc[ok, 'ghi_cs_lag0']
    k_tgt = train.loc[ok, 'y'] / train.loc[ok, 'ghi_cs_target']
    return {'k_mean': float(k_tgt.mean()), 'gamma': float(np.corrcoef(k_now, k_tgt)[0, 1])}


def cliper(data, params):
    """Optimal convex combination of climatology and persistence (Yang et al.
    2020) in clear-sky-index space: k(t+h) = gamma k(t) + (1 - gamma) mean(k).
    Falls back to climatology when the sun is too low at issue time."""
    k_now = data['kW_lag0'].values / np.maximum(data['ghi_cs_lag0'].values, 1e-6)
    k_hat = params['gamma'] * k_now + (1 - params['gamma']) * params['k_mean']
    k_hat = np.where(data['cosz_lag0'].values > 0.1, k_hat, params['k_mean'])
    return np.clip(k_hat * data['ghi_cs_target'].values, 0, 1.1 * CAPACITY_KW)


def fit_models(train, val, feats):
    models = {'linear regression': make_pipeline(StandardScaler(), LinearRegression())
              .fit(train[feats], train['y'])}
    if XGBRegressor is not None:
        gbm = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8,
                           early_stopping_rounds=100, random_state=42, n_jobs=-1)
        gbm.fit(train[feats], train['y'], eval_set=[(val[feats], val['y'])], verbose=False)
    else:
        gbm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                            random_state=42).fit(train[feats], train['y'])
    models['XGBoost'] = gbm
    return models


def predict(model, data, feats):
    return np.clip(model.predict(data[feats]), 0, 1.1 * CAPACITY_KW)


def score(y, pred):
    y = np.asarray(y)
    err = np.asarray(pred) - y
    rmse = np.sqrt(np.mean(err ** 2))
    return {'RMSE_kW': rmse, 'MAE_kW': np.mean(np.abs(err)),
            'nRMSE_%': 100 * rmse / CAPACITY_KW, 'R2': 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2),
            'n_hours': len(err)}


def dm_test(y, pred_a, pred_b, days, lag=7):
    """Diebold-Mariano test on daily sums of squared-error differences, which
    absorbs within-day autocorrelation; Newey-West variance over `lag` days.
    Negative statistic: A is more accurate than B."""
    y = np.asarray(y)
    loss = (np.asarray(pred_a) - y) ** 2 - (np.asarray(pred_b) - y) ** 2
    d = pd.Series(loss).groupby(np.asarray(days)).sum().values
    n, dc = len(d), d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def split(data):
    yr = data['target_time'].dt.year
    sun = data['cosz_target'] > 0
    train, val = data[(yr <= 2020) & sun], data[(yr == 2021) & sun]
    test = data[(yr == 2022) & (data['cosz_target'] > MIN_COS_ZENITH)]
    return train, val, test


def run_all(plant_glob, nasa_glob=None, horizons=(1, 3, 6, 24), utc_offset=None,
            use_nasa=False, out_prefix='results/ai4ts'):
    hourly = load_plant_hourly(plant_glob)
    nasa = load_nasa(nasa_glob) if nasa_glob else None
    align = check_time_alignment(hourly, nasa)
    if utc_offset is None:
        winter, summer = align['best_utc_offset']
        utc_offset = 'America/New_York' if abs(summer - winter - 1) <= 0.25 else winter
        print(f'Using plant clock = {utc_offset}')
    nasa_shift = int(align['best_nasa_shift_h'].iloc[0]) if nasa is not None else 0
    if nasa is not None and align['best_nasa_shift_h'].nunique() > 1:
        print('WARNING: NASA shift differs by season; convert both to UTC before using NASA.')

    yrs = hourly.index.year
    sunny = solar_geometry(hourly.index, utc_offset)['cosz'] > MIN_COS_ZENITH
    dropped = pd.DataFrame({'daytime': sunny.groupby(yrs).sum(),
                            'missing': (~hourly['valid'] & sunny).groupby(yrs).sum(),
                            'outage': (hourly['outage'] & sunny).groupby(yrs).sum()})
    dropped.index.name = 'year'
    print('\nDAYTIME HOURS DROPPED (missing data / outage) BY YEAR\n' + dropped.to_string())
    align.assign(plant_clock=str(utc_offset)).to_csv(f'{out_prefix}_alignment.csv', index=False)
    dropped.to_csv(f'{out_prefix}_dropped.csv')

    rows, dm_rows = [], []
    for h in horizons:
        data = build_dataset(hourly, h, utc_offset, nasa if use_nasa else None, nasa_shift)
        train, val, test = split(data)
        days = test['target_time'].dt.date.values
        preds = baselines(test)
        preds['CLIPER'] = cliper(test, cliper_params(train))
        for oracle in (False, True):
            feats = feature_columns(data, oracle)
            tag = ' + target-hour irradiance' if oracle else ''
            for name, model in fit_models(train, val, feats).items():
                preds[name + tag] = predict(model, test, feats)
                if oracle and h == horizons[0] and name == 'XGBoost':
                    monthly = _monthly_rmse(model, val, test, feats)
        ref = score(test['y'], preds['smart persistence'])['RMSE_kW']
        ref_cliper = score(test['y'], preds['CLIPER'])['RMSE_kW']
        for name, p in preds.items():
            rows.append({'h': h, 'model': name, **score(test['y'], p)})
            rows[-1]['skill_vs_smart'] = 1 - rows[-1]['RMSE_kW'] / ref
            rows[-1]['skill_vs_cliper'] = 1 - rows[-1]['RMSE_kW'] / ref_cliper
        for a, b in [('XGBoost', 'smart persistence'), ('XGBoost', 'linear regression'),
                     ('linear regression', 'smart persistence'), ('XGBoost', 'CLIPER'),
                     ('linear regression', 'CLIPER'), ('CLIPER', 'smart persistence')]:
            s, pv = dm_test(test['y'], preds[a], preds[b], days)
            dm_rows.append({'h': h, 'A': a, 'B': b, 'DM_stat': s, 'p_value': pv})

    results, dm = pd.DataFrame(rows), pd.DataFrame(dm_rows)
    pd.set_option('display.width', 160)
    print('\nRESULTS (2022 test year, daytime hours)\n' + results.round(3).to_string(index=False))
    print('\nDIEBOLD-MARIANO (negative = A better)\n' + dm.round(4).to_string(index=False))
    print(f'\nMONTHLY RMSE, XGBoost with target-hour irradiance, h={horizons[0]} '
          '(look for the 2021 months behind the validation/test gap)\n' + monthly.round(0).to_string())
    results.to_csv(f'{out_prefix}_results.csv', index=False)
    dm.to_csv(f'{out_prefix}_dm.csv', index=False)
    monthly.to_csv(f'{out_prefix}_monthly.csv', index_label='month')
    return results, dm


def _monthly_rmse(model, val, test, feats):
    out = {}
    for name, df in [('2021 (val)', val[val['cosz_target'] > MIN_COS_ZENITH]), ('2022 (test)', test)]:
        err = pd.Series(predict(model, df, feats) - df['y'].values, index=df['target_time'].values)
        out[name] = np.sqrt((err ** 2).groupby(err.index.month).mean())
    return pd.DataFrame(out)


# ---------------------------------------------------------------- LSTM

def make_sequences(X, y, seq_len=24):
    """Fixed LSTM windows: the window ENDS at row i and is paired with y[i], the
    target for issue time i. (Ending the window at i-1 would turn a
    "1-hour" LSTM into a 2-hour forecast.) X and y must be on the complete hourly
    grid, NaN where missing; windows that touch a gap are dropped."""
    X, y = np.asarray(X, dtype='float32'), np.asarray(y, dtype='float32')
    win = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0).transpose(0, 2, 1)
    tgt = y[seq_len - 1:]
    keep = ~np.isnan(win).any(axis=(1, 2)) & ~np.isnan(tgt)
    return win[keep], tgt[keep], np.flatnonzero(keep) + seq_len - 1


def lstm_arrays(hourly, h, utc_offset, seq_len=24):
    """Train/val/test LSTM windows with the same inputs as the other models.
    The target is scaled to capacity (training on raw kW converges much more
    slowly); multiply predictions by
    CAPACITY_KW and score them only where cosz_target > MIN_COS_ZENITH."""
    data = build_dataset(hourly, h, utc_offset, dropna=False)
    feats = feature_columns(data)
    yr = data['target_time'].dt.year.values
    ok = data['ok'].values.astype(bool)
    sc = StandardScaler().fit(data.loc[ok & (yr <= 2020), feats])
    y = np.where(ok, data['y'] / CAPACITY_KW, np.nan)
    Xs, ys, rows = make_sequences(sc.transform(data[feats]), y, seq_len)
    out = {}
    for name, mask in [('train', yr <= 2020), ('val', yr == 2021), ('test', yr == 2022)]:
        k = mask[rows]
        out[name] = (Xs[k], ys[k], data.iloc[rows[k]])
    return out


def score_lstm(h, test, pred_scaled, name='LSTM', cliper_ref=None):
    """Score LSTM test predictions (the model's raw output, scaled by capacity)
    on daytime hours, next to smart persistence and, if `cliper_ref` (from
    cliper_params(train)) is given, CLIPER on exactly the same hours, with DM
    tests against both. `test` is lstm_arrays(...)['test']. Collect the
    returned rows for every horizon and save them as results/ai4ts_lstm.csv."""
    rows = test[2]
    day = rows['cosz_target'].values > MIN_COS_ZENITH
    y = rows['y'].values[day]
    days = rows['target_time'].dt.date.values[day]
    pred = np.clip(np.asarray(pred_scaled).ravel()[day] * CAPACITY_KW, 0, 1.1 * CAPACITY_KW)
    refs = {'smart persistence': baselines(rows[day])['smart persistence']}
    if cliper_ref is not None:
        refs['CLIPER'] = cliper(rows[day], cliper_ref)
    out = [{'h': h, 'model': name, **score(y, pred)}]
    out += [{'h': h, 'model': f'{r} (LSTM hours)', **score(y, p)} for r, p in refs.items()]
    for r, p in refs.items():
        key = 'smart' if r == 'smart persistence' else 'cliper'
        rmse_ref = score(y, p)['RMSE_kW']
        for row in out:
            row[f'skill_vs_{key}'] = 1 - row['RMSE_kW'] / rmse_ref
        stat, pv = dm_test(y, pred, p, days)
        out[0][f'DM_vs_{key}'], out[0][f'p_vs_{key}'] = stat, pv
    out[0]['p_value'] = out[0]['p_vs_smart']  # backward compatible column
    return pd.DataFrame(out)
