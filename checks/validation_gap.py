"""Why time-interpolated data makes 2021 (validation) look much harder than 2022 (test).

    python3 checks/validation_gap.py '/path/to/BS_*.csv' [utc_offset]

1. Lists the long runs of missing plane-of-array (POA) readings in each year.
2. Trains XGBoost with target-hour irradiance at h=1 on clean 2016-2020 hours,
   then scores daytime hours of 2021 and 2022 two ways:
     clean:        hours with missing data or outages removed (as run_all does)
     interpolated: gaps filled by time interpolation and kept, outages kept
                   (a common preprocessing step)
   Smart persistence is scored the same way for reference; it uses only power,
   which has almost no gaps.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ai4ts_fixes as fx  # noqa: E402


def poa_gaps(pattern, min_hours=6):
    rows = []
    for f in sorted(glob.glob(pattern)):
        raw = pd.read_csv(f, usecols=['Timestamp', 'POAI'], parse_dates=['Timestamp'])
        miss = raw['POAI'].isna().values
        run_id = np.cumsum(np.r_[True, miss[1:] != miss[:-1]])
        for _, g in raw[miss].groupby(run_id[miss]):
            hours = len(g) / 60
            if hours >= min_hours:
                rows.append({'start': g['Timestamp'].iloc[0], 'end': g['Timestamp'].iloc[-1],
                             'hours': round(hours, 1)})
        print(f'{Path(f).name}: {miss.sum():,} minutes without POA')
    return pd.DataFrame(rows)


def main(pattern, utc_offset=-5.0):
    gaps = poa_gaps(pattern)
    print('\nPOA GAPS OF 6 HOURS OR MORE\n' + gaps.to_string(index=False))

    hourly = fx.load_plant_hourly(pattern)
    data = fx.build_dataset(hourly, 1, utc_offset)
    train, val, test = fx.split(data)
    feats = fx.feature_columns(data, oracle=True)
    model = fx.fit_models(train, val, feats)['XGBoost']

    interp = hourly.copy()
    for c in ['kW', 'POAI', 'GHI', 'TmpC']:
        interp[c] = interp[c].interpolate(method='time')
    interp['valid'], interp['outage'] = True, False
    filled = fx.build_dataset(interp, 1, utc_offset)

    rows = []
    for name, d in [('clean', data), ('interpolated', filled)]:
        yr = d['target_time'].dt.year
        for year in (2021, 2022):
            s = d[(yr == year) & (d['cosz_target'] > fx.MIN_COS_ZENITH)]
            xgb = fx.score(s['y'], fx.predict(model, s, feats))
            smart = fx.score(s['y'], fx.baselines(s)['smart persistence'])
            rows.append({'data': name, 'year': year, 'n_hours': xgb['n_hours'],
                         'XGB_target_hour_RMSE': round(xgb['RMSE_kW']),
                         'XGB_target_hour_R2': round(xgb['R2'], 3),
                         'smart_persistence_RMSE': round(smart['RMSE_kW'])})
    out = pd.DataFrame(rows)
    print('\nDAYTIME RMSE (kW), h=1\n' + out.to_string(index=False))
    for name in ('clean', 'interpolated'):
        r = out[out.data == name].set_index('year')['XGB_target_hour_RMSE']
        print(f'{name}: 2021 / 2022 RMSE ratio = {r[2021] / r[2022]:.2f}')
    return gaps, out


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else -5.0)
