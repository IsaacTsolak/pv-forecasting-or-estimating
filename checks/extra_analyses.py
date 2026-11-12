"""Two follow-up checks quoted in the Results section.

    python3 checks/extra_analyses.py '/path/to/BS_*.csv' [utc_offset]

1. Why the target-hour model is better at h=1 than at longer horizons: refit it
   without any power features (lags and same hour yesterday). If its error at
   h=1 rises to the long-horizon level, recent power was carrying plant state
   (availability, snow, curtailment) that irradiance alone does not.
2. DM test vs. a naive hourly t-test, for the same pairs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import ai4ts_fixes as fx  # noqa: E402

POWER_FEATS = ['kW_lag0', 'kW_lag1', 'kW_lag2', 'kW_sameHourYesterday']


def main(pattern, utc_offset=-5.0):
    hourly = fx.load_plant_hourly(pattern)
    rows, tests = [], []
    for h in (1, 3, 6, 24):
        data = fx.build_dataset(hourly, h, utc_offset)
        train, val, test = fx.split(data)
        all_feats = fx.feature_columns(data, oracle=True)
        for label, feats in [('target-hour, all features', all_feats),
                             ('target-hour, no power features',
                              [f for f in all_feats if f not in POWER_FEATS])]:
            m = fx.fit_models(train, val, feats)['XGBoost']
            s = fx.score(test['y'], fx.predict(m, test, feats))
            rows.append({'h': h, 'model': label, 'nRMSE_%': round(s['nRMSE_%'], 2),
                         'R2': round(s['R2'], 3)})
        causal = fx.feature_columns(data)
        models = fx.fit_models(train, val, causal)
        preds = {'smart persistence': fx.baselines(test)['smart persistence'],
                 'CLIPER': fx.cliper(test, fx.cliper_params(train)),
                 **{k: fx.predict(v, test, causal) for k, v in models.items()}}
        y, days = test['y'].values, test['target_time'].dt.date.values
        for a, b in [('XGBoost', 'smart persistence'), ('linear regression', 'smart persistence'),
                     ('XGBoost', 'linear regression'), ('XGBoost', 'CLIPER'),
                     ('linear regression', 'CLIPER')]:
            d = (preds[a] - y) ** 2 - (preds[b] - y) ** 2
            dm, p_dm = fx.dm_test(y, preds[a], preds[b], days)
            t, p_t = stats.ttest_1samp(d, 0)
            tests.append({'h': h, 'A': a, 'B': b, 'DM': round(dm, 2), 'p_DM': round(p_dm, 4),
                          'naive_t': round(t, 2), 'p_naive': p_t})
    print('TARGET-HOUR MODEL WITH AND WITHOUT POWER FEATURES\n' + pd.DataFrame(rows).to_string(index=False))
    print('\nDM TEST VS NAIVE HOURLY T-TEST\n' + pd.DataFrame(tests).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else -5.0)
