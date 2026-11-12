"""Two data checks quoted in the Data and Results sections.

    python3 checks/temperature_and_snow.py '/path/BS_*.csv' /path/POWER_*_LST.csv /path/lex_daily.csv

1. Is the plant's TmpC air or module temperature? Compare it with NASA POWER
   2 m air temperature (moved to the plant clock): module temperature runs
   20-30 C above air temperature in full sun and slightly below it at night.
2. Does snow explain the winter peaks in monthly error? Score the target-hour
   XGBoost model (h=1) on winter daytime hours of 2021-2022, split by whether
   snow covered the ground that day at the nearest NOAA station, Lexington
   Blue Grass Airport (GHCN-Daily USW00093820, ~37 km from the plant; SNWD >= 10 mm).
   Download: https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries
   &stations=USW00093820&startDate=2016-09-01&endDate=2022-12-31&dataTypes=SNOW,SNWD&format=csv&units=metric
   (MERRA-2 snow depth from NASA POWER was tried first; at 0.5 deg it shows snow
   on the ground for almost all of January in both years, which is implausible.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ai4ts_fixes as fx  # noqa: E402

PLANT_UTC, NASA_UTC = -5, -6  # established by the clock check


def load_station(path):
    return pd.read_csv(path, parse_dates=['DATE']).set_index('DATE')


def temperature(hourly, nasa):
    air = nasa['T2M'].copy()
    air.index = air.index + pd.Timedelta(hours=PLANT_UTC - NASA_UTC)  # NASA LST -> plant clock
    d = pd.DataFrame({'TmpC': hourly['TmpC'], 'air': air, 'POAI': hourly['POAI']}).dropna()
    d['diff'] = d['TmpC'] - d['air']
    bins = pd.cut(d['POAI'], [-1, 5, 200, 400, 600, 800, 1000, 1400])
    out = d.groupby(bins, observed=True)['diff'].agg(['mean', 'std', 'count']).round(2)
    print('TmpC MINUS NASA 2 m AIR TEMPERATURE, BY POA IRRADIANCE (W/m2)\n' + out.to_string())
    slope = np.polyfit(d.loc[d.POAI > 50, 'POAI'], d.loc[d.POAI > 50, 'diff'], 1)[0]
    print(f'slope: {slope * 1000:.1f} C per 1000 W/m2; corr(TmpC, air) = {d.TmpC.corr(d.air):.3f}')


def snow(hourly, daily):
    data = fx.build_dataset(hourly, 1, PLANT_UTC)
    train, val, test = fx.split(data)
    feats = fx.feature_columns(data, oracle=True)
    model = fx.fit_models(train, val, feats)['XGBoost']
    winter = pd.concat([val[val.cosz_target > fx.MIN_COS_ZENITH], test])
    winter = winter[winter.target_time.dt.month.isin([12, 1, 2, 3])]
    day = winter.target_time.dt.normalize()
    depth = daily['SNWD'].reindex(day.values).values
    err = fx.predict(model, winter, feats) - winter['y'].values
    rows = []
    for name, m in [('snow on the ground', depth >= 10), ('no snow', depth < 10)]:
        rows.append({'days': name, 'hours': int(m.sum()), 'n_days': day[m].nunique(),
                     'RMSE_kW': round(float(np.sqrt(np.mean(err[m] ** 2))))})
    print('\nTARGET-HOUR XGBOOST, h=1, WINTER (DEC-MAR) DAYTIME HOURS 2021-2022\n'
          + pd.DataFrame(rows).to_string(index=False))
    for name, df in [('2021 (val)', val), ('2022 (test)', test)]:
        df = df[df.cosz_target > fx.MIN_COS_ZENITH]
        dd = df.target_time.dt.normalize()
        snowy = daily['SNWD'].reindex(dd.values).values >= 10
        e = fx.predict(model, df, feats) - df['y'].values
        print(f'{name}: RMSE all {np.sqrt(np.mean(e ** 2)):.0f} kW, '
              f'without snow days {np.sqrt(np.mean(e[~snowy] ** 2)):.0f} kW '
              f'({dd[snowy].nunique()} snow days, {snowy.sum()} hours)')
    sw = daily.loc['2021':'2022', 'SNWD']
    print('\nStation days with snow on the ground by month (2021-2022):')
    print(sw[sw >= 10].groupby([sw[sw >= 10].index.year, sw[sw >= 10].index.month]).size().to_string())


if __name__ == '__main__':
    hourly = fx.load_plant_hourly(sys.argv[1])
    temperature(hourly, fx.load_nasa(sys.argv[2]))
    snow(hourly, load_station(sys.argv[3]))
