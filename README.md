# Target-hour irradiance leakage in hourly solar power forecasting

Code for the paper *Forecasting or Estimating? Target-Hour Irradiance Leakage in Hourly Solar Power Forecasting*. It reproduces every table, figure and in-text number from public data.

## Setup

Python 3.11. Install the pinned versions used for the paper:

```
pip install -r requirements.txt
```

XGBoost results can differ slightly across CPU platforms (for example Apple Silicon vs. x86), by up to a few tenths of a point of nRMSE. None of the paper's conclusions change between platforms we tested.

## Data

All three sources are public. Put the downloaded files in `data/`.

**1. Plant measurements (required).** One-minute records from the E.W. Brown 10 MW solar facility, published by LG&E and KU Energy at https://lge-ku.com/live-solar-generation/historical-data. Download the yearly files for 2016 through 2022 and save them as `data/BS_2016.csv` … `data/BS_2022.csv`. The record starts on 1 September 2016.

**2. NASA POWER hourly data (clock check and temperature check).** From the hourly point API (https://power.larc.nasa.gov/docs/services/api/temporal/hourly/), for latitude 37.76, longitude −84.85, 2016-09-01 to 2022-12-31, in local solar time (LST), with at least the parameters `ALLSKY_SFC_SW_DWN` and `T2M`:

```
https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE&latitude=37.76&longitude=-84.85&start=20160901&end=20221231&format=CSV&time-standard=LST
```

Save it as `data/POWER_<anything>_LST.csv`. If the API limits the date range, download it in pieces; any number of `POWER_*_LST.csv` files works. Keep UTC downloads out of this pattern, since the loader merges every matching file.

**3. NOAA snow depth (snow check only).** GHCN-Daily, station USW00093820 (Lexington Blue Grass Airport, KY):

```
https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00093820&startDate=2016-09-01&endDate=2022-12-31&dataTypes=SNOW,SNWD&format=csv&units=metric
```

Save it as `data/lex_daily.csv`.

## Running

Run everything from the repository root.

**Main results** (Tables 1–3, Figures 1–2; under a minute on a laptop CPU):

```
python3 -c "import ai4ts_fixes as fx; fx.run_all('data/BS_*.csv', 'data/POWER_*_LST.csv')"
```

This runs the clock check, builds the hourly data set, trains linear regression and XGBoost with and without target-hour irradiance at 1, 3, 6 and 24 hours, scores all baselines, runs the Diebold–Mariano tests, and writes `results/ai4ts_*.csv`.

**Neural models** (LSTM and MLP; about a minute per fit on a CPU):

```
python3 lstm_experiments.py select 'data/BS_*.csv'        # compare five designs on the 2021 validation year
python3 lstm_experiments.py summary
python3 lstm_experiments.py final 'data/BS_*.csv' lstm24  # chosen design, 3 seeds, 2022 test year
python3 lstm_experiments.py final 'data/BS_*.csv' mlp
```

Then combine the two final files into the input that `make_tables.py` reads:

```
python3 -c "import pandas as pd; pd.concat([pd.read_csv(f'results/lstm_final_{d}.csv') for d in ('lstm24', 'mlp')]).to_csv('results/ai4ts_lstm.csv', index=False)"
```

**Tables, figures and in-text numbers:**

```
python3 make_tables.py
```

This writes `tables/*.tex` and `figures/*.pdf` from `results/ai4ts_*.csv`.

**Data checks quoted in the text:**

```
python3 checks/validation_gap.py 'data/BS_*.csv'        # 2021 sensor gaps and the 2.8x validation gap
python3 checks/extra_analyses.py 'data/BS_*.csv'        # target-hour model without power features; DM vs. naive t-test
python3 checks/temperature_and_snow.py 'data/BS_*.csv' 'data/POWER_*_LST.csv' data/lex_daily.csv
```

## Files

| File | Purpose |
| --- | --- |
| `ai4ts_fixes.py` | Data loading, clock check, features, baselines (incl. CLIPER), models, scoring, DM test |
| `lstm_experiments.py` | LSTM, GRU and MLP designs, selection on 2021, final runs on 2022 |
| `make_tables.py` | Paper tables, figures and in-text number macros |
| `checks/validation_gap.py` | Effect of interpolating 2021 sensor gaps |
| `checks/extra_analyses.py` | Target-hour model without power features; naive t-test comparison |
| `checks/temperature_and_snow.py` | Air vs. module temperature; snow-day errors |
