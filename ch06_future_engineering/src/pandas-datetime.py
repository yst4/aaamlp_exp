from tsfresh.feature_extraction.feature_calculators import mean_abs_change
from tsfresh.feature_extraction.feature_calculators import count_below_mean
from tsfresh.feature_extraction.feature_calculators import count_above_mean
from tsfresh.feature_extraction.feature_calculators import abs_energy
from keras.src.backend.openvino.numpy import ptp
from torch import var
from torch import std
from keras.src.backend.jax.numpy import mean
from matplotlib.dates import YearLocator
import pandas as pd
import numpy as np
from tsfresh.feature_extraction import feature_calculators as fc


s = pd.date_range('2020-01-06', '2020-01-10', freq='10h').to_series()

futures = {
    "dayofweek": s.dt.dayofweek.values,
    "dayofyear": s.dt.dayofyear.values,
    "hour": s.dt.hour.values,
    "is_leap_year": s.dt.is_leap_year.values,
    "quarter": s.dt.quarter.values,
    "weekofyear": s.dt.isocalendar().week.values, # orgin code (for old pandas): s.dt.weekofyear.values
}

def generate_features(df):
    dt = df['date'].dt
    features = {
        'year': dt.year,
        'month': dt.month,
        'dayofweek': dt.dayofweek,
        'weekofyear': dt.isocalendar().week,
        'weekend': dt.dayofweek >= 5,
    }
    return df.assign(**features)


future_dict = {}

future_dict['mean'] = np.mean()
future_dict['max'] = np.max()
future_dict['min'] = np.min()
future_dict['std'] = np.std()
future_dict['var'] = np.var()
future_dict['ptp'] = np.ptp()

future_dict['10'] = np.percentile(x, 10)
future_dict['60'] = np.percentile(x, 60)
future_dict['90'] = np.percentile(x, 90)

future_dict['5'] = np.quantile(x, 5)
future_dict['95'] = np.quantile(x, 95)
future_dict['99'] = np.quantile(x, 99)

stats = ['mean', 'max', 'min', 'std', 'var', 'ptp']

# np というモジュールから文字列と同名の関数を取り出して実行する
future_dict2 = {name: getattr(np, name)(x) for name in stats}

future_dict
funcs = [np.mean, np.max, np.min, np.std, np.var, np.ptp]
future_dict3 = {f.__name__: f(x) for f in funcs}
future_dict3


future_dict['abs_energy'] = fc.abs_energy(x)
future_dict['count_above_mean'] = fc.count_above_mean(x)
future_dict['count_below_mean'] = fc.count_below_mean(x)
future_dict['mean_abs_change'] = fc.mean_abs_change(x)
future_dict['mean_change'] = np.mean(np.diff(x))
