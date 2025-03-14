import datetime as dt
import pandas as pd


def temparature(x: pd.Series) -> float:
    """Fix `temparature` features"""
    if x.mintemp <= x.temparature <= x.maxtemp:
        return x.temparature
    else:
        return sorted([x.mintemp, x.temparature, x.maxtemp])[1]


def season(s: pd.Series) -> int:
    """Create `season` feature"""
    if 60 <= s.day <= 151:
        return 2    # spring
    elif 152 <= s.day <= 243:
        return 3    # summer
    elif 244 <= s.day <= 334:
        return 4    # autumn
    else:
        return 1    # winter


def month(s: pd.Series) -> int:
    """Create `month` feature"""
    base = dt.date(1, 1, 1)
    return (base + dt.timedelta(days=s.day - 1)).month
