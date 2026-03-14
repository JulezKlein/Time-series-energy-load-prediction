import meteostat as ms
from datetime import date
import pandas as pd
import numpy as np
import holidays
from pathlib import Path

from entsoe import EntsoePandasClient
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False


def _load_project_dotenv() -> None:
    project_root_env = Path(__file__).resolve().parents[1] / ".env"
    cwd_env = Path.cwd() / ".env"

    for env_path in (project_root_env, cwd_env):
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return

    load_dotenv(override=False)


_load_project_dotenv()


def get_load_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    country_code: str = "DE",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetches ENTSO-E load data for a country and time range.

    Parameters:
    start (pd.Timestamp): Start timestamp with timezone (e.g. Europe/Brussels).
    end (pd.Timestamp): End timestamp with timezone.
    country_code (str): ENTSO-E country code.
    api_key (str | None): ENTSO-E API key. If None, reads ENTSOE_API_KEY from env.

    Returns:
    pd.DataFrame: Daily load feature dataframe with lag and rolling columns.
    """
    resolved_api_key = api_key or os.getenv("ENTSOE_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "ENTSOE_API_KEY is not set. Provide api_key or set environment variable.")

    # Documentation: https://github.com/EnergieID/entsoe-py/tree/master
    client = EntsoePandasClient(api_key=resolved_api_key)
    load_data = client.query_load(country_code, start=start, end=end)

    if isinstance(load_data, pd.DataFrame):
        load_series = load_data.iloc[:, 0]
    else:
        load_series = load_data

    df_load = (
        load_series.resample("D")
        .mean()
        .to_frame(name="load")
    )

    # Add lag features and rolling mean for temporal patterns
    df_load["load_lag_1"] = df_load["load"].shift(1)  # Yesterday's load
    df_load["load_lag_7"] = df_load["load"].shift(6)  # Load from the target day last week
    df_load["load_lag_14"] = df_load["load"].shift(13)  # Load from target day two weeks ago
    df_load["rolling_mean_7"] = df_load["load"].rolling(7).mean()  # 7-day rolling mean
    df_load["rolling_mean_14"] = df_load["load"].rolling(14).mean()  # 14-day rolling mean
    df_load["std_7"] = df_load["load"].rolling(7).std()  # 7-day rolling std
    df_load["std_14"] = df_load["load"].rolling(14).std()  # 14-day rolling std

    # Add the target for the prediction task: load of the next day
    df_load["load_t+1"] = df_load["load"].shift(-1)
    df_load["load_t+2"] = df_load["load"].shift(-2)
    df_load["load_t+3"] = df_load["load"].shift(-3)
    df_load["load_t+4"] = df_load["load"].shift(-4)

    df_load = df_load.reset_index().rename(columns={"index": "time"})
    return df_load


def get_weather_and_calender_data(start_date: date, end_date: date, locations: int = 4) -> pd.DataFrame:
    """
    Fetches weather data for a given location and date range.

    Parameters:
    start_date (date): The start date for the weather data.
    end_date (date): The end date for the weather data.
    locations (int): The number of nearby weather stations to consider.

    Returns:
    pd.DataFrame: A DataFrame containing the weather data.
    """
    # Specify locations and time range
    CITIES = {
        "Frankfurt": ms.Point(50.1155, 8.6842, 113),
        "Berlin": ms.Point(52.5200, 13.4050, 34),
        "Munich": ms.Point(48.1351, 11.5820, 519),
        "Cologne": ms.Point(50.9375, 6.9603, 37),
        "Hamburg": ms.Point(53.5511, 9.9937, 8),
    }
    
    HEATING_THRESHOLD = 17  # °C, below which heating demand typically increases
    COOLING_THRESHOLD = 22  # °C, above which cooling demand typically increases

    # Fetch daily average, minimum and maximum temperature for each city
    city_weather = {}
    for city_name, point in list(CITIES.items())[:locations]:
        stations = ms.stations.nearby(point, limit=3)
        ts = ms.daily(stations, start_date, end_date)
        city_df = ms.interpolate(ts, point).fetch()
        city_weather[city_name] = city_df[[ms.Parameter.TEMP, ms.Parameter.TMIN, ms.Parameter.TMAX, ms.Parameter.WSPD, ms.Parameter.TSUN, ms.Parameter.CLDC]].rename(
            columns={
                ms.Parameter.TEMP: "Temp",
                ms.Parameter.TMIN: "Min Temp",
                ms.Parameter.TMAX: "Max Temp",
                ms.Parameter.WSPD: "Wind Speed",
                ms.Parameter.TSUN: "Sunshine Duration",
                ms.Parameter.CLDC: "Cloud Cover",
            }
        )

    # Build per-metric dataframes across all cities
    temp_df = pd.DataFrame({city: metrics["Temp"]
                           for city, metrics in city_weather.items()})
    tmin_df = pd.DataFrame(
        {city: metrics["Min Temp"] for city, metrics in city_weather.items()})
    tmax_df = pd.DataFrame(
        {city: metrics["Max Temp"] for city, metrics in city_weather.items()})
    wspd_df = pd.DataFrame(
        {city: metrics["Wind Speed"] for city, metrics in city_weather.items()})
    sshn_df = pd.DataFrame(
        {city: metrics["Sunshine Duration"] for city, metrics in city_weather.items()})
    cldc_df = pd.DataFrame(
        {city: metrics["Cloud Cover"] for city, metrics in city_weather.items()})

    weather_df = pd.DataFrame(
        {
            "Temp": temp_df.mean(axis=1),               # °C
            "Min Temp": tmin_df.mean(axis=1),           # °C
            "Max Temp": tmax_df.mean(axis=1),           # °C
            "Wind Speed": wspd_df.mean(axis=1),         # km/h
            "Sunshine Duration": sshn_df.mean(axis=1),  # minutes
            "Cloud Cover": cldc_df.mean(axis=1),
        }
    )

    # Temperature-based features for heating and cooling demand estimation. Based on the assumed thresholds, these features capture the degree to which a day is likely to require heating or cooling.
    weather_df["Cooling Degrees"] = (weather_df["Temp"] - COOLING_THRESHOLD).clip(lower=0)
    weather_df["Heating Degrees"] = (HEATING_THRESHOLD - weather_df["Temp"]).clip(lower=0)
    
    datetime_index = pd.DatetimeIndex(pd.to_datetime(weather_df.index))
    weather_df.index = datetime_index

    weather_df["dayofweek"] = datetime_index.dayofweek
    weather_df["month"] = datetime_index.month

    # Use Germany-wide public holidays. Following holidays docs, check
    # membership directly against the holiday object for date safety.
    holiday_years = range(start_date.year, end_date.year + 2)
    de_holidays = holidays.country_holidays("DE", years=holiday_years)
    weather_df["is_holiday"] = pd.Series(
        [int(day in de_holidays) for day in datetime_index.date],
        index=weather_df.index,
    )

    weather_df["dow_sin"] = np.sin(2 * np.pi * weather_df["dayofweek"] / 7)
    weather_df["dow_cos"] = np.cos(2 * np.pi * weather_df["dayofweek"] / 7)
    weather_df["month_sin"] = np.sin(2 * np.pi * weather_df["month"] / 12)
    weather_df["month_cos"] = np.cos(2 * np.pi * weather_df["month"] / 12)

    weather_df = weather_df.drop(columns=["dayofweek", "month"])

    return weather_df


def get_matched_weather_load_data(
    start_date: date,
    end_date: date,
    country_code: str = "DE",
    locations: int = 4,
    api_key: str | None = None,
    align_calendar_to_target_day: bool = True,
    production_data: bool = False
) -> pd.DataFrame:
    """
    Fetches weather and load data and matches both by daily timestamps.

    Parameters:
    start_date (date): Start date for the dataset.
    end_date (date): End date for the dataset.
    country_code (str): ENTSO-E country code.
    locations (int): Number of cities used for weather averaging.
    api_key (str | None): ENTSO-E API key.
    align_calendar_to_target_day (bool): If True, shift calendar features by
        one day so row t contains calendar context for target day t+1.
    dropna_subset (list[str] | None): If provided, drop rows only when one of
        these columns is missing. If None, drop rows with any missing value.

    Returns:
    pd.DataFrame: Combined weather and load dataframe matched on `time`.
    """
    # Weather data by meteostat API, averaged across multiple cities
    weather_df = get_weather_and_calender_data(
        start_date=start_date, end_date=end_date, locations=locations)
    weather_df = weather_df.reset_index().rename(columns={"index": "time"})
    weather_df["time"] = pd.DatetimeIndex(pd.to_datetime(weather_df["time"])).normalize()

    if align_calendar_to_target_day:
        calendar_feature_cols = [
            "is_holiday",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ]
        # Row t predicts load_tomorrow (t+1), so use tomorrow's calendar context.
        weather_df = weather_df.sort_values("time").reset_index(drop=True)
        weather_df[calendar_feature_cols] = weather_df[calendar_feature_cols].shift(-1)

    # Load data by ENTSO-E API, resampled to daily frequency
    load_start = pd.Timestamp(start_date, tz="Europe/Brussels")
    load_end = pd.Timestamp(
        end_date, tz="Europe/Brussels") + pd.Timedelta(days=1)
    load_df = get_load_data(
        start=load_start,
        end=load_end,
        country_code=country_code,
        api_key=api_key,
    )

    load_df["time"] = pd.to_datetime(load_df["time"])
    if isinstance(load_df["time"].dtype, pd.DatetimeTZDtype):
        load_df["time"] = load_df["time"].dt.tz_convert(
            "Europe/Brussels").dt.tz_localize(None)
    load_df["time"] = load_df["time"].dt.normalize()

    # Merge weather and load data on daily timestamps, keeping only matching days
    merged_df = pd.merge(weather_df, load_df, on="time", how="inner")
    merged_df = merged_df.sort_values("time").reset_index(drop=True)
    if production_data:
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.startswith("load_t")])  
    # Drop rows with missing values after merge (induced by lag features)
    merged_df = merged_df.dropna()
    return merged_df


if __name__ == "__main__":

    # Weather data fetching
    weather_df = get_weather_and_calender_data(
        date(2026, 2, 1), date(2026, 2, 28), locations=3)
    print(weather_df.head())

    # Load data fetching
    load_df = get_load_data(
        start=pd.Timestamp("2026-02-25", tz="Europe/Brussels"),
        end=pd.Timestamp("2026-02-28", tz="Europe/Brussels") +
        pd.Timedelta(days=1),
        country_code="DE",
        api_key=None,
    )
    print(load_df.head())

    # Merged weather and load data fetching
    merged_df = get_matched_weather_load_data(
        date(2025, 12, 1), date(2025, 12, 28),
        country_code="DE",
        locations=3,
        api_key=None,
    )
    print(merged_df.head())
