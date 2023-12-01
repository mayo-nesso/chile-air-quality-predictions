import math
import os
import warnings
from datetime import datetime
from typing import Dict

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import interact
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split


def create_na_heatmap_plot(df: pd.DataFrame):
    """Creates an interactive heatmap with showing na values

    Args:
        df (pd.DataFrame): The dataframe with the data.

    """

    def _interactive_na_heatmap(station):
        data = df.query("Station == @station").isnull()
        try:
            plt.figure(figsize=(12, 6))
            sns.heatmap(data, yticklabels=False, cbar=False, cmap="crest")
            plt.title(f"N/A Values\nStation: {station}", fontsize=FONT_SIZE_TITLE)
            plt.xticks(fontsize=FONT_SIZE_TICKS)
            plt.yticks(fontsize=FONT_SIZE_TICKS)
        except ValueError:
            print("Heatmap cannot be shown for selected values...")

    # Widget for picking the station
    station_selection = widgets.Dropdown(options=df.Station.unique(), description="Station")

    # Putting it all together
    interact(_interactive_na_heatmap, station=station_selection)


def get_mean_errors(y_test: np.array, y_pred: np.array) -> dict[str:float]:
    """
    Calculate various mean error metrics.

    Parameters
    ----------
    y_test : np.array
        True values.
    y_pred : np.array
        Predicted values.

    Returns
    -------
    dict
        Dictionary containing mean error metrics:
        - 'MAE': Mean Absolute Error
        - 'RMSE': Root Mean Squared Error
        - 'MAPE': Mean Absolute Percentage Error
    """
    MEEs = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
    }

    return MEEs


def calculate_mae_for_station_using_mean(df: pd.DataFrame, target: str) -> tuple:
    """
    Calculate Mean Absolute Error (MAE) for imputing missing values using the mean.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target column.
    target : str
        The target column for which missing values are to be imputed.

    Returns
    -------
    tuple
        A tuple containing:
        - model: Trained imputation model (SimpleImputer)
        - mean_errors: Dictionary with mean error metrics (MAE, RMSE, MAPE)
    """
    warnings.filterwarnings("ignore")
    df2 = df.dropna(inplace=False)
    train_df, test_df = train_test_split(df2, test_size=0.2, random_state=42)

    model = SimpleImputer(missing_values=np.nan, strategy="mean")
    model.fit(train_df[[target]])

    y_test = test_df[target].values

    test_df2 = test_df.copy()
    test_df2.loc[test_df.index, target] = float("NAN")

    y_pred = model.transform(test_df2[[target]])[:, 0]

    return model, get_mean_errors(y_test, y_pred)


def calculate_mae_for_station_using_mode(df: pd.DataFrame, target: str) -> tuple:
    """
    Calculate Mean Absolute Error (MAE) for imputing missing values using the mode.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target column.
    target : str
        The target column for which missing values are to be imputed.

    Returns
    -------
    tuple
        A tuple containing:
        - model: Trained imputation model (SimpleImputer)
        - mean_errors: Dictionary with mean error metrics (MAE, RMSE, MAPE)
    """
    warnings.filterwarnings("ignore")
    df2 = df.dropna(inplace=False)
    train_df, test_df = train_test_split(df2, test_size=0.2, random_state=42)

    model = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    model.fit(train_df[[target]])

    y_test = test_df[target].values

    test_df2 = test_df.copy()
    test_df2.loc[test_df.index, target] = float("NAN")

    y_pred = model.transform(test_df2[[target]])[:, 0]

    return model, get_mean_errors(y_test, y_pred)


def calculate_mae_for_station_using_lr(df: pd.DataFrame, independent: str, target: str) -> tuple:
    """
    Calculate Mean Absolute Error (MAE) using Linear Regression for imputing missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the independent and target columns.
    independent : str
        The name of the independent variable column.
    target : str
        The target column for which missing values are to be imputed.

    Returns
    -------
    tuple
        A tuple containing:
        - model: Trained linear regression model (LinearRegression)
        - mean_errors: Dictionary with mean error metrics (MAE, RMSE, MAPE)
    """
    warnings.filterwarnings("ignore")
    df2 = df.copy()
    df2[independent] = df2[independent].bfill()
    df2 = df.dropna(inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df2[[independent]], df2[[target]], test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, get_mean_errors(y_test, y_pred)


def print_mae_results(inputation_scores: dict):
    # Print out the MAE result
    print("{:<25} {:<10} {:<10} {:<10}".format("MODEL", "MAE", "RMSE", "MAPE"))
    for model_name, model_scores in inputation_scores.items():
        print(
            "{:<25} {:<10.2f} {:<10.2f} {:<10.2f}".format(
                model_name, model_scores["MAE"], model_scores["RMSE"], model_scores["MAPE"]
            )
        )


def save_interim_data(df: pd.DataFrame, filename: str):
    """
    Save a DataFrame to an interim data folder in Feather format.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    filename : str
        The name of the file (without extension) to be used for saving.

    Notes
    -----
    The data is saved in Feather format for efficiency and interoperability with other data analysis tools.

    Examples
    --------
    >>> save_interim_data(my_dataframe, "processed_data")
    """
    data_interim_folder = "./data/interim/"
    # Create the full destination path for saving the DataFrame in Feather format
    destination_path = os.path.join(data_interim_folder, filename + ".feather")
    # Save the DataFrame to the specified destination in Feather format
    df.to_feather(destination_path)


# def get_na_length_mean(df: pd.DataFrame, pollutant: str) -> float:
#     """
#     Calculate the mean duration of continuous NaN values for a specified pollutant.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame containing the pollutant column.
#     pollutant : str
#         The name of the pollutant column for which to calculate NaN durations.

#     Returns
#     -------
#     float
#         The mean duration of continuous NaN values for the specified pollutant.
#     """
#     # Identify NaN values in the pollutant column
#     na_values = df[pollutant].isna()

#     # Find continuous occurrences of NaN and calculate their duration
#     na_sequences = (na_values != na_values.shift()).cumsum()
#     na_durations = na_values.groupby(na_sequences).transform("size") * na_values

#     # Since `na_durations` is a sequence of durations, like
#     # [0, 4,4,4,4 , 0, 0 ,0 , 2,2, 0, 1, 0]
#     # we will "compress" the array to get only one digit per sequence.
#     compressed = compress_array(na_durations)

#     # Discard any 0 na_duration (i.e., uptime)
#     only_na_durations = compressed[compressed != 0]

#     # # Calculate the mean duration of continuous NaN occurrences
#     na_length_mean = only_na_durations.mean()
#     return na_length_mean


# def compress_array(arr: np.ndarray) -> np.ndarray:
#     """
#     Compresses an array by keeping only the first occurrence of each unique value.

#     Parameters:
#     - arr (numpy.ndarray): The input array to be compressed.

#     Returns:
#     - numpy.ndarray: The compressed array containing only the first occurrence of each unique value.

#     Example:
#     >>> original_array = np.array([1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 6])
#     >>> compress_array(original_array)
#     array([1, 2, 3, 4, 5, 6])
#     """
#     # Find where the array changes, we use values to avoid working with index
#     changes = np.concatenate(([True], arr[:-1].values != arr[1:].values))
#     result_array = arr[changes]
#     return result_array.reset_index(drop=True)


def get_na_percentage(df: pd.DataFrame, pollutant: str) -> float:
    """
    Calculate the percentage of missing values for a specified pollutant.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the pollutant column.
    pollutant : str
        The name of the pollutant column for which to calculate the missing value percentage.

    Returns
    -------
    float
        The percentage of missing values for the specified pollutant.
    """
    return df[pollutant].isna().sum() / len(df[pollutant])


def impute_target_missing_values_with_lineal_regression_model(
    df_with_missing: pd.DataFrame,
    model,
    indepentend_var: str,
    target: str,
) -> pd.DataFrame:
    """
    Impute missing values in the target column using a linear regression model.

    Parameters
    ----------
    df_with_missing : pd.DataFrame
        Input DataFrame containing missing values in the target column.
    model : object
        Trained linear regression model for imputation.
    independent_var : str
        The name of the independent variable column used in the regression model.
    target : str
        The name of the target column with missing values to be imputed.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values and a flag column indicating linear regression imputation.
    """
    # Create a flag for imputed data
    imputed_flag = df_with_missing[[target]]
    # Create a flag column for the pollutant
    imputed_flag[f"{target}_imputed_flag"] = np.where(
        imputed_flag[target].isnull(), "linear_reg", None
    )
    imputed_flag.drop(target, axis=1, inplace=True)

    # Copy the original DataFrame for imputation
    df_to_impute = df_with_missing.copy()

    # Predict missing values using the linear regression model
    predicted = model.predict(df_with_missing[[indepentend_var]])

    # Assign the imputed values back to the DataFrame
    df_to_impute[target + "_imp_lr"] = predicted
    df_to_impute[target] = df_with_missing[target].combine_first(df_to_impute[target + "_imp_lr"])

    # Drop the intermediate column used for imputation
    df_to_impute = df_to_impute.drop(columns=[target + "_imp_lr"])

    # Combine imputed values with the flag data
    imputed_values_with_flag = df_to_impute.join(imputed_flag)

    return imputed_values_with_flag


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Forked code from Coursera - AI For good - utils.py
FONT_SIZE_TICKS = 12
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 16
pollutants_list = ["PM2.5", "PM10"]


def plot_distribution_of_gaps(df: pd.DataFrame, target: str, distribution_length=200):
    """Plots the distribution of the gap sizes in the dataframe

    Args:
        df (pd.DataFrame): The dataframe
        target (str): The chosen pollutant for which it plots the distribution
    """

    def get_size_down_periods(df, target):
        """Get the size of the downtime periods for the sensor"""
        distribution = [0] * (distribution_length)
        x = []
        i = -1
        total_missing = 0
        count = 0
        for row in df[target].values:
            if math.isnan(row):
                total_missing += 1
                if i == 0:
                    count = 1
                    i = 1
                else:
                    count += 1
            else:
                try:
                    if count > 0:
                        distribution[count] += 1
                        x.append(count)
                except Exception as e:
                    print(
                        f"Exception [{e}]: index: {count}, when 'distribution_length' max index is: {distribution_length}.\n"
                        + f"This means that a NA gap of {count} was found, but we can't register it on the array that the code\n"
                        + "Try modifiyng the paraemeter 'distribution_length' to a higer value.\n"
                    )
                i = 0
                count = 0

        # On the first 'slot' we just put a zero.
        # Doesn't make much sense to talk about distribution of gaps of no downtime (ie; 0)
        distribution[0] = 0

        return distribution

    def get_last_non_zero_element(distribution: list[int]):
        last_non_zero_index = len(distribution)

        for i in range(len(distribution) - 1, -1, -1):
            if distribution[i] != 0:
                last_non_zero_index = i
                break
        return last_non_zero_index

    def print_stats(df, target, distribution):
        # Gap length stats:
        # Amount
        na_count = len(df[target])
        # Percenatege of NA
        na_perc = get_na_percentage(df, pollutant=target)

        gap_lengths = get_gaps(distribution)
        # Mean of Gap Lengths
        gap_length_mean = np.mean(gap_lengths)
        # Median of Gap Lengths
        gap_length_median = np.median(gap_lengths)
        # Mode of Gap Lengths
        gap_length_mode = get_mode(gap_lengths)
        # Max of Gap Lengths
        gap_length_mode_max = np.max(gap_lengths)

        print(
            "NA Sequence Length Stats:\n"
            + f"\tCount: {na_count},\n"
            + f"\tPercentage: {na_perc*100:.3f}%,\n"
            + f"\tMean: {gap_length_mean:.1f}\n"
            + f"\tMedian: {gap_length_median}\n"
            + f"\tMode: {gap_length_mode}\n"
            + f"\tMax: {gap_length_mode_max}\n"
        )

    def get_gaps(distribution: list[int]):
        gaps = []
        for index, frequency in enumerate(distribution):
            # Since we created the distribution where each index is
            # the duration of the NA gap, for simplicity we create another variable
            # reflecting that.
            gap_duration = index
            # Skip irrelevant data
            if gap_duration == 0 or frequency == 0:
                continue

            # We add the gap_duration as many times as it was registered (frequency)
            gap_size = [gap_duration]
            gaps.extend(gap_size * frequency)

        return gaps

    def get_mode(elements):
        vals, counts = np.unique(elements, return_counts=True)
        return vals[np.argmax(counts)]

    distribution = get_size_down_periods(df, target=target)

    plt.figure(figsize=(10, 6))
    plt.plot(distribution)
    plt.xlabel("Gap size (Hours)", fontsize=FONT_SIZE_AXES)
    plt.ylabel("Frequency", fontsize=FONT_SIZE_AXES)
    plt.title("Distribution of gaps in the data", fontsize=FONT_SIZE_TITLE)

    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)

    padding = 10
    custom_x_lim = get_last_non_zero_element(distribution) + padding
    plt.xlim(0, custom_x_lim)
    plt.grid()
    plt.show()

    # 2nd graph
    gaps_in_hours = [0] * len(distribution)
    for i in range(len(distribution)):
        gaps_in_hours[i] = distribution[i] * i

    plt.figure(figsize=(10, 6))
    plt.plot(gaps_in_hours)
    plt.xlabel("Gap size (Hours)", fontsize=FONT_SIZE_AXES)
    plt.ylabel("Size of downtime windows in hours", fontsize=FONT_SIZE_AXES)
    plt.title("Distribution of downtime in the data", fontsize=FONT_SIZE_TITLE)

    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.xlim(0, custom_x_lim)
    plt.grid()
    plt.show()

    print_stats(df, target, distribution)


def visualize_missing_values_estimation_pm25(df: pd.DataFrame, day: datetime):
    """Visualizes two ways of interpolating the data: nearest neighbor and last value
    and compares them to the real data

    Args:
        df (pd.DataFrame): The dataframe
        day (datetime): The chosen day to plot
    """
    warnings.filterwarnings("ignore")

    day = day.date()

    # Filter out the data for the day for the USM station
    rows_of_day = df.apply(lambda row: row["DateTime"].date() == day, axis=1)
    sample = df[rows_of_day]

    def draw(sample, station, missing_index, target):
        sample = sample.copy()
        sample.insert(
            0,
            "time_discriminator",
            (sample["DateTime"].dt.dayofyear * 100000 + sample["DateTime"].dt.hour * 100).values,
            True,
        )
        #### Actual values
        real = sample[sample["Station"] == station]
        example1 = real.copy()
        real = real.reset_index()
        example1 = example1.reset_index()
        example1.loc[missing_index, target] = float("NaN")

        missing = missing_index
        missing_before_after = [missing[0] - 1] + missing + [missing[-1] + 1]
        dates = set(list(example1.loc[missing_index, "DateTime"].astype(str)))

        plt.figure(figsize=(10, 5))
        plt.plot(
            missing_before_after,
            real.loc[missing_before_after][target],
            "r--o",
            label="actual values",
        )
        plt.plot(example1.index, example1[target], "-*")

        # Nan segment
        sample_copy = sample.copy()
        sample_copy = sample_copy.reset_index()
        to_nan = sample_copy.apply(
            lambda row: str(row["DateTime"]) in dates and row["Station"] == station, axis=1
        )
        sample_copy.loc[to_nan, target] = float("NaN")
        #### KNN
        imputer = KNNImputer(n_neighbors=1)
        imputer.fit(sample_copy[["time_discriminator", "Latitude", "Longitude", target]])
        example1[f"new{target}"] = imputer.transform(
            example1[["time_discriminator", "Latitude", "Longitude", target]]
        )[:, 3]
        plt.plot(
            missing_before_after,
            example1.loc[missing_before_after][f"new{target}"],
            "g--o",
            label="nearest neighbor",
        )

        #### Last Known value
        example1[f"ffill{target}"] = example1.fillna(method="ffill")[target]
        plt.plot(
            missing_before_after,
            example1.loc[missing_before_after][f"ffill{target}"],
            "y--*",
            label="last known value",
        )

        #####

        #### Lineal regression
        lr_model, _ = calculate_mae_for_station_using_lr(example1, "PM10", target)

        sample_array = np.array(sample_copy.loc[to_nan, "PM10"]).reshape(-1, 1)
        example1.loc[missing_before_after, f"lr_{target}"] = example1.loc[missing_before_after][
            target
        ]
        example1.loc[missing, f"lr_{target}"] = lr_model.predict(sample_array)
        plt.plot(
            missing_before_after,
            example1.loc[missing_before_after][f"lr_{target}"],
            "b--*",
            label="lineal regression",
        )
        ###

        plt.xlabel("Hour of day", fontsize=FONT_SIZE_AXES)
        plt.ylabel(f"{target} concentration", fontsize=FONT_SIZE_AXES)
        plt.title("Estimating missing values", fontsize=FONT_SIZE_TITLE)
        plt.legend(loc="upper left", fontsize=FONT_SIZE_TICKS)
        plt.xticks(fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()

    def selector(station, hour_start, window_size, target):
        missing_index_list = list(range(hour_start, hour_start + window_size))
        draw(sample=sample, station=station, missing_index=missing_index_list, target=target)

    # Widgets for selecting the parameters
    station_selection = widgets.Dropdown(
        options=df.Station.unique(), description="Station", value="Cerrillos II"
    )

    hour_start_selection = widgets.Dropdown(
        options=list([2, 3, 4, 5, 6, 7, 8, 9, 10]), description="Hour start", value=3
    )
    window_size_selection = widgets.Dropdown(
        options=list([1, 2, 3, 5, 6, 9, 12]), description="Window size", value=1
    )

    return interact(
        selector,
        station=station_selection,
        hour_start=hour_start_selection,
        window_size=window_size_selection,
        target="PM2.5",
    )


def visualize_missing_values_estimation_pm10(df: pd.DataFrame, day: datetime):
    """Visualizes two ways of interpolating the data: nearest neighbor and last value
    and compares them to the real data

    Args:
        df (pd.DataFrame): The dataframe
        day (datetime): The chosen day to plot
    """
    warnings.filterwarnings("ignore")

    day = day.date()

    # Filter out the data for the day for the USM station
    rows_of_day = df.apply(lambda row: row["DateTime"].date() == day, axis=1)
    sample = df[rows_of_day]

    def draw(sample, station, missing_index, target):
        sample = sample.copy()
        sample.insert(
            0,
            "time_discriminator",
            (sample["DateTime"].dt.dayofyear * 100000 + sample["DateTime"].dt.hour * 100).values,
            True,
        )
        #### Actual values
        real = sample[sample["Station"] == station]
        example1 = real.copy()
        real = real.reset_index()
        example1 = example1.reset_index()
        example1.loc[missing_index, target] = float("NaN")

        missing = missing_index
        missing_before_after = [missing[0] - 1] + missing + [missing[-1] + 1]
        dates = set(list(example1.loc[missing_index, "DateTime"].astype(str)))

        plt.figure(figsize=(10, 5))
        plt.plot(
            missing_before_after,
            real.loc[missing_before_after][target],
            "r--o",
            label="actual values",
        )
        plt.plot(example1.index, example1[target], "-*")

        # Nan segment
        sample_copy = sample.copy()
        sample_copy = sample_copy.reset_index()
        to_nan = sample_copy.apply(
            lambda row: str(row["DateTime"]) in dates and row["Station"] == station, axis=1
        )
        sample_copy.loc[to_nan, target] = float("NaN")
        #### KNN
        imputer = KNNImputer(n_neighbors=1)
        imputer.fit(sample_copy[["time_discriminator", "Latitude", "Longitude", target]])
        example1[f"new{target}"] = imputer.transform(
            example1[["time_discriminator", "Latitude", "Longitude", target]]
        )[:, 3]
        plt.plot(
            missing_before_after,
            example1.loc[missing_before_after][f"new{target}"],
            "g--o",
            label="nearest neighbor",
        )

        #### Last Known value
        example1[f"ffill{target}"] = example1.fillna(method="ffill")[target]
        plt.plot(
            missing_before_after,
            example1.loc[missing_before_after][f"ffill{target}"],
            "y--*",
            label="last known value",
        )

        #####

        plt.xlabel("Hour of day", fontsize=FONT_SIZE_AXES)
        plt.ylabel(f"{target} concentration", fontsize=FONT_SIZE_AXES)
        plt.title("Estimating missing values", fontsize=FONT_SIZE_TITLE)
        plt.legend(loc="upper left", fontsize=FONT_SIZE_TICKS)
        plt.xticks(fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()

    def selector(station, hour_start, window_size, target):
        missing_index_list = list(range(hour_start, hour_start + window_size))
        draw(sample=sample, station=station, missing_index=missing_index_list, target=target)

    # Widgets for selecting the parameters
    station_selection = widgets.Dropdown(
        options=df.Station.unique(), description="Station", value="Cerrillos II"
    )
    target_pollutant_selection = "PM10"

    hour_start_selection = widgets.Dropdown(
        options=list([2, 3, 4, 5, 6, 7, 8, 9, 10]), description="Hour start", value=3
    )
    window_size_selection = widgets.Dropdown(
        options=list([1, 2, 3, 5, 6, 9, 12]), description="Window size", value=1
    )

    return interact(
        selector,
        station=station_selection,
        hour_start=hour_start_selection,
        window_size=window_size_selection,
        target=target_pollutant_selection,
    )


def calculate_mae_for_nearest_station(
    df: pd.DataFrame, target: str, n_neighbors: int = 1
) -> (KNNImputer, Dict[str, float]):
    """Create a nearest neighbor model and run it on your test data

    Args:
        df (pd.DataFrame): The dataframe
        target (str): The chosen pollutant for which it plots the distribution
    Returns:
    --------
    model : KNNImputer model
    mean_errors : dict
        A dictionary containing the Mean Absolute Error (MAE), the Root Mean Squared Error (RMSE), and
        the Mean Absolute Percentage Error (MAPE) calculated between the original data and the interpolated
        values after introducing NaNs.
    """
    df2 = df.dropna(inplace=False)
    df2.insert(
        0,
        "time_discriminator",
        (df2["DateTime"].dt.dayofyear * 100000 + df2["DateTime"].dt.hour * 100).values,
        True,
    )

    train_df, test_df = train_test_split(df2, test_size=0.2, random_state=42)

    model = KNNImputer(n_neighbors=n_neighbors)
    model.fit(train_df[["time_discriminator", "Latitude", "Longitude", target]])

    y_test = test_df[target].values

    test_df2 = test_df.copy()
    test_df2.loc[test_df.index, target] = float("NAN")

    y_pred = model.transform(test_df2[["time_discriminator", "Latitude", "Longitude", target]])[
        :, 3
    ]

    return model, get_mean_errors(y_test, y_pred)


def impute_pollutant_missing_values_with_knn(
    df_with_missing: pd.DataFrame,
    pollutant: str,
    knn_model,
) -> pd.DataFrame:
    """
    Imputes missing values in pollutant variables using KNN imputation and flags the imputed data.

    Args:
        df_with_missing (pd.DataFrame): The dataframe with missing data.
        pollutant (str): Name of the column to be imputed.
        knn_model: Trained KNN imputer model.

    Returns:
        imputed_values_with_flag (pd.DataFrame): The dataframe with imputed values and flags.
    """
    # Create a flag for imputed data
    imputed_flag = df_with_missing[[pollutant]]

    # Disable warnings
    warnings.filterwarnings("ignore")

    # Create a flag column for the pollutant
    imputed_flag[f"{pollutant}_imputed_flag"] = np.where(
        imputed_flag[pollutant].isnull(), "knn", None
    )
    imputed_flag.drop(pollutant, axis=1, inplace=True)

    # Prepare data for imputation
    df_to_impute = df_with_missing.copy()
    df_to_impute["time_discriminator"] = (
        df_to_impute["DateTime"].dt.dayofyear * 100000 + df_to_impute["DateTime"].dt.hour * 100
    )

    # Impute missing values using the KNN model
    predicted = knn_model.transform(
        df_to_impute[["time_discriminator", "Latitude", "Longitude", pollutant]]
    )

    # Update the dataframe with predicted values
    df_to_impute[pollutant] = predicted[:, 3]

    # Remove temporary columns
    df_to_impute = df_to_impute.drop(columns=["time_discriminator"])

    # Combine imputed values with flag data
    imputed_values_with_flag = df_to_impute.join(imputed_flag)

    return imputed_values_with_flag

    ##


#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#


##
#
