import os
from datetime import date, datetime, timedelta
from time import sleep
from typing import List

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import utm
from ipywidgets import interact
from mpl_toolkits.axes_grid1 import make_axes_locatable
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_folder_if_not_exists(folder_path):
    """
    Check if a folder exists, and create it if it doesn't.

    Parameters
    ----------
    folder_path : str
        The path of the folder to check/create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def download_csv_from_url(url: str, destination_folder: str, filename: str):
    """
    Download a CSV file from a given URL and save it to a specified destination folder with the given filename.

    Parameters
    ----------
    url : str
        The URL of the CSV file to be downloaded.
    destination_folder : str
        The folder where the downloaded CSV file will be saved.
    filename : str
        The desired filename for the downloaded CSV file.

    Returns
    -------
    None
        The function does not return any value.

    Notes
    -----
    The function checks if the file already exists in the specified destination folder.
    If the file exists, it prints a message and exits without downloading the file again.
    If the file does not exist, it attempts to download the file from the provided URL.
    Successful downloads are saved to the specified destination folder with the given filename.

    Examples
    --------
    >>> download_csv_from_url('https://example.com/data.csv', '/path/to/save', 'my_data.csv')
    """
    full_path = os.path.join(destination_folder, filename)

    if os.path.exists(full_path):
        print(f"The CSV file already exists at: {full_path}")
        return  # The file already exists, no need to download it again

    try:
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        print(f"Downloading {filename}...")
        response = session.get(url)
        response.raise_for_status()
        content = response.content

        with open(full_path, "wb") as file:
            file.write(content)

        print(f"The CSV file has been successfully downloaded to: {full_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")


def get_coordinates_df() -> pd.DataFrame:
    """
    Retrieve GPS coordinates for stations from a CSV file and convert UTM coordinates to latitude and longitude.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing station information with added columns for latitude and longitude.
    """
    # Define the path to the data raw folder and the location data filename
    data_input_folder = "./data/input/"
    location_data_filename = "info_estaciones.csv"

    # Join the data domain and location data filename to get the full path
    location_data_path = os.path.join(data_input_folder, location_data_filename)
    station_data = pd.read_csv(location_data_path)

    # Rename columns for clarity
    station_data = station_data.rename(
        columns={"Nombre": "Station", "Huso Horario": "Zone Number", "Hemisferio": "Hemisphere"}
    )
    # Convert UTM coordinates to latitude and longitude using the convert_utm_to_lat_long function
    station_data[["Latitude", "Longitude"]] = station_data[
        ["UTM E", "UTM N", "Zone Number", "Hemisphere"]
    ].apply(lambda x: convert_utm_to_lat_long(*x), axis=1, result_type="expand")

    # Drop unnecessary columns
    station_data = station_data.drop(columns=["Id", "UTM E", "UTM N", "Zone Number", "Hemisphere"])

    return station_data


def convert_utm_to_lat_long(easting: int, northing: int, zone_number: int, hemisphere: str):
    """
    Convert UTM coordinates to latitude and longitude.

    Parameters
    ----------
    easting : int
        The easting (X-coordinate) value in UTM.
    northing : int
        The northing (Y-coordinate) value in UTM.
    zone_number : int
        The UTM zone number.
    hemisphere : str
        The hemisphere indicator, either 'N' for Northern Hemisphere or 'S' for Southern Hemisphere.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the latitude and longitude values in decimal degrees.

    Notes
    -----
    The function uses the utm library to perform the conversion. Ensure that the utm library
    is installed in your environment before using this function.

    Examples
    --------
    >>> convert_utm_to_lat_long(500000, 4649776, 33, 'N')
    (37.774929, -122.419416)
    """
    return utm.to_latlon(easting, northing, zone_number, northern=hemisphere.startswith("N"))


def get_last_two_sundays_ago(date_in: date):
    """
    Get the date of the Sunday two weeks before the given date.

    Parameters
    ----------
    date_in : date
        The reference date.

    Returns
    -------
    date
        The date corresponding to the Sunday two weeks before the input date.
    """
    back_in_time = date_in - timedelta(weeks=2)
    while back_in_time.weekday() != 6:  # 6 means Sunday (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
        back_in_time -= timedelta(days=1)

    return back_in_time


def get_url(
    region: str, station_id: str, pollutant: str, frequency: str, last_measurement: date, days: int
) -> str:
    """
    Generate a URL for downloading air quality data based on specified parameters.

    Parameters
    ----------
    region : str
        The region code.
    station_id : str
        The station code.
    pollutant : str
        The pollutant code.
    frequency : str
        The frequency of measurements (e.g., daily, hourly).
    last_measurement : date
        The date of the last measurement.
    days : int
        The number of days to go back from the last measurement date.

    Returns
    -------
    str
        The generated URL for downloading air quality data.
    """
    freq_measurements = frequency
    freq_average = freq_measurements
    date_to = last_measurement
    date_from = date_to - timedelta(days=days)

    date_to_YYMMDD = date_to.strftime("%y%m%d")
    date_from_YYMMDD = date_from.strftime("%y%m%d")
    url_format = (
        "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?"
        "outtype=xcl"
        f"&macro=./{region}/{station_id}/Cal/{pollutant}//{pollutant}.{freq_measurements}.{freq_average}.ic"
        f"&from={date_from_YYMMDD}"
        f"&to={date_to_YYMMDD}"
        "&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath="
    )
    return url_format


def get_filename(pollutant: str, region: str, station_id: str, desired_date: date):
    """
    Generate a filename for the downloaded air quality data file.

    Parameters
    ----------
    pollutant : str
        The pollutant code.
    region : str
        The region code.
    station_id : str
        The station code.
    desired_date : date
        The desired date for the data.

    Returns
    -------
    str
        The generated filename.
    """
    # Craft the name for the downloaded file...
    desired_date_str = desired_date.strftime("%y%m%d")
    return f"{pollutant}_{region}_{station_id}_{desired_date_str}.csv"


def get_pollutant_df(
    stations: dict, pollutant: str, timeframe_in_days: int = 365, use_cache: bool = True
) -> pd.DataFrame:
    """
    Download air quality data for multiple stations, combine it into a DataFrame,
    and add additional information.

    We explore each of the stations and notice that the download link for the 'csv'
    data follows a pattern.
    It specifies the station, the pollutant (PM10, PM2.5, etc.), and the date range.

    For example, the download link for the 'csv' file for the Pudahuel station is:
    "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&macro=./RM/D15/Cal/PM25//PM25.diario.diario.ic&from=000101&to=231022&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath="

    We note that `RM` is the region code, `D15` is the station code, `PM25` is the
    pollutant, `diario.diario` is the frequency type of the measurements to download,
    and the parameters `from` and `to` specify the desired date range.


    Parameters
    ----------
    stations : dict
        A dictionary mapping station names to their corresponding region and station ID.
    pollutant : str
        The pollutant code.
    timeframe_in_days : int, optional
        The number of days of historical data to download, default is 365.
    use_cache : bool, optional
        Flag to use cached files if already donwloaded, default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the downloaded air quality data with additional information.

    Notes
    -----
    This function iterates through the specified stations, downloads air quality data for the given pollutant,
    and combines the data into a single DataFrame. It also adds extra information such as station name, pollutant,
    and a datetime column derived from the date and time columns in the downloaded data.

    Examples
    --------
    >>> stations = {'Station1': ('RM', 'D15'), 'Station2': ('RM', 'D20')}
    >>> df = get_df(stations, 'PM10', timeframe_in_days=365, use_cache=True)
    """
    # destination folder for cache files...
    data_cache_folder = "./data/input/cache"
    # create if not exists
    create_folder_if_not_exists(data_cache_folder)

    dfs = []
    # # #
    # Download data;
    for station in stations:
        # Craft the url!
        region, station_id = stations[station]
        desired_date = get_last_two_sundays_ago(datetime.today())

        # Craft the name for the downloaded file...
        filename = get_filename(pollutant, region, station_id, desired_date)

        full_path = os.path.join(data_cache_folder, filename)
        should_download_files = not use_cache or not os.path.exists(full_path)
        if should_download_files:
            target_url = get_url(
                region, station_id, pollutant, "horario", desired_date, timeframe_in_days
            )
            download_csv_from_url(target_url, data_cache_folder, filename)
            # a little pause after downloading a file...
            sleep(1)

        # Load csv into a dataFrame, then add some extra info...
        single_df = pd.read_csv(
            full_path,
            sep=";",
        )
        single_df["Station"] = station
        single_df["Pollutant"] = pollutant

        # Combine the date and time columns and create a datetime column
        hh = pd.to_timedelta(single_df["HORA (HHMM)"] // 100, unit="H")
        single_df["DateTime"] = pd.to_datetime(single_df["FECHA (YYMMDD)"], format="%y%m%d") + hh

        dfs.append(single_df)

    # # #
    # Concat all the data
    df = pd.concat(dfs).reset_index(drop=True)
    # After crafting DateTime columne, we drop HORA and FECHA columns
    df = df.drop(columns=["HORA (HHMM)", "FECHA (YYMMDD)"])
    # Since the csv files has an ending semi-colon ';' we remove the last unnamed column `Unnamed: 5`
    df = df.drop(columns=["Unnamed: 5"])

    # Rename columns
    df = df.rename(
        columns={
            "Registros validados": "Validated Records",
            "Registros preliminares": "Preliminary Records",
            "Registros no validados": "Unvalidated Records",
        }
    )

    # Return the dataFrame with all the data
    return df


def create_station_na_heatmap(df: pd.DataFrame, sub_title: str = None):
    """Creates an interactive heatmap with showing na values

    Args:
        df (pd.DataFrame): The dataframe with the data.

    """

    def _interactive_na_heatmap(station):
        data = df.query("Station == @station").isnull()
        try:
            plt.figure(figsize=(12, 6))
            sns.heatmap(data, yticklabels=False, cbar=False, cmap="crest")
            plt.title(f"{sub_title} N/A Values (blue)\nStation: {station}")
        except ValueError:
            print("Heatmap cannot be shown for selected values...")

    # Widget for picking the station
    station_selection = widgets.Dropdown(options=df.Station.unique(), description="Station")

    # Putting it all together
    interact(_interactive_na_heatmap, station=station_selection)


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


def trim_unreported_data(df: pd.DataFrame, pollutants: list) -> pd.DataFrame:
    """
    Trims unreported data to simplify analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    pollutants : list
        List of pollutant columns to consider.

    Returns
    -------
    pd.DataFrame
        DataFrame with unreported data trimmed.

    Notes
    -----
    This function addresses the issue of reporting delays by trimming data that is not yet available for analysis.
    """

    # Drop rows where specified pollutants have NA values
    df_valid = df.dropna(subset=pollutants)

    # Find the latest datetime for each station
    latest_datetime_per_station = df_valid.groupby("Station")["DateTime"].max()

    # Find the earliest datetime among the latest datetimes for all stations
    latest_common_datetime = latest_datetime_per_station.min()

    # Return the subset of the original DataFrame up to the latest common datetime
    return df[df["DateTime"] <= latest_common_datetime]


def merge_gps_data(measurement_data: pd.DataFrame, station_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge air quality measurement and station data based on the 'Station' column.

    Parameters:
    - measurement_data (pd.DataFrame): DataFrame containing air quality measurement data.
    - station_data (pd.DataFrame): DataFrame containing station information.

    Returns:
    pd.DataFrame: Merged DataFrame with combined measurement and station data.

    This function performs an inner join on the 'Station' column, incorporating GPS coordinates
    from the station data into the measurement data. It also renames and drops unnecessary columns.

    Example:
    >>> merged_data = merge_gps_data(measurement_df, station_df)
    """
    # Merge DataFrames based on the 'Station' column using an inner join
    df = pd.merge(measurement_data, station_data, on="Station", how="inner")

    # Rename columns for clarity
    df = df.rename(
        columns={
            "Latitude_x": "Latitude",
            "Longitude_x": "Longitude",
        }
    )
    # Drop unnecessary columns related to pollutants
    df = df.drop(columns=["Pollutant_x", "Pollutant_y"])

    return df


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Forked code from Coursera - AI For good - utils.py
pollutants_list = ["PM2.5", "PM10"]

FONT_SIZE_TICKS = 12
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 16


def create_histogram_plot(df: pd.DataFrame, bins: int):
    """Creates an interactive histogram

    Args:
        df (pd.DataFrame): The dataframe with the data.
        bins (int): number of bins for the histogram.

    """

    def _interactive_histogram_plot(station, pollutant):
        data = df[df.Station == station]
        x = data[pollutant].values
        try:
            plt.figure(figsize=(12, 6))
            plt.xlabel(f"{pollutant} concentration", fontsize=FONT_SIZE_AXES)
            plt.ylabel("Number of measurements", fontsize=FONT_SIZE_AXES)
            plt.hist(x, bins=bins)
            plt.title(f"Pollutant: {pollutant} - Station: {station}", fontsize=FONT_SIZE_TITLE)
            plt.xticks(fontsize=FONT_SIZE_TICKS)
            plt.yticks(fontsize=FONT_SIZE_TICKS)
            plt.show()
        except ValueError:
            print("Histogram cannot be shown for selected values as there is no data")

    # Widget for picking the city
    station_selection = widgets.Dropdown(options=df.Station.unique(), description="Station")

    # Widget for picking the continuous variable
    pollutant_selection = widgets.Dropdown(
        options=pollutants_list,
        description="Pollutant",
    )

    # Putting it all together
    interact(_interactive_histogram_plot, station=station_selection, pollutant=pollutant_selection)


def create_boxplot(df: pd.DataFrame):
    """Creates a boxplot of pollutant values for each sensor station

    Args:
        df (pd.DataFrame): The dataframe with the data.

    """

    labels = df["Station"].unique()

    def _interactive_boxplot(cat_var):
        medians = []
        for value in df["Station"].unique():
            median = 1000
            try:
                rows = df[cat_var].loc[df["Station"] == value]
                if rows.isnull().sum() != rows.shape[0]:
                    median = rows.median()
            except Exception as e:
                print(f"Wrong: E:{e}")
            medians.append(median)
        orderInd = np.argsort(medians)

        plt.figure(figsize=(17, 7))
        scale = "linear"
        plt.yscale(scale)
        sns.boxplot(data=df, y=cat_var, x="Station", order=labels[orderInd], color="seagreen")
        plt.title(f"Distributions of {cat_var}", fontsize=FONT_SIZE_TITLE)
        plt.xlabel("Station", fontsize=FONT_SIZE_AXES)
        plt.ylabel(f"{cat_var} concentration", fontsize=FONT_SIZE_AXES)
        plt.xticks(fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()

    # Widget for picking the continuous variable
    cont_widget_histogram = widgets.Dropdown(
        options=pollutants_list,
        description="Pollutant",
    )

    interact(_interactive_boxplot, cat_var=cont_widget_histogram)


def create_scatterplot(df: pd.DataFrame):
    """Creates a scatterplot for pollutant values.
    The pollutants on the x and y axis can be chosen with a dropdown menu.

    Args:
        df (pd.DataFrame): The dataframe with the data.

    """
    df = df[pollutants_list]  # Take only the pollutants to scatter
    df_clean = df.dropna(inplace=False)

    def _interactive_scatterplot(var_x, var_y):
        x = df_clean[var_x].values
        y = df_clean[var_y].values
        bins = [200, 200]  # number of bins

        hh, locx, locy = np.histogram2d(x, y, bins=bins)
        z = np.array([hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])] for a, b in zip(x, y)])
        idx = z.argsort()
        x2, y2, z2 = x[idx], y[idx], z[idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        s = ax.scatter(x2, y2, c=z2, cmap="jet", marker=".", s=1)

        ax.set_xlabel(f"{var_x} concentration", fontsize=FONT_SIZE_AXES)
        ax.set_ylabel(f"{var_y} concentration", fontsize=FONT_SIZE_AXES)

        ax.set_title(
            f"{var_x} vs. {var_y} (color indicates density of points)", fontsize=FONT_SIZE_TITLE
        )
        ax.tick_params(labelsize=FONT_SIZE_TICKS)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(s, cax=cax, cmap="jet", values=z2, orientation="vertical")
        plt.show()

    cont_x_widget = widgets.Dropdown(options=pollutants_list, description="X-Axis")
    cont_y_widget = widgets.Dropdown(options=pollutants_list, description="Y-Axis", value="PM10")

    interact(_interactive_scatterplot, var_x=cont_x_widget, var_y=cont_y_widget)


def create_time_series_plot(df: pd.DataFrame, start_date: str, end_date: str):
    """Creates a time series plot, showing the concentration of pollutants over time.
    The pollutant and the measuring station can be selected with a dropdown menu.

    Args:
        df (pd.DataFrame): The dataframe with the data.
        start_date (str): minimum date for plotting.
        end_date (str): maximum date for plotting.

    """

    def _interactive_time_series_plot(station, pollutant, date_range):
        data = df[df.Station == station]
        data = data[data.DateTime > date_range[0]]
        data = data[data.DateTime < date_range[1]]
        plt.figure(figsize=(12, 6))
        plt.plot(data["DateTime"], data[pollutant], "-")
        plt.title(f"Temporal change of {pollutant}", fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f"{pollutant} concentration", fontsize=FONT_SIZE_AXES)
        plt.xticks(rotation=20, fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()

    # Widget for picking the station
    station_selection = widgets.Dropdown(options=df.Station.unique(), description="Station")

    # Widget for picking the pollutant
    pollutant_selection = widgets.Dropdown(
        options=pollutants_list,
        description="Pollutant",
    )

    dates = pd.date_range(start_date, end_date, freq="D")

    options = [(date.strftime(" %d/%m/%Y "), date) for date in dates]
    index = (0, len(options) - 1)

    # Slider for picking the dates
    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description="Dates",
        orientation="horizontal",
        layout={"width": "500px"},
    )

    # Putting it all together
    interact(
        _interactive_time_series_plot,
        station=station_selection,
        pollutant=pollutant_selection,
        date_range=selection_range_slider,
    )
