# Libraries
import pandas as pd
import datetime
import numpy as np

# Constants

airports_colums_to_ignore = [
    "acReg",
    "mode_s",
    "status",
    "sqt",
    "stand_last_change",
    "stand_scheduled",
    "aldt_received",
    "stand_prepared",
    "stand_auto_start",
    "stand_active",
    "stand_docking",
    "aibt_received",
    "vdgs_in",
    "plb_on",
    "pca_on",
    "gpu_on",
    "towbar_on",
    "pca_off",
    "gpu_off",
    "acars_out",
    "vdgs_out",
    "stand_free",
    "roll",
    "speed",
    "last_distance_to_gate",
    "last_in_sector",
    "ship",
    "partition"
]

data_filepath = './data/'

acTypes_mapping = {
    'acType': {
        'MD88': 'MD-88',
        'CRJ/2': 'CRJ 100/200 Standard',  # Multiple
        'B757/2-WL': '757-200, -200PF with winglets',
        'A321/2': 'A321-200',
        'B717/2': '717-200',
        'B737/9-WL': '737-900 with winglets',
        'CRJ/9': 'CRJ 900 Standard',
        'MD90/3': 'MD-90-30',
        'A320/2': 'A320-200',
        'CRJ/7': 'CRJ 700/701/702 Standard',
        'B737/8-WL': '737-800 with winglets',
        'B737/7-WL': '737-700 with winglets',
        'A319': 'A319-100',
        'B767/4': '767-400ER',
        'B767/3-WL': '767-300',
        'B757/3-WL': '757-300',
        'A330/3': 'A330-300',
        'B777/2-LR': '777-200LR',
        'A330/2': 'A330-200',
        'CS100': 'A220-100',  # Airbus 220-100 aircraft
        'A350/9': 'A350-900',
        'B777/2': '777-200',
        'A319-SL': 'A319-100 Sharklet',
        'CS/100': 'A220-100',
        'MD90': 'MD-90-30',
        '7HD': '777-200ER',
        '73S': '737-200',
        'B757/3': '757-300',
        'B767-3': '767-300',
        'B737/9': '737-900',
        'A320': 'A320-200',
        'E175': 'EMB 175 Standard',
        'B737/8': '737-800'
    }
}

weather_columns_to_remove = [
    'PGTM',
    'SNOW',
    'SNWD'
]

weather_aggregation = {
    'AWND': ['mean', 'min', 'max'],
    'PRCP': ['sum'],
    'TAVG': 'mean',
    'TMIN': 'min',
    'TMAX': 'max',
    'WDF2': ['mean', 'min', 'max'],
    'WDF5': ['mean', 'min', 'max'],
    'WSF2': ['mean', 'min', 'max'],
    'WSF5': ['mean', 'min', 'max'],
    'WT01': 'max',
    'WT02': 'max',
    'WT03': 'max',
    'WT08': 'max'
}

weather_categorical_columns = [
    'WT01',
    'WT02',
    'WT03',
    'WT08'
]

col_with_wrong_type = [
    'Wingspan, ft',
    'Length, ft',
    'Wheelbase, ft',
    'Cockpit to Main Gear (CMG)',
    'MGW\n(Outer to Outer)',
    'Max Ramp\nMax Taxi'
]


# Core functions

def delete_to_ignore_columns_from(df, columns):
    return df.drop(columns, axis=1)


def clean_predicted_var(df):
    return df[(~pd.isnull(df.aldt)) & (~pd.isnull(df.aibt))]


def delete_na(df, column):
    return df[-df[column].isna()]


def delete_incorrect_data(df,column):
    return df[df[column].apply(len) != 4]


def convert_string_to_date(date_time_str, with_time=True):
    if with_time:
        conversion = datetime.datetime.strptime(date_time_str, '%m/%d/%Y %H:%M')
    else:
        conversion = datetime.datetime.strptime(date_time_str, '%m/%d/%Y')
    return conversion


def change_date_for_column(df, column, time):
    return df[column].apply(convert_string_to_date, with_time=time)


def clean_output(df, output):
    return df[-((df[output] <= datetime.timedelta(0)) | (df[output] > datetime.timedelta(0, 10800)))] #10800s = 3hours


def resample_ts(df, column, dict_agg):
    return df.groupby(column).agg(dict_agg)


def convert_na_to_0(df, column):
    return ~pd.isna(df[column]) * 1


def mean_target_encoding(df, grouping_columns, target_columns):

    for target_column in target_columns:
        mean_group = df.groupby(grouping_columns)[target_column].mean().reset_index()
        mean_group.columns = grouping_columns + ['_'.join(grouping_columns) + '_mean_' + target_column]
        df = df.merge(mean_group, on=grouping_columns)

    return df


def convert_object_to_float(df, columns):
    df[columns] = df[columns].applymap(lambda x: float(x))
    return df


def get_number_of_scheduled_flights_before(df, date):
    return df[
        (df.sto < date) &
        (df.sto >= date - datetime.timedelta(hours=1))
    ].shape[0]


def impute_xxx_aircrafts(df):

    xxx_flights = df[df.acType == 'XXX']

    for index, flight in xxx_flights.iterrows():
        df.at[index, 'acType'] = get_closest_aircraft_by_aldt(df, flight.carrier, flight.flight, flight.aldt)


def get_closest_aircraft_by_aldt(df, carrier, flight, date):
    flights = df[
        (df.flight == flight) &
        (df.carrier == carrier) &
        (df.acType != 'XXX')
    ]

    if flights.shape[0] == 0:
        return 'UNKNOWN'

    closest_date = min(flights.aldt, key=lambda x: abs(x - date))

    return flights[flights.aldt == closest_date].acType.values[0]


def get_moving_average_for_each_day(df):
    df['date'] = df.aibt.dt.date

    dates = df.date.unique()

    moving_averages = {}

    for d in dates:
        moving_averages[d] = df[(df.date < d) & (df.date >= df.date - datetime.timedelta(days=60))]['output_in_seconds'].mean()

    return moving_averages


# Final preprocess functions
def preprocess_airports_data(is_train=True):

    if is_train:

        airports_data = pd.read_csv(
            data_filepath + 'Airport_Data.csv',
            dtype=str
        )

    else :
        airports_data = pd.read_csv(
            data_filepath + 'Test_Set_Airport_Data.csv',
            dtype=str
        )

    airports_data = delete_to_ignore_columns_from(airports_data, airports_colums_to_ignore)
    airports_data = delete_na(airports_data, "aibt")
    airports_data = delete_na(airports_data, "aldt")
    airports_data = delete_incorrect_data(airports_data, "aibt")
    airports_data = delete_incorrect_data(airports_data, "aldt")

    airports_data.aibt = change_date_for_column(airports_data, "aibt", True)

    airports_data.aldt = change_date_for_column(airports_data, "aldt", True)
    airports_data.eibt = change_date_for_column(airports_data, "eibt", True)

    airports_data.sto = change_date_for_column(airports_data, "sto", True)
    airports_data['date'] = airports_data.sto.dt.date

    airports_data["output_in_seconds"] = airports_data.aibt - airports_data.aldt
    airports_data = clean_output(airports_data, "output_in_seconds")
    airports_data["output_in_seconds"] = airports_data.output_in_seconds.apply(datetime.timedelta.total_seconds).astype(float)

    airports_data["error_in_seconds"] = airports_data.eibt - airports_data.aibt
    airports_data['error_in_seconds'] = airports_data.error_in_seconds.apply(datetime.timedelta.total_seconds)

    airports_data = airports_data.replace(acTypes_mapping)

    airports_data['weekday'] = airports_data.aldt.apply(lambda x: x.weekday())
    airports_data['hour_minute'] = airports_data.aldt.apply(lambda x: x.hour + x.minute / 60)
    airports_data['quantile_hour'] = pd.qcut(airports_data.hour_minute, 5, labels=[0, 1, 2, 3, 4])
    airports_data['hour'] = airports_data.aldt.apply(lambda x: x.hour)

    nb_flight_weekday_hour = airports_data.groupby(['weekday', 'hour'])['flight'].count().reset_index()
    nb_flight_weekday_hour.columns = ['weekday', 'hour', 'flights_mean_count_per_hour']

    airports_data = airports_data.merge(
        nb_flight_weekday_hour,
        how='left',
        on=['weekday', 'hour']
    )

    airports_data.loc[:, 'flight'] = airports_data.loc[:, 'flight'].astype(float)

    impute_xxx_aircrafts(airports_data)

    return airports_data


def preprocess_weather_data(is_train=True):

    weather = pd.read_csv(data_filepath + 'weather_data_prep.csv')

    weather_clean = delete_to_ignore_columns_from(weather, weather_columns_to_remove)

    weather_clean.DATE = change_date_for_column(weather_clean, 'DATE', False).dt.date
    weather_clean = resample_ts(weather_clean, 'DATE', weather_aggregation)
    weather_clean[weather_categorical_columns] = convert_na_to_0(weather_clean, weather_categorical_columns)

    weather_clean.columns = weather_clean.columns.map('|'.join).str.strip('|')

    return weather_clean


def preprocess_aircrafts_data(is_train=True):
    aircrafts = pd.read_excel(data_filepath + 'ACchar.xlsx', sheet_name='test')
    aircrafts = aircrafts[aircrafts['Physical Class (Engine)'] != 'tbd']

    aircrafts = aircrafts.replace({'tbd': np.nan}).replace({
        'ICAO Code': {np.nan: 'CRJ9'},
        'ATCT Weight Class': {np.nan: 'Unknown'}
    })

    return aircrafts


def preprocess_final_data(is_train=True, cibt_based=False):
    airports = preprocess_airports_data(is_train=is_train)

    aircrafts = preprocess_aircrafts_data(is_train=is_train)

    weather = preprocess_weather_data(is_train=is_train)

    df = airports.merge(aircrafts, left_on='acType', right_on='Model', how='left')

    df = df.dropna(subset=['MTOW'])

    df.loc[:, '# Engines'] = df.loc[:, '# Engines'].apply(lambda x: 1 if x == x else 0)

    df = df.merge(weather, left_on='date', right_on='DATE')

    mean_target_encoding_columns = [
        ['carrier', 'flight'],
        ['stand']
    ]

    mean_target_output_columns = ['output_in_seconds']

    if is_train:
        for columns in mean_target_encoding_columns:
            df = mean_target_encoding(
                df,
                columns,
                mean_target_output_columns
            )

            output_columns = ['_'.join(columns) + '_mean_' + col for col in mean_target_output_columns]
            df.drop_duplicates(subset=columns)[columns + output_columns].to_csv(data_filepath + '_'.join(columns) + '.csv', index=None)
    else:
        for columns in mean_target_encoding_columns:
            file = pd.read_csv(data_filepath + '_'.join(columns) + '.csv')
            df = df.merge(file, on=columns)

    one_hot_encoding_columns = [
        'carrier',
        'runway',
        'Manufacturer',
        'AAC',
        'ADG',
        'TDG',
        'Wingtip Configuration',
        'Main Gear Config',
        'ICAO Code',
        'ATCT Weight Class',
        'weekday',
        'quantile_hour'
    ]

    df = pd.get_dummies(df, columns=one_hot_encoding_columns)

    to_drop_columns = [
        'plb_off',
        'eobt',
        'aobt',
        'atot',
        'chocks_on',
        'date_completed',
        'actype',
        'model',
        'physical_class_engine',
        'wake_category',
        'years_manufactured',
        'note',
        'hour',
        'flight',
        'stand',
        'aldt',
        'eibt',
        'date',
        'error_in_seconds',
        'sto'
    ]

    if not cibt_based:
        to_drop_columns.append('cibt')
        to_drop_columns.append('aibt')

    df = convert_object_to_float(df, col_with_wrong_type)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_'). \
        str.replace('(', '').str.replace(')', '').str.replace('|', '').str.replace(r"\n", "").str.replace(',', '')
    return df.drop(columns=to_drop_columns, axis=1)


# Execution flow

df_train = preprocess_final_data(is_train=True)
df_test = preprocess_final_data(is_train=False)

