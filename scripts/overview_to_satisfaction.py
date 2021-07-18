from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, '../scripts/')

from data_cleaner import DataCleaner
from data_information import DataInfo
from results_pickler import ResultPickler
from data_manipulation import DataManipulator
from data_loader import load_df_from_csv
from data_loader import load_df_from_csv, load_df_from_excel

results = ResultPickler()


fields_description = load_df_from_csv("./data/fields_description.csv")


telData = load_df_from_excel(
    "./data/teleCo_data.xlsx", na_values=['undefined'])

hand_set = telData['Handset Type'].value_counts()[:10]
# Saving the data
results.add_data('top_10_handsets', hand_set)


# Findig top 3 based on unique value counts
hand_set_manuf = telData['Handset Manufacturer'].value_counts()[:3]
# Saving the data
results.add_data('top_3_handset_manufacturers', hand_set_manuf)


# #### Top 5 Handset per top 3 Handset Manufacturer


# Top 5 Apple Handsets
# Get apple manufacture handsets only
apple_df = telData[telData['Handset Manufacturer'] == 'Apple']
apple_df = apple_df.loc[:, ['Handset Type']]
best_apple = apple_df.value_counts()[:5]

# Saving the data
results.add_data('best_5_apple_handsets', best_apple)


# Top 5 Samsung Handsets
# Get samsung manufacture handsets only
samsung_df = telData[telData['Handset Manufacturer'] == 'Samsung']
samsung_df = samsung_df.loc[:, ['Handset Type']]
best_samsung = samsung_df.value_counts()[:5]

# Saving the data
results.add_data('best_5_samsung_handsets', best_samsung)


# Top 5 Huawei Handsets
# Get huawei manufacture handsets only
huawei_df = telData[telData['Handset Manufacturer'] == 'Huawei']
huawei_df = huawei_df.loc[:, ['Handset Type']]
best_huawei = huawei_df.value_counts()[:5]

# Saving the data
results.add_data('best_5_huawei_handsets', best_huawei)


# Top of all
# Grouping Handset type and Hanset Manufaturer colums and sorting them based on their size
new_df = telData.loc[:, ['Handset Type', 'Handset Manufacturer']]
value = new_df.groupby(['Handset Manufacturer', 'Handset Type']).size()
total_list = pd.Series(dtype='object')
for i in hand_set_manuf.index:
    total_list = total_list.append(value[i])
top_5 = total_list.sort_values(ascending=False)[:5]

# Saving the data
results.add_data('best_5_handsets', top_5)


# Instantiating the DataInfo Class to use it data extraction methods
explorer = DataInfo(telData)

aggs_by_col = {'Bearer Id': 'count',
               'Dur. (ms)': 'sum',
               'Total UL (Bytes)': 'sum',
               'Total DL (Bytes)': 'sum',
               'Social Media DL (Bytes)': 'sum',
               'Social Media UL (Bytes)': 'sum',
               'Google DL (Bytes)': 'sum',
               'Google UL (Bytes)': 'sum',
               'Email DL (Bytes)': 'sum',
               'Email UL (Bytes)': 'sum',
               'Youtube DL (Bytes)': 'sum',
               'Youtube UL (Bytes)': 'sum',
               'Netflix DL (Bytes)': 'sum',
               'Netflix UL (Bytes)': 'sum',
               'Gaming DL (Bytes)': 'sum',
               'Gaming UL (Bytes)': 'sum',
               'Other DL (Bytes)': 'sum',
               'Other UL (Bytes)': 'sum',
               'Gaming DL (Bytes)': 'sum',
               'Gaming UL (Bytes)': 'sum',
               }

ex1 = telData.groupby('MSISDN/Number').agg(aggs_by_col)

# ## Data Cleaning and Manipulation
# instantiating DataCleaner class to use it data cleaning methods
cleaner = DataCleaner(explorer.df)

# Fixing Columns
# changing datatypes
explorer.get_dataframe_columns_unique_value_count()

# removing single valued columns
single_value_cols = explorer.get_dataframe_columns_unique_value_count()
cleaner.remove_single_value_columns(single_value_cols)

# #### Standardizing Values
# changing bytes to megabytes
cleaner.convert_bytes_to_megabytes(['Total UL (Bytes)', 'Total DL (Bytes)', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                                    'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)'])

# #### Filtering Data
# method that eliminates duplicate values
cleaner.remove_duplicates()
# #### Fixing Missing Values
explorer = DataInfo(cleaner.df)


# merging columns to generate new values
# function for merging the columns, basic subtraction
def merge_add(col1: pd.Series, col2: pd.Series):
    return col1.add(col2)


cleaner.create_new_columns_from('TCP retransmission (Bytes)',
                                'TCP UL Retrans. Vol (Bytes)', 'TCP DL Retrans. Vol (Bytes)', merge_add)
tcp_transmition_col = cleaner.df.loc[:, ['TCP retransmission (Bytes)']]

# remove columns that have high missing value percentages than the provided value
cleaner.remove_unwanted_columns(
    explorer.get_columns_missing_percentage_greater_than(30.0))


# add 'TCP retransmission (Bytes)' again
cleaner.df['TCP retransmission (Bytes)'] = tcp_transmition_col

# remove the last row of the dataframe
cleaner.df.drop([cleaner.df.index[-1]], inplace=True)

# reinstantiate DataInfo from the cleaners dataframe
explorer = DataInfo(cleaner.df)

# Check if skew for numeric missing values and if not fill missing values with mean value but if skew use median

# calling method that fills numeric values with median or mean based on skeewness
cleaner.fill_numeric_values(explorer.get_columns_with_missing_values())

explorer = DataInfo(cleaner.df)


# using method to forward fill and bfill the remaining non-numeric values
cleaner.fill_non_numeric_values(
    explorer.get_columns_with_missing_values(), ffill=True, bfill=True)

# #### Adding Merged and Spliting Columns and Standardizing

# spliting date-time value to separate date and time columns for each start and end. (start_date, start_time, end_date, end_time)
cleaner.separate_date_time_column('Start', 'start')
cleaner.separate_date_time_column('End', 'end')

# merging columns to generate new values
# function for merging the columns, basic subtraction


def merge_sub(col1: pd.Series, col2: pd.Series):
    return col1.sub(col2)


def merge_add(col1: pd.Series, col2: pd.Series):
    return col1.add(col2)


# Total Data
cleaner.create_new_columns_from(
    'Total Data (MegaBytes)', 'Total UL (MegaBytes)', 'Total DL (MegaBytes)', merge_add)
# Total Social Media Data
cleaner.create_new_columns_from('Total Social Media Data (MegaBytes)',
                                'Social Media UL (MegaBytes)', 'Social Media DL (MegaBytes)', merge_add)
# Total Google Data
cleaner.create_new_columns_from('Total Google Data (MegaBytes)',
                                'Google UL (MegaBytes)', 'Google DL (MegaBytes)', merge_add)
# Total Email Data
cleaner.create_new_columns_from(
    'Total Email Data (MegaBytes)', 'Email UL (MegaBytes)', 'Email DL (MegaBytes)', merge_add)
# Total Youtube Data
cleaner.create_new_columns_from('Total Youtube Data (MegaBytes)',
                                'Youtube UL (MegaBytes)', 'Youtube DL (MegaBytes)', merge_add)
# Total Netflix Data
cleaner.create_new_columns_from('Total Netflix Data (MegaBytes)',
                                'Netflix UL (MegaBytes)', 'Netflix DL (MegaBytes)', merge_add)
# Total Gaming Data
cleaner.create_new_columns_from('Total Gaming Data (MegaBytes)',
                                'Gaming UL (MegaBytes)', 'Gaming DL (MegaBytes)', merge_add)
# Total Other Data
cleaner.create_new_columns_from(
    'Total Other Data (MegaBytes)', 'Other UL (MegaBytes)', 'Other DL (MegaBytes)', merge_add)
# Jitter
cleaner.create_new_columns_from(
    'Jitter (ms)', 'Activity Duration UL (ms)', 'Activity Duration DL (ms)', merge_sub)
# Avg Delay
cleaner.create_new_columns_from(
    'Avg Delay (ms)', 'Avg RTT UL (ms)', 'Avg RTT DL (ms)', merge_add)
# Avg Throughput
cleaner.create_new_columns_from(
    'Avg Throughput (kbps)', 'Avg Bearer TP UL (kbps)', 'Avg Bearer TP DL (kbps)', merge_add)

# Changing the specified columns to category type using cleaners function
cleaner.change_columns_type_to(['IMSI', 'Handset Manufacturer',
                                'Handset Type', 'IMEI', 'MSISDN/Number', 'Bearer Id'], 'category')


# convert to hour
def ms_to_hr(col: pd.Series):
    convertion_value = 1 / (1000 * 60 * 60)
    return col * convertion_value


cleaner.standardized_column(['Dur. (ms)', 'Dur. (ms).1'], [
                            'Dur. (hr)', 'Total Duration (hr)'], ms_to_hr)

# convert to second


def ms_to_sec(col: pd.Series):
    convertion_value = 1 / 1000
    return col * convertion_value


cleaner.standardized_column(['Start ms', 'End ms', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)'], [
                            'Start sec', 'End sec', 'Activity Duration DL (sec)', 'Activity Duration UL (sec)'], ms_to_sec)

# #### Save Generated Clean Data
# optimize dataframe
cleaner.optimize_df()

# save the final cleaned data
cleaner.save_clean_data('/teleCo_clean_data.csv')

# saving our results to a pickle file
results.save_data('./data/overview_results.pickle')

############################# End of Overview ############################
############################# start of Engagement #########################


results = ResultPickler()


# Fetching and selecting the specified metric columns only
clean_data = load_df_from_csv("./data/teleCo_clean_data.csv")
cleaner = DataCleaner(clean_data)
cleaner.remove_unwanted_columns(cleaner.df.columns[0])
clean_df = cleaner.df

df = clean_df.copy(deep=True)
aggs_by_col = {'Bearer Id': 'count',
               'Total Duration (hr)': 'sum',
               'Total Data (MegaBytes)': 'sum'
               }

df = df.groupby('MSISDN/Number').agg(aggs_by_col)

dataManipulator = DataManipulator(df)
top_10_session = dataManipulator.get_top_sorted_by_column(
    'Bearer Id', length=10)
# Saving the data
results.add_data('top_10_session', top_10_session)

# #### Based on Duration of Session (Total Duration (hr))


top_10_duration = dataManipulator.get_top_sorted_by_column(
    'Total Duration (hr)', length=10)
# Saving the data
results.add_data('top_10_duration', top_10_duration)


top_10_total_traffic = dataManipulator.get_top_sorted_by_column(
    'Total Data (MegaBytes)', length=10)
# Saving the data
results.add_data('top_10_total_traffic', top_10_total_traffic)

standard_df = df.copy(deep=True)
dataManipulator = DataManipulator(standard_df)
std_df = dataManipulator.standardize_column('Bearer Id')
std_df = dataManipulator.standardize_column('Total Duration (hr)')
std_df = dataManipulator.standardize_column('Total Data (MegaBytes)')

standardized_df = std_df.copy(deep=True)


engage_df = clean_df.copy(deep=True)
aggs_by_col = {'Total Social Media Data (MegaBytes)': 'sum',
               'Total Google Data (MegaBytes)': 'sum',
               'Total Email Data (MegaBytes)': 'sum',
               'Total Youtube Data (MegaBytes)': 'sum',
               'Total Netflix Data (MegaBytes)': 'sum',
               'Total Gaming Data (MegaBytes)': 'sum',
               'Total Other Data (MegaBytes)': 'sum', }

engage_df = engage_df.groupby('MSISDN/Number').agg(aggs_by_col)

dataManipulator = DataManipulator(engage_df)
top_10_socialmedia_users = dataManipulator.get_top_sorted_by_column(
    'Total Social Media Data (MegaBytes)', length=10)
# Saving the data
results.add_data('top_10_socialmedia_users', top_10_socialmedia_users)


top_10_google_users = dataManipulator.get_top_sorted_by_column(
    'Total Google Data (MegaBytes)', length=10)
# Saving the data
results.add_data('top_10_google_users', top_10_google_users)


top_10_email_users = dataManipulator.get_top_sorted_by_column(
    'Total Email Data (MegaBytes)', length=10)
# Saving the data
results.add_data('top_10_email_users', top_10_email_users)


top_10_youtube_users = dataManipulator.get_top_sorted_by_column(
    'Total Youtube Data (MegaBytes)', length=10)
# Saving the data
results.add_data('top_10_youtube_users', top_10_youtube_users)


top_10_netflix_users = dataManipulator.get_top_sorted_by_column(
    'Total Netflix Data (MegaBytes)', length=10)
# Saving the data
results.add_data('top_10_netflix_users', top_10_netflix_users)


top_10_gaming_users = dataManipulator.get_top_sorted_by_column(
    'Total Gaming Data (MegaBytes)', length=10)
# Saving the data
results.add_data('top_10_gaming_users', top_10_gaming_users)

top_10_other_users = dataManipulator.get_top_sorted_by_column(
    'Total Other Data (MegaBytes)', length=10)
# Saving the data
results.add_data('top_10_other_users', top_10_other_users)

most_used_df = clean_df.copy(deep=True)
most_used_df = most_used_df.loc[:, ['Total Google Data (MegaBytes)', 'Total Email Data (MegaBytes)', 'Total Youtube Data (MegaBytes)',
                                    'Total Netflix Data (MegaBytes)', 'Total Gaming Data (MegaBytes)', 'Total Other Data (MegaBytes)']]


total_app_data_usage = pd.DataFrame(
    most_used_df.sum(), columns=['Total Data (MegaBytes)'])
most_used_apps = total_app_data_usage.sort_values(
    by='Total Data (MegaBytes)', ascending=False).iloc[:4, :]

# Saving the data
results.add_data('most_used_apps', most_used_apps)


kmeans_kwargs = {"init": "random", "n_init": 10,
                 "max_iter": 300, "random_state": 42}
# A list holds the SSE values for each k
sse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(standardized_df)
    sse.append(kmeans.inertia_)

kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")


kmeans = KMeans(init="random", n_clusters=kl.elbow,
                n_init=10, max_iter=300, random_state=42)
label = kmeans.fit_predict(standardized_df)
centroids = kmeans.cluster_centers_
# print(f'# Centroids of the clustering:\n{centroids}')
# print(f'# The number of iterations required to converge: {kmeans.inertia_}')
# print(f'# The number of iterations required to converge: {kmeans.n_iter_}')
# Getting index based on clusters
u_labels = np.unique(label)

# plotting the results:
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
plt.title(
    f'User K-Means Classification with {kl.elbow} Groups (Standardized Data)')
ax.set_xlabel('Number of Sessions')
ax.set_ylabel('Total Duration (hr)')
ax.set_zlabel('Total Data (MegaBytes)')
for i in u_labels:
    ax.scatter(standardized_df[label == i].iloc[:, 0], standardized_df[label ==
                                                                       i].iloc[:, 1], standardized_df[label == i].iloc[:, 2], marker='o', label=i)
ax.scatter(centroids[:, 0], centroids[:, 1],
           centroids[:, 2], marker='x', color='black')
ax.legend()
fig.savefig('./data/engagement_cluster.png')


standardized_df['cluster'] = label
cleaner = DataCleaner(standardized_df)
cleaner.save_clean_data('./data/engagement_cluster.csv')


centroids_df = pd.DataFrame(centroids, columns=[
                            'Bearer Id', 'Total Duration (hr)', 'Total Data (MegaBytes)'])
cleaner = DataCleaner(centroids_df)
cleaner.save_clean_data('./data/engagement_centroid.csv')


# saving our results to a pickle file
results.save_data('./data/engagement_results.pickle')


####################################End of engagement######################
####################################Start of experience####################

results = ResultPickler()

# Fetching and selecting the specified metric columns only
clean_data = load_df_from_csv("./data/teleCo_clean_data.csv")
cleaner = DataCleaner(clean_data)
cleaner.remove_unwanted_columns(cleaner.df.columns[0])
clean_df = cleaner.df

# Apply outlier fixing method if they have outlier values
working_df = cleaner.fix_outlier_columns(
    ['TCP retransmission (Bytes)', 'Avg Delay (ms)', 'Avg Throughput (kbps)'])
working_df.loc[:, 'Handset Type'].isnull().sum().sum()

# aggregating based on MSISDN/Number with the specified metrics and create working Dataframe
aggs_by_col = {'TCP retransmission (Bytes)': 'sum',
               'Avg Delay (ms)': 'sum',
               'Avg Throughput (kbps)': 'sum'
               }

working_df = working_df.groupby('MSISDN/Number').agg(aggs_by_col)
rn_df = clean_df.loc[:, ['MSISDN/Number', 'Handset Type']]
rn_df = rn_df.groupby('MSISDN/Number')
working_df['Handset Type'] = rn_df.first(
).loc[working_df.index.to_list(), ['Handset Type']]

explorer = DataInfo(working_df)
most_occuring_values = explorer.get_mode()
# Saving the data
results.add_data('most_occuring_values', most_occuring_values)

min_max_10_TCP = explorer.get_min_max_of_column(
    'TCP retransmission (Bytes)', 10)
# Saving the data
results.add_data('min_max_10_TCP', min_max_10_TCP)

min_max_10_RTT = explorer.get_min_max_of_column('Avg Delay (ms)', 10)
# Saving the data
results.add_data('min_max_10_RTT', min_max_10_RTT)

min_max_10_Throughput = explorer.get_min_max_of_column(
    'Avg Throughput (kbps)', 10)
# Saving the data
results.add_data('min_max_10_Throughput', min_max_10_Throughput)


# aggregating based on Handset Type with the Avg Throughput (kbps) metric
aggs_by_col = {'Avg Throughput (kbps)': 'sum'}

handset_thr_group_df = working_df.groupby('Handset Type').agg(aggs_by_col)

top_10_handsets_Throughput = handset_thr_group_df.sort_values(
    'Avg Throughput (kbps)', ascending=False).iloc[:10, :]
# Saving the data
results.add_data('top_10_handsets_Throughput', top_10_handsets_Throughput)


least_10_handsets_Throughput = handset_thr_group_df.sort_values(
    'Avg Throughput (kbps)', ascending=False).iloc[-10:, :]
# Saving the data
results.add_data('least_10_handsets_Throughput', least_10_handsets_Throughput)

# aggregating based on Handset Type with the Avg Throughput (kbps) metric
aggs_by_col = {'TCP retransmission (Bytes)': 'sum'}

handset_tcp_group_df = working_df.groupby('Handset Type').agg(aggs_by_col)

top_10_handsets_TCP = handset_tcp_group_df.sort_values(
    'TCP retransmission (Bytes)', ascending=False).iloc[:10, :]
# Saving the data
results.add_data('top_10_handsets_TCP', top_10_handsets_TCP)

least_10_handsets_TCP = handset_tcp_group_df.sort_values(
    'TCP retransmission (Bytes)', ascending=True).iloc[:10, :]
# Saving the data
results.add_data('least_10_handsets_TCP', least_10_handsets_TCP)

# Standardize values before clustering to make it easy for the algorithm

std_df = working_df.copy(deep=True)
dataManipulator = DataManipulator(std_df)
std_df = dataManipulator.standardize_column('Avg Throughput (kbps)')
std_df = dataManipulator.standardize_column('Avg Delay (ms)')
std_df = dataManipulator.standardize_column('TCP retransmission (Bytes)')
std_df = std_df.iloc[:, :-1]

# #### Finding the optimimum k-value
kmeans_kwargs = {"init": "random", "n_init": 10,
                 "max_iter": 300, "random_state": 42}
# A list holds the SSE values for each k
sse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(std_df)
    sse.append(kmeans.inertia_)

kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")

km = KMeans(init="random", n_clusters=kl.elbow, n_init=10)
label = km.fit_predict(std_df)
centroids = km.cluster_centers_
# print(f'# Centroids of the clustering:\n{centroids}')
# print(f'# The number of iterations required to converge: {km.inertia_}')
# print(f'# The number of iterations required to converge: {km.n_iter_}')

# Getting index based on clusters
u_labels = np.unique(label)

# plotting the results:
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
plt.title(
    f'User K-Means Classification with {kl.elbow} Groups (Standardized Data)')
ax.set_xlabel('TCP retransmission (Bytes)')
ax.set_ylabel('Avg Delay (ms)')
ax.set_zlabel('Avg Throughput (kbps)')
for i in u_labels:
    ax.scatter(std_df[label == i].iloc[:, 0], std_df[label == i].iloc[:,
                                                                      1], std_df[label == i].iloc[:, 2], marker='o', label=i)
ax.scatter(centroids[:, 0], centroids[:, 1],
           centroids[:, 2], marker='x', color='black')
ax.legend()
fig.savefig('./data/experience_cluster.png')

std_df['cluster'] = label
cleaner = DataCleaner(std_df)
cleaner.save_clean_data('./data/experience_cluster.csv')

centroids_df = pd.DataFrame(centroids, columns=[
                            'TCP retransmission (Bytes)', 'Avg Delay (ms)', 'Avg Throughput (kbps)'])
cleaner = DataCleaner(centroids_df)
cleaner.save_clean_data('./data/experience_centroid.csv')

results.save_data('./data/experience_results.pickle')

#############################End of experience#################
#############################Start of satisfaction##############

results = ResultPickler()

# Fetching and selecting the specified metric columns only
clean_data = load_df_from_csv("./data/teleCo_clean_data.csv")
cleaner = DataCleaner(clean_data)
cleaner.remove_unwanted_columns(cleaner.df.columns[0])
clean_df = cleaner.df

# Apply outlier fixing method if they have outlier values
working_df = cleaner.fix_outlier_columns(
    ['TCP retransmission (Bytes)', 'Avg Delay (ms)', 'Avg Throughput (kbps)'])
working_df.loc[:, 'Handset Type'].isnull().sum().sum()

# aggregating based on MSISDN/Number with the specified metrics and create working Dataframe
aggs_by_col = {'TCP retransmission (Bytes)': 'sum',
               'Avg Delay (ms)': 'sum',
               'Avg Throughput (kbps)': 'sum'
               }

working_df = working_df.groupby('MSISDN/Number').agg(aggs_by_col)
rn_df = clean_df.loc[:, ['MSISDN/Number', 'Handset Type']]
rn_df = rn_df.groupby('MSISDN/Number')
working_df['Handset Type'] = rn_df.first(
).loc[working_df.index.to_list(), ['Handset Type']]

explorer = DataInfo(working_df)
most_occuring_values = explorer.get_mode()
# Saving the data
results.add_data('most_occuring_values', most_occuring_values)

min_max_10_TCP = explorer.get_min_max_of_column(
    'TCP retransmission (Bytes)', 10)
# Saving the data
results.add_data('min_max_10_TCP', min_max_10_TCP)

min_max_10_RTT = explorer.get_min_max_of_column('Avg Delay (ms)', 10)
# Saving the data
results.add_data('min_max_10_RTT', min_max_10_RTT)


min_max_10_Throughput = explorer.get_min_max_of_column(
    'Avg Throughput (kbps)', 10)
# Saving the data
results.add_data('min_max_10_Throughput', min_max_10_Throughput)

# aggregating based on Handset Type with the Avg Throughput (kbps) metric
aggs_by_col = {'Avg Throughput (kbps)': 'sum'}

handset_thr_group_df = working_df.groupby('Handset Type').agg(aggs_by_col)

top_10_handsets_Throughput = handset_thr_group_df.sort_values(
    'Avg Throughput (kbps)', ascending=False).iloc[:10, :]
# Saving the data
results.add_data('top_10_handsets_Throughput', top_10_handsets_Throughput)

least_10_handsets_Throughput = handset_thr_group_df.sort_values(
    'Avg Throughput (kbps)', ascending=False).iloc[-10:, :]
# Saving the data
results.add_data('least_10_handsets_Throughput', least_10_handsets_Throughput)

# aggregating based on Handset Type with the Avg Throughput (kbps) metric
aggs_by_col = {'TCP retransmission (Bytes)': 'sum'}

handset_tcp_group_df = working_df.groupby('Handset Type').agg(aggs_by_col)

top_10_handsets_TCP = handset_tcp_group_df.sort_values(
    'TCP retransmission (Bytes)', ascending=False).iloc[:10, :]
# Saving the data
results.add_data('top_10_handsets_TCP', top_10_handsets_TCP)

least_10_handsets_TCP = handset_tcp_group_df.sort_values(
    'TCP retransmission (Bytes)', ascending=True).iloc[:10, :]
# Saving the data
results.add_data('least_10_handsets_TCP', least_10_handsets_TCP)

# Standardize values before clustering to make it easy for the algorithm

std_df = working_df.copy(deep=True)
dataManipulator = DataManipulator(std_df)
std_df = dataManipulator.standardize_column('Avg Throughput (kbps)')
std_df = dataManipulator.standardize_column('Avg Delay (ms)')
std_df = dataManipulator.standardize_column('TCP retransmission (Bytes)')
std_df = std_df.iloc[:, :-1]

kmeans_kwargs = {"init": "random", "n_init": 10,
                 "max_iter": 300, "random_state": 42}
# A list holds the SSE values for each k
sse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(std_df)
    sse.append(kmeans.inertia_)

kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")

km = KMeans(init="random", n_clusters=kl.elbow, n_init=10)
label = km.fit_predict(std_df)
centroids = km.cluster_centers_
# print(f'# Centroids of the clustering:\n{centroids}')
# print(f'# The number of iterations required to converge: {km.inertia_}')
# print(f'# The number of iterations required to converge: {km.n_iter_}')

# Getting index based on clusters
u_labels = np.unique(label)

# plotting the results:
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
plt.title(
    f'User K-Means Classification with {kl.elbow} Groups (Standardized Data)')
ax.set_xlabel('TCP retransmission (Bytes)')
ax.set_ylabel('Avg Delay (ms)')
ax.set_zlabel('Avg Throughput (kbps)')
for i in u_labels:
    ax.scatter(std_df[label == i].iloc[:, 0], std_df[label == i].iloc[:,
                                                                      1], std_df[label == i].iloc[:, 2], marker='o', label=i)
ax.scatter(centroids[:, 0], centroids[:, 1],
           centroids[:, 2], marker='x', color='black')
ax.legend()
fig.savefig('./data/experience_cluster.png')

std_df['cluster'] = label
cleaner = DataCleaner(std_df)
cleaner.save_clean_data('./data/experience_cluster.csv')

centroids_df = pd.DataFrame(centroids, columns=[
                            'TCP retransmission (Bytes)', 'Avg Delay (ms)', 'Avg Throughput (kbps)'])
cleaner = DataCleaner(centroids_df)
cleaner.save_clean_data('./data/experience_centroid.csv')

results.save_data('./data/experience_results.pickle')
