# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TeleCo Company: Data Analysis
# %% [markdown]
# ## Loading Data

# %%
import graph_utils
from data_manipulation import DataManipulator
from data_loader import load_df_from_csv
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from IPython import get_ipython
from data_cleaner import DataCleaner
from data_information import DataInfo
from results_pickler import ResultPickler
from data_loader import load_df_from_csv, load_df_from_excel
import pandas as pd
import sys
sys.path.insert(0, '../scripts/')


# %%


# %%


# %%
results = ResultPickler()


# %%
#Loading Descriptions Data
fields_description = load_df_from_csv("../data/fields_description.csv")
pd.set_option('display.max_colwidth', None)
fields_description.head()


# %%
#Loading Actual Telecom Data
telData = load_df_from_excel(
    "../data/teleCo_data.xlsx", na_values=['undefined'])
pd.set_option('max_column', None)
telData.head()

# %% [markdown]
# ### Guidance Sub-tasks
# %% [markdown]
# #### Top 10 Handsets used by customers

# %%
#Findig top 10 based on unique value counts
hand_set = telData['Handset Type'].value_counts()[:10]
#Saving the data
results.add_data('top_10_handsets', hand_set)
#result
hand_set

# %% [markdown]
# #### Top 3 Handset Manufacturers

# %%
#Findig top 3 based on unique value counts
hand_set_manuf = telData['Handset Manufacturer'].value_counts()[:3]
#Saving the data
results.add_data('top_3_handset_manufacturers', hand_set_manuf)
#result
hand_set_manuf

# %% [markdown]
# #### Top 5 Handset per top 3 Handset Manufacturer

# %%
#Top 5 Apple Handsets
#Get apple manufacture handsets only
apple_df = telData[telData['Handset Manufacturer'] == 'Apple']
apple_df = apple_df.loc[:, ['Handset Type']]
best_apple = apple_df.value_counts()[:5]

#Saving the data
results.add_data('best_5_apple_handsets', best_apple)
#result
best_apple


# %%
#Top 5 Samsung Handsets
#Get samsung manufacture handsets only
samsung_df = telData[telData['Handset Manufacturer'] == 'Samsung']
samsung_df = samsung_df.loc[:, ['Handset Type']]
best_samsung = samsung_df.value_counts()[:5]

#Saving the data
results.add_data('best_5_samsung_handsets', best_samsung)
#result
best_samsung


# %%
#Top 5 Huawei Handsets
#Get huawei manufacture handsets only
huawei_df = telData[telData['Handset Manufacturer'] == 'Huawei']
huawei_df = huawei_df.loc[:, ['Handset Type']]
best_huawei = huawei_df.value_counts()[:5]

#Saving the data
results.add_data('best_5_huawei_handsets', best_huawei)
#result
best_huawei


# %%
#Top of all
#Grouping Handset type and Hanset Manufaturer colums and sorting them based on their size
new_df = telData.loc[:, ['Handset Type', 'Handset Manufacturer']]
value = new_df.groupby(['Handset Manufacturer', 'Handset Type']).size()
total_list = pd.Series(dtype='object')
for i in hand_set_manuf.index:
    total_list = total_list.append(value[i])
top_5 = total_list.sort_values(ascending=False)[:5]

#Saving the data
results.add_data('best_5_handsets', top_5)
#result
top_5

# %% [markdown]
# #### Interpretation and Recommendation
# %% [markdown]
# The phones widely used in the network are manufactured by the Apple, Samsung and Huawei, contributing for 164,827 users. The most used phone is Hwawei B528S-23A covering 19752 of the phones followed by Apple iPhpone 6S, 6, 7 and Se. The Huawei B528S-23A is favoured by users, so we know that selling similar Huawei phones, make promotions on its newer version if it comes and selling the same phone are profitable. We also can infer that iPhones are the next widely used phones and can implement similar marketing or business ideas to them as well.
# %% [markdown]
# ## Variable Identification
# %% [markdown]
# For better understanding of the columns, exploring the data description [here](https://docs.google.com/spreadsheets/d/1pcNqeUeIph6xAQzlI54KCvi8HM91SUNeeDbdOq3rvbE/edit?usp=sharing) will help
# %% [markdown]
# ## Data Understanding

# %%
#Instantiating the DataInfo Class to use it data extraction methods
explorer = DataInfo(telData)


# %%
#column extractor method usage
explorer.get_columns()


# %%
#calling method that gives details like rows and columns, memory usage and more
explorer.get_basic_description()


# %%
#calling a description information providing method
explorer.get_description()


# %%
#Method that gives description on a single column
explorer.get_column_description('Bearer Id')


# %%
# Data Inforamtion regarding null-values and type based on Columns
explorer.get_memory_usage()

# %% [markdown]
# ### Task 1.1
# %% [markdown]
# #### Number of xDR Sessions,Session duration, the total download (DL) and upload (UL) data, the total data volume (in Bytes) during this session for each application Aggregated per User('MSISDN/Number')
#

# %%
pd.set_option('display.float_format', '{:.2f}'.format)


# %%
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
ex1

# %% [markdown]
# ## Data Cleaning and Manipulation

# %%
# instantiating DataCleaner class to use it data cleaning methods
cleaner = DataCleaner(explorer.df)

# %% [markdown]
# #### Fixing Columns

# %%
# changing datatypes
explorer.get_dataframe_columns_unique_value_count()


# %%
#removing single valued columns
single_value_cols = explorer.get_dataframe_columns_unique_value_count()
cleaner.remove_single_value_columns(single_value_cols)

# %% [markdown]
# #### Standardizing Values

# %%
#changing bytes to megabytes
cleaner.convert_bytes_to_megabytes(['Total UL (Bytes)', 'Total DL (Bytes)', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                                    'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)'])

# %% [markdown]
# #### Filtering Data

# %%
# method that eliminates duplicate values
cleaner.remove_duplicates()

# %% [markdown]
# #### Fixing Missing Values

# %%
explorer = DataInfo(cleaner.df)


# %%
explorer.get_columns_with_missing_values()


# %%
# get the number of total missing values
explorer.get_total_missing_values()


# %%
# get all columns having missing values
explorer.get_column_based_missing_values()


# %%
# provide missing percentage of each column
explorer.get_column_based_missing_percentage()


# %%
# method that gets columns having missing values greater than provided percentage
explorer.get_columns_missing_percentage_greater_than(30.0)


# %%
#merging columns to generate new values
#function for merging the columns, basic subtraction
def merge_add(col1: pd.Series, col2: pd.Series):
    return col1.add(col2)


cleaner.create_new_columns_from('TCP retransmission (Bytes)',
                                'TCP UL Retrans. Vol (Bytes)', 'TCP DL Retrans. Vol (Bytes)', merge_add)
tcp_transmition_col = cleaner.df.loc[:, ['TCP retransmission (Bytes)']]


# %%
cleaner.df.info()


# %%
# remove columns that have high missing value percentages than the provided value
cleaner.remove_unwanted_columns(
    explorer.get_columns_missing_percentage_greater_than(30.0))
cleaner.df.shape


# %%
#add 'TCP retransmission (Bytes)' again
cleaner.df['TCP retransmission (Bytes)'] = tcp_transmition_col


# %%
# remove the last row of the dataframe
cleaner.df.drop([cleaner.df.index[-1]], inplace=True)
cleaner.df.shape


# %%
#reinstantiate DataInfo from the cleaners dataframe
explorer = DataInfo(cleaner.df)


# %%
# identify columns having missing values
explorer.get_columns_with_missing_values()


# %%
explorer.get_column_based_missing_values()

# %% [markdown]
# Check if skew for numeric missing values and if not fill missing values with mean value but if skew use median

# %%
# calling method that fills numeric values with median or mean based on skeewness
cleaner.fill_numeric_values(explorer.get_columns_with_missing_values())
cleaner.df.sample(10)


# %%
explorer = DataInfo(cleaner.df)
explorer.get_columns_with_missing_values()


# %%
# using method to forward fill and bfill the remaining non-numeric values
cleaner.fill_non_numeric_values(
    explorer.get_columns_with_missing_values(), ffill=True, bfill=True)

# %% [markdown]
# #### Adding Merged and Spliting Columns and Standardizing

# %%
#spliting date-time value to separate date and time columns for each start and end. (start_date, start_time, end_date, end_time)
cleaner.separate_date_time_column('Start', 'start')
cleaner.separate_date_time_column('End', 'end')


# %%
#merging columns to generate new values
#function for merging the columns, basic subtraction
def merge_sub(col1: pd.Series, col2: pd.Series):
    return col1.sub(col2)


def merge_add(col1: pd.Series, col2: pd.Series):
    return col1.add(col2)


#Total Data
cleaner.create_new_columns_from(
    'Total Data (MegaBytes)', 'Total UL (MegaBytes)', 'Total DL (MegaBytes)', merge_add)
#Total Social Media Data
cleaner.create_new_columns_from('Total Social Media Data (MegaBytes)',
                                'Social Media UL (MegaBytes)', 'Social Media DL (MegaBytes)', merge_add)
#Total Google Data
cleaner.create_new_columns_from('Total Google Data (MegaBytes)',
                                'Google UL (MegaBytes)', 'Google DL (MegaBytes)', merge_add)
#Total Email Data
cleaner.create_new_columns_from(
    'Total Email Data (MegaBytes)', 'Email UL (MegaBytes)', 'Email DL (MegaBytes)', merge_add)
#Total Youtube Data
cleaner.create_new_columns_from('Total Youtube Data (MegaBytes)',
                                'Youtube UL (MegaBytes)', 'Youtube DL (MegaBytes)', merge_add)
#Total Netflix Data
cleaner.create_new_columns_from('Total Netflix Data (MegaBytes)',
                                'Netflix UL (MegaBytes)', 'Netflix DL (MegaBytes)', merge_add)
#Total Gaming Data
cleaner.create_new_columns_from('Total Gaming Data (MegaBytes)',
                                'Gaming UL (MegaBytes)', 'Gaming DL (MegaBytes)', merge_add)
#Total Other Data
cleaner.create_new_columns_from(
    'Total Other Data (MegaBytes)', 'Other UL (MegaBytes)', 'Other DL (MegaBytes)', merge_add)
#Jitter
cleaner.create_new_columns_from(
    'Jitter (ms)', 'Activity Duration UL (ms)', 'Activity Duration DL (ms)', merge_sub)
#Avg Delay
cleaner.create_new_columns_from(
    'Avg Delay (ms)', 'Avg RTT UL (ms)', 'Avg RTT DL (ms)', merge_add)
#Avg Throughput
cleaner.create_new_columns_from(
    'Avg Throughput (kbps)', 'Avg Bearer TP UL (kbps)', 'Avg Bearer TP DL (kbps)', merge_add)


# %%
# Changing the specified columns to category type using cleaners function
cleaner.change_columns_type_to(['IMSI', 'Handset Manufacturer',
                                'Handset Type', 'IMEI', 'MSISDN/Number', 'Bearer Id'], 'category')


# %%
#convert to hour
def ms_to_hr(col: pd.Series):
    convertion_value = 1 / (1000 * 60 * 60)
    return col * convertion_value


cleaner.standardized_column(['Dur. (ms)', 'Dur. (ms).1'], [
                            'Dur. (hr)', 'Total Duration (hr)'], ms_to_hr)

#convert to second


def ms_to_sec(col: pd.Series):
    convertion_value = 1 / 1000
    return col * convertion_value


cleaner.standardized_column(['Start ms', 'End ms', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)'], [
                            'Start sec', 'End sec', 'Activity Duration DL (sec)', 'Activity Duration UL (sec)'], ms_to_sec)

# %% [markdown]
# #### Save Generated Clean Data

# %%
#optimize dataframe
cleaner.optimize_df()


# %%
#save the final cleaned data
cleaner.save_clean_data('../data/teleCo_clean_data.csv')


# %%
#saving our results to a pickle file
results.save_data('../data/overview_results.pickle')


####################3
######################3
#########################

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %% [markdown]
# ## User Engagement analysis

# %%
sys.path.insert(0, '../scripts')


# %%
# Setting Notebook preference options, and
# Importing External Modules
pd.set_option('max_column', None)
pd.set_option('display.float_format', '{:.2f}'.format)
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
# Importing Local Modules


# %%
results = ResultPickler()

# %% [markdown]
# We are gonna track the user’s engagement using the following engagement metrics:
# - sessions frequency (by xDr session identifier which is bearer id)
# - the duration of the session (by Total Duration of the xDR (in ms) which is Total Duration in our clean data)
# - the sessions total traffic (download and upload (bytes)) (by the sum of Total DL (Bytes) and Total UL (Bytes) which in Total Data (MegaBytes) in our clean data)
#

# %%
#Fetching and selecting the specified metric columns only
clean_data = load_df_from_csv("../data/teleCo_clean_data.csv")
cleaner = DataCleaner(clean_data)
cleaner.remove_unwanted_columns(cleaner.df.columns[0])
clean_df = cleaner.df
# df = cleaner.change_columns_type_to(['IMSI','Handset Manufacturer','Handset Type','IMEI','MSISDN/Number','Bearer Id'],'category')
# df = cleaner.df.loc[:,['Bearer Id','Total Duration (hr)','Total Data (MegaBytes)']]


# %%
df = clean_df.copy(deep=True)
aggs_by_col = {'Bearer Id': 'count',
               'Total Duration (hr)': 'sum',
               'Total Data (MegaBytes)': 'sum'
               }

df = df.groupby('MSISDN/Number').agg(aggs_by_col)
df

# %% [markdown]
# ### Top 10 users based on Each Metric
# %% [markdown]
# #### Based on Frequency on Sessions (Bearer Id count)

# %%
dataManipulator = DataManipulator(df)
top_10_session = dataManipulator.get_top_sorted_by_column(
    'Bearer Id', length=10)
#Saving the data
results.add_data('top_10_session', top_10_session)
#result
top_10_session

# %% [markdown]
# #### Based on Duration of Session (Total Duration (hr))

# %%
top_10_duration = dataManipulator.get_top_sorted_by_column(
    'Total Duration (hr)', length=10)
#Saving the data
results.add_data('top_10_duration', top_10_duration)
#result
top_10_duration

# %% [markdown]
# #### Based on sessions total traffic (Total Data (MegaBytes))

# %%
top_10_total_traffic = dataManipulator.get_top_sorted_by_column(
    'Total Data (MegaBytes)', length=10)
#Saving the data
results.add_data('top_10_total_traffic', top_10_total_traffic)
#result
top_10_total_traffic

# %% [markdown]
# ### Normalizing each engagement metric
# %% [markdown]
# #### Normalizing Bearer Id

# %%
# graph_utils.plot_hist(df, column='Bearer Id', color='red')


# %%
# graph_utils.plot_hist(df, column='Total Duration (hr)', color='red')


# %%
# graph_utils.plot_hist(df, column='Total Data (MegaBytes)', color='red')


# %%
# imp_df = df.copy(deep=True)
# dataManipulator = DataManipulator(imp_df)


# %%
# scaled_df = dataManipulator.scale_column('Bearer Id')
# scaled_df = dataManipulator.scale_column('Total Duration (hr)')
# scaled_df = dataManipulator.scale_column('Total Data (MegaBytes)')
# scaled_df


# %%
# graph_utils.plot_hist(scaled_df, column='Bearer Id', color='yellow')


# %%
# dataManipulator = DataManipulator(scaled_df)
# normalized_df = dataManipulator.normalize_column('Bearer Id')
# normalized_df = dataManipulator.normalize_column('Total Duration (hr)')
# normalized_df = dataManipulator.normalize_column('Total Data (MegaBytes)')
# normalized_df


# %%
# graph_utils.plot_hist(normalized_df, column='Bearer Id', color='green')


# %%
# graph_utils.plot_hist(normalized_df, column='Total Duration (hr)', color='green')


# %%
# graph_utils.plot_hist(normalized_df, column='Total Data (MegaBytes)', color='green')


# %%


# %% [markdown]
# ### Standardizing the Data

# %%
standard_df = df.copy(deep=True)
dataManipulator = DataManipulator(standard_df)


# %%
std_df = dataManipulator.standardize_column('Bearer Id')
std_df = dataManipulator.standardize_column('Total Duration (hr)')
std_df = dataManipulator.standardize_column('Total Data (MegaBytes)')
std_df


# %%
# graph_utils.plot_hist(std_df, column='Bearer Id', color='green')

# %% [markdown]
# ### Clustering to 3 Groups

# %%
standardized_df = std_df.copy(deep=True)
# clust_df = clust_df.iloc[:,1:]


# %%
kmeans = KMeans(init="random", n_clusters=3, n_init=10,
                max_iter=300, random_state=42)
label = kmeans.fit_predict(standardized_df)
centroids = kmeans.cluster_centers_
print(f'# Centroids of the clustering:\n{centroids}')
print(f'# The number of iterations required to converge: {kmeans.inertia_}')
print(f'# The number of iterations required to converge: {kmeans.n_iter_}')
#Getting index based on clusters
u_labels = np.unique(label)

#plotting the results:
plt.figure(figsize=(10, 5))
plt.title('User K-Means Classification with 3 Groups (Standardized Data)')
for i in u_labels:
    plt.scatter(standardized_df[label == i].iloc[:, 0],
                standardized_df[label == i].iloc[:, 1], marker='o', label=i)
plt.scatter(centroids[:, 0], centroids[:, 1],
            centroids[:, 2], marker='x', color='black')
plt.legend()
plt.show()

# %% [markdown]
# ### Computing the minimum, maximum, average & total non-normalized metrics for each cluster

# %%
cluster_map_standardized = pd.DataFrame()
cluster_map_standardized['data_index'] = standardized_df.index.values
cluster_map_standardized['cluster'] = kmeans.labels_


# %%
standardized_cluster_1 = cluster_map_standardized[cluster_map_standardized.cluster == 0].iloc[:, 0].values.tolist(
)
cluster_1_df = standardized_df.loc[standardized_cluster_1, :]


# %%
standardized_cluster_2 = cluster_map_standardized[cluster_map_standardized.cluster == 1].iloc[:, 0].values.tolist(
)
cluster_2_df = standardized_df.loc[standardized_cluster_2, :]


# %%
standardized_cluster_3 = cluster_map_standardized[cluster_map_standardized.cluster == 2].iloc[:, 0].values.tolist(
)
cluster_3_df = standardized_df.loc[standardized_cluster_3, :]


# %%
cluster_1_df


# %%
#Separate the indexes and using them select rows from the starting main dataframe containing the initial metrics grouped by the MSISDN Number
cluster_1_index = cluster_1_df.index.values.tolist()
cluster_2_index = cluster_2_df.index.values.tolist()
cluster_3_index = cluster_3_df.index.values.tolist()
cluster_1_index[:5]


# %%
#selecting the values using the indexes


# %%
cluster_1_adf = df.loc[cluster_1_index, :]
cluster_2_adf = df.loc[cluster_2_index, :]
cluster_3_adf = df.loc[cluster_3_index, :]
cluster_1_adf.head()


# %%
# calculating the minimum, maximum, average & total values for each column
cluster_1_explorer = DataInfo(cluster_1_adf)
cluster_1_explorer.get_column_dispersion_with_total_params()


# %%
cluster_2_explorer = DataInfo(cluster_2_adf)
cluster_2_explorer.get_column_dispersion_with_total_params()


# %%
cluster_2_explorer.df


# %%
cluster_3_explorer = DataInfo(cluster_3_adf)
cluster_3_explorer.get_column_dispersion_with_total_params()


# %%


# %%
unstandardized_df = df.copy(deep=True)


# %%
km = KMeans(init="random", n_clusters=3, n_init=10,
            max_iter=300, random_state=42)
label = km.fit_predict(unstandardized_df)
centroids = km.cluster_centers_
print(f'# Centroids of the clustering:\n\t{centroids}')
print(f'# The number of iterations required to converge: {km.inertia_}')
print(f'# The number of iterations required to converge: {km.n_iter_}')
#Getting index based on clusters
u_labels = np.unique(label)

#plotting the results:
plt.figure(figsize=(10, 5))
plt.title('User K-Means Classification with 3 Groups (Unstandardized Data)')
for i in u_labels:
    plt.scatter(unstandardized_df[label == i].iloc[:, 0],
                unstandardized_df[label == i].iloc[:, 1], marker='o', label=i)
plt.scatter(centroids[:, 0], centroids[:, 1],
            centroids[:, 2], marker='x', color='black')
plt.legend()
plt.show()

# %% [markdown]
# ### 10 Most Engaged Users Per Application

# %%
engage_df = clean_df.copy(deep=True)


# %%
aggs_by_col = {'Total Social Media Data (MegaBytes)': 'sum',
               'Total Google Data (MegaBytes)': 'sum',
               'Total Email Data (MegaBytes)': 'sum',
               'Total Youtube Data (MegaBytes)': 'sum',
               'Total Netflix Data (MegaBytes)': 'sum',
               'Total Gaming Data (MegaBytes)': 'sum',
               'Total Other Data (MegaBytes)': 'sum', }

engage_df = engage_df.groupby('MSISDN/Number').agg(aggs_by_col)
engage_df

# %% [markdown]
# #### Top 10 Users Engaged in Social Media Activities

# %%
dataManipulator = DataManipulator(engage_df)
top_10_socialmedia_users = dataManipulator.get_top_sorted_by_column(
    'Total Social Media Data (MegaBytes)', length=10)
#Saving the data
results.add_data('top_10_socialmedia_users', top_10_socialmedia_users)
#result
top_10_socialmedia_users

# %% [markdown]
# #### Top 10 Users Engaged in Google related Activities

# %%
top_10_google_users = dataManipulator.get_top_sorted_by_column(
    'Total Google Data (MegaBytes)', length=10)
#Saving the data
results.add_data('top_10_google_users', top_10_google_users)
#result
top_10_google_users

# %% [markdown]
# #### Top 10 Users Engaged in Email related Activities

# %%
top_10_email_users = dataManipulator.get_top_sorted_by_column(
    'Total Email Data (MegaBytes)', length=10)
#Saving the data
results.add_data('top_10_email_users', top_10_email_users)
#result
top_10_email_users

# %% [markdown]
# #### Top 10 Users Engaged in Youtube Activity

# %%
top_10_youtube_users = dataManipulator.get_top_sorted_by_column(
    'Total Youtube Data (MegaBytes)', length=10)
#Saving the data
results.add_data('top_10_youtube_users', top_10_youtube_users)
#result
top_10_youtube_users

# %% [markdown]
# #### Top 10 Users Engaged in Netflix Activity

# %%
top_10_netflix_users = dataManipulator.get_top_sorted_by_column(
    'Total Netflix Data (MegaBytes)', length=10)
#Saving the data
results.add_data('top_10_netflix_users', top_10_netflix_users)
#result
top_10_netflix_users

# %% [markdown]
# #### Top 10 Users Engaged in Gaming related Activities

# %%
top_10_gaming_users = dataManipulator.get_top_sorted_by_column(
    'Total Gaming Data (MegaBytes)', length=10)
#Saving the data
results.add_data('top_10_gaming_users', top_10_gaming_users)
#result
top_10_gaming_users

# %% [markdown]
# #### Top 10 Users Engaged in Other Activities

# %%
top_10_other_users = dataManipulator.get_top_sorted_by_column(
    'Total Other Data (MegaBytes)', length=10)
#Saving the data
results.add_data('top_10_other_users', top_10_other_users)
#result
top_10_other_users

# %% [markdown]
# ### Top 3 Most Used Applications

# %%
most_used_df = clean_df.copy(deep=True)
most_used_df = most_used_df.loc[:, ['Total Google Data (MegaBytes)', 'Total Email Data (MegaBytes)', 'Total Youtube Data (MegaBytes)',
                                    'Total Netflix Data (MegaBytes)', 'Total Gaming Data (MegaBytes)', 'Total Other Data (MegaBytes)']]


# %%
total_app_data_usage = pd.DataFrame(
    most_used_df.sum(), columns=['Total Data (MegaBytes)'])
most_used_apps = total_app_data_usage.sort_values(
    by='Total Data (MegaBytes)', ascending=False).iloc[:4, :]

#Saving the data
results.add_data('most_used_apps', most_used_apps)
#result
most_used_apps

# %% [markdown]
# The Top 3 Most Used Applications are Gaming, Youtube, Netflix in order respectively (considering Other is not really an application)
# %% [markdown]
# ### k-means optimized (elbow method) clustering of users based on metrics
# %% [markdown]
# #### Finding the optimimum k-value

# %%
kmeans_kwargs = {"init": "random", "n_init": 10,
                 "max_iter": 300, "random_state": 42}
# A list holds the SSE values for each k
sse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(standardized_df)
    sse.append(kmeans.inertia_)


# %%
plt.style.use("fivethirtyeight")
plt.plot(range(1, 20), sse)
plt.xticks(range(1, 20))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# %%
kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")
kl.elbow


# %%
kmeans = KMeans(init="random", n_clusters=kl.elbow,
                n_init=10, max_iter=300, random_state=42)
label = kmeans.fit_predict(standardized_df)
centroids = kmeans.cluster_centers_
print(f'# Centroids of the clustering:\n{centroids}')
print(f'# The number of iterations required to converge: {kmeans.inertia_}')
print(f'# The number of iterations required to converge: {kmeans.n_iter_}')
#Getting index based on clusters
u_labels = np.unique(label)

#plotting the results:
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
fig.show()
fig.savefig('../data/engagement_cluster.png')


# %%
standardized_df['cluster'] = label
cleaner = DataCleaner(standardized_df)
cleaner.save_clean_data('../data/engagement_cluster.csv')


# %%
centroids_df = pd.DataFrame(centroids, columns=['Bearer Id', 'Total Duration (hr)','Total Data (MegaBytes)'])
cleaner = DataCleaner(centroids_df)
cleaner.save_clean_data('../data/engagement_centroid.csv')


# %%
#saving our results to a pickle file
results.save_data('../data/engagement_results.pickle')


########################################
# %%

############################################
#################################
##############################

