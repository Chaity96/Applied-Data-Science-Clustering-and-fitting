import pandas as pd

# Function to read the data from the file
file = "wb_cc_dataset.csv"

def convert_to_WB_format(file):
    df = pd.read_csv(file)

    df = df.set_index('Country Name', inplace = True)
    df = df.drop(
                    ['Country Code' , 'Indicator Name' , 'Indicator Code'] ,
                    axis = 1
                )

    df_formatted = df.T

    return df_formatted


# using "wbdata" to get the data from the World Bank directly
import wbdata

# countries to be extracted
pakistan = "PK"
india = "IN"

# indicators to be extracted
indicator_1 = {"EN.ATM.CO2E.KT" : "CO2 emissions (kt)"}

wbd_df = wbdata.get_dataframe(
                                indicator_1 ,
                                country = pakistan , 
                                convert_date = False
                            )

india_df = wbdata.get_dataframe(
                                indicator_1 ,
                                country = india ,
                                convert_date = False
                                )

# dropping the empty fields of the dataframes :
wbd_df.dropna(inplace = True)
india_df.dropna(inplace = True)


# Visualisng the data :
import matplotlib.pyplot as plt
import numpy as np

wbd_df.plot(
            kind = 'bar' ,
            color = 'green'
            )
plt.title('CO2 emissions (kt) for Pakistan')

india_df.plot(
              kind = 'bar' ,
              color = 'blue'
              )
plt.title('CO2 emissions (kt) for India')

plt.show()

N = len(wbd_df)
index = np.arange(N)
bar_width = 0.35

plt.bar(
        index , 
        wbd_df['CO2 emissions (kt)'] , 
        bar_width , 
        label = 'Pakistan' ,
        color = 'green'
        )

plt.bar(
        index + bar_width , 
        india_df['CO2 emissions (kt)'] , 
        bar_width , 
        label = 'India' , 
        color = 'blue'
        )
        
plt.title('CO2 emissions (kt) of Pakistan & India')
plt.legend()
plt.show()


# Normalizing the Data (MaxAbs Scaling Method) :

from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
scaler.fit(wbd_df)

wbd_df_normalized = scaler.transform(wbd_df)

print(wbd_df_normalized)

# Clustering the Data (MeanShift Clustering Method):

from sklearn.cluster import MeanShift

ms = MeanShift()

ms.fit(wbd_df_normalized)

# Cluster Labels:
labels = ms.labels_

# Adding a new column "Clusters" to the dataframe:
wbd_df['Cluster'] = labels

#print the dataframe
print(wbd_df)

# Visualizing the Clusters : 

plt.scatter(wbd_df.index ,
            wbd_df['CO2 emissions (kt)'] ,
            c = wbd_df['Cluster'] , 
            cmap = 'rainbow')

plt.xlabel('Years')
plt.ylabel('CO2 emissions (kt)')
plt.tick_params(axis = 'x' , labelsize = 7)
plt.legend()
plt.show()

# Printing number of clusters
n_clusters = len(set(labels))
print('Number of clusters:' , n_clusters)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Function to calculate lower and upper limits of the confidence range:
def err_ranges(popt , pcov , x):
    y = func(x , popt[0] , popt[1])
    n = len(x)
    s_err = np.sqrt(np.diag(pcov))
    y_err_low = y - s_err * np.sqrt(n)
    y_err_high = y + s_err * np.sqrt(n)
    return y_err_low, y_err_high

# Define the exponential growth model:
def func(x , a , b):
    return a * np.exp(b * x)

# Load the data
x_data = np.linspace(0 , 5 , 100)
y_data = func(x_data , 2.5 , 1.3)
y_noise = 0.2 * np.random.normal(size = x_data.size)
y_data = y_data + y_noise

# Fit the model to the data
popt, pcov = curve_fit(func , x_data , y_data)

# Make predictions for the next 10 and 20 years:
x_pred_10 = np.linspace(5 , 15 , 2)
y_pred_10 = func(x_pred_10 , *popt)
x_pred_20 = np.linspace(5 , 25 , 2)
y_pred_20 = func(x_pred_20 , *popt)

# Calculate lower and upper limits of the confidence range for the predictions:
y_err_low_10, y_err_high_10 = err_ranges(popt , pcov , x_pred_10)
y_err_low_20, y_err_high_20 = err_ranges(popt , pcov , x_pred_20)

# Plot the data, best fitting function and the confidence range:
plt.plot(x_data , 
         y_data , 
         'o' , 
         label = 'Data')

plt.plot(x_pred_10 , 
         y_pred_10, '-' , 
         label = 'Prediction (10 years)')

plt.plot(x_pred_20 , 
         y_pred_20, '-' , 
         label = 'Prediction (20 years)')

plt.fill_between(x_pred_10 , 
                 y_err_low_10 , 
                 y_err_high_10 , 
                 color = 'gray' , 
                 alpha = 0.2)

plt.fill_between(x_pred_20 , 
                 y_err_low_20 , y_err_high_20 , 
                 color = 'gray' , 
                 alpha = 0.2)
                 
plt.xlabel('Years')
plt.ylabel('Value')
plt.legend()
plt.show()

