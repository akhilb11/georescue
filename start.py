#!/usr/bin/python
#
# import the necessary modules
import pandas as pd
import geopy
import json
from geopy.geocoders import Nominatim
from sklearn.neighbors import RadiusNeighborsClassifier

# import Flask and set it up
from flask import Flask, request
app = Flask(__name__)

# Recieves latitude, longitude, and time in order to predict the dispatch type
def givenAddressTime(lat, long, time):
    # Create dataframe from parameter values in order to compare with values later on
    df_entered = pd.DataFrame({'latitude': [lat], 'longitude': [long], 'received_timestamp': [time]})
    # Read in CSV file and create dataframe
    df = pd.read_csv('./data/sfpd_dispatch_data_subset.csv', sep=',')
    x = df[['latitude', 'longitude', 'received_timestamp']]
    # Convert values in the timestamp column to float so that they can be compared
    x['received_timestamp'] = x['received_timestamp'].apply(convertTimestamp)
    # Creating dataframe "y" so that the values can be fitted against dataframe "x"
    y = df[['call_type']]
    # Using scitools module to create neighbor so that it can figure out what the dispatch of the given values are based on K Nearest Neighbors
    neighbor = RadiusNeighborsClassifier(radius=10.0, weights='distance', outlier_label="Error, cannot predict this far")
    neighbor.fit(x, y.values.ravel())
    return (neighbor.predict(df_entered))

# Converts timestamp field to float so that it can be compared
# Can easily be iterated over each row using the ".apply" method in pandas
def convertTimestamp(row):
    # Strips away the ":" and space that is in the string so that python can get the number values
    splitStamp = row.split(" ")
    splitTime = splitStamp[1].replace(":", "")
    row = float(splitTime)
    # Returns the row value which is now a float
    return row

# Initial method called by "/search" route
# Called when submit button is clicked on the search page
def searchValue(address):
    # Takes input string and splits it up between the string address and the timestamp (which is then made into a float)
    splitAddress = address.split(", ")
    splitAddress[1] = splitAddress[1].replace(":", "")
    # Geolocator set in order to look up the coordinates of the address
    geolocator = Nominatim()
    location = geolocator.geocode(splitAddress[0])
    # The coordinates and the time are then sent into the "givenAddressTime" method and it returns a string which is the dispatch type and is sent by the searchValue method to the html page
    dispatch = "Predicted Dispatch Type: " + str(givenAddressTime(location.latitude, location.longitude, float(splitAddress[1])))
    return dispatch

# Returns json string of dataframe with latitude and longitude of a given dispatch type for the heatmap
def returnCoordinates(dispatchType):
    # reads in the CSV data and sets it to a new dataframe
    df = pd.read_csv('./data/sfpd_dispatch_data_subset.csv', sep=',')
    # Selecting which data is needed
    x = df[['call_type', 'latitude', 'longitude']]
    # Isolating the rows that are equivalent to the clicked dispatch type
    coordinates = df.loc[x['call_type'] == dispatchType]
    coordinates = coordinates[['latitude', 'longitude']]
    # Converting the dataframe to json and sending it back to javascript
    response = coordinates.to_json(orient='values')
    return response

# Returns the data required for the bar graph
def returnBarData():
    # reads in the CSV data and sets it to a new dataframe
    df = pd.read_csv('./data/sfpd_dispatch_data_subset.csv', sep=',')
    # Selecting which data is needed
    x = df[['call_type']]
    # Grouping all the rows in the dataframe by call type and getting rid of the duplicates while also counting how many duplicates each unique value of call type has
    dispatchCount = x.groupby(['call_type']).size().reset_index(name='count')
    # Converting the dataframe to json and sending it back to javascript
    response = dispatchCount.to_json(orient='values')
    return response

def returnAvgTimeData():
    # reads in the CSV data and sets it to a new dataframe
    df = pd.read_csv('./data/sfpd_dispatch_data_subset.csv', sep=',')
    # Selecting which data is needed
    x = df[['call_type', 'on_scene_timestamp', 'received_timestamp']]
    # Getting rid of all the rows that have a blank
    x = x.dropna()
    # Applying the ConvertTimestamp method in order to get float values
    x['received_timestamp'] = x['received_timestamp'].apply(convertTimestamp)
    x['on_scene_timestamp'] = x['on_scene_timestamp'].apply(convertTimestamp)
    # Subtracting the float values to get the total time spent to reach the scene
    x['total_time'] = x['on_scene_timestamp'].sub(x['received_timestamp'])
    # Multiplying all the values by 0.01 to convert it to minutes and taking the absolute value of all values because the method just subtracted with military time
    x['total_time'] = x['total_time'].apply(lambda x: x*0.01).abs()
    # Taking the mean of the total time column based on call type so that the average call response time can be found for each unique dispatch type
    x = x.groupby('call_type', as_index=False)['total_time'].mean()
    finalTimeValues = x[['call_type', 'total_time']]
    # Converting the dataframe to json and sending it back to javascript
    response = finalTimeValues.to_json(orient='values')
    return response

def returnBoxData():
    # reads in the CSV data and sets it to a new dataframe
    df = pd.read_csv('./data/sfpd_dispatch_data_subset.csv', sep=',')
    # Selecting which data is needed
    x = df[['call_type', 'box']]
    # Grouping the duplicates call types and getting rid of all the rows with duplicate box values
    boxCount = x.groupby(['call_type', 'box']).agg({'box':'nunique'})
    # Taking the count of each duplicate call type row so that a new row with the count of the total number of unique boxes each dispatch call has can be found
    boxCount = boxCount.groupby(['call_type']).size().reset_index(name='count')
    finalboxCount = boxCount[['call_type', 'count']]
    # Converting the dataframe to json and sending it back to javascript
    response = finalboxCount.to_json(orient='values')
    return response

def returnZipData():
    # Read in dataframe from file
    df = pd.read_csv('./data/sfpd_dispatch_data_subset.csv', sep=',')
    # Select data from dataframe
    x = df[['zipcode_of_incident', 'box', 'on_scene_timestamp', 'received_timestamp', 'station_area']]
    # Calculate amount of boxes based on ZipCode
    boxCount = x.groupby(['zipcode_of_incident', 'box']).agg({'box':'nunique'})
    boxCount = boxCount.groupby(['zipcode_of_incident']).size().reset_index(name='count')
    finalboxCount = boxCount[['zipcode_of_incident', 'count']]
    # Calculate amount of station areas based on ZipCode
    y = df[['zipcode_of_incident', 'station_area']]
    stationCount = y.groupby(['zipcode_of_incident', 'station_area']).agg({'station_area':'nunique'})
    stationCount = stationCount.groupby(['zipcode_of_incident']).size().reset_index(name='station_count')
    finalstationCount = stationCount[['zipcode_of_incident', 'station_count']]
    # Calculate amount of station areas based on ZipCode
    x = x.dropna()
    x['received_timestamp'] = x['received_timestamp'].apply(convertTimestamp)
    x['on_scene_timestamp'] = x['on_scene_timestamp'].apply(convertTimestamp)
    x['total_time'] = x['on_scene_timestamp'].sub(x['received_timestamp'])
    x['total_time'] = x['total_time'].apply(lambda x: x*0.01).abs()
    x = x.groupby('zipcode_of_incident', as_index=False)['total_time'].mean()
    finalTimeValues = x[['zipcode_of_incident', 'total_time']]
    # Put dataframes together
    finalDF = finalboxCount.merge(finalstationCount,on='zipcode_of_incident').merge(finalTimeValues,on='zipcode_of_incident')
    finalDF['zipcode_of_incident'] = finalDF['zipcode_of_incident'].astype(str)
    # Converting the dataframe to json and sending it back to javascript
    response = finalDF.to_json(orient='values')
    return response

# Setting main app route to index (this is the homepage)
@app.route('/')
def home():
    with open("index.html", "r") as f: return f.read()

# Setting app route to go to the main page in case user clicks on logo
@app.route('/index.html')
def index():
    with open("index.html", "r") as f: return f.read()

# Setting route for heatmap
@app.route('/heatmap.html')
def heatmap():
    with open("heatmap.html", "r") as f: return f.read()

# Setting route for search page
@app.route('/search.html')
def search():
    with open("search.html", "r") as f: return f.read()

# Setting route for about page
@app.route('/about.html')
def about():
    with open("about.html", "r") as f: return f.read()

# Setting route for stats page
@app.route('/stats.html')
def stats():
    with open("stats.html", "r") as f: return f.read()

# Setting route for "/api" in order to send post or get request to python method
@app.route('/api', methods=['GET', 'POST'])
def api():
    result = searchValue(request.form['i1'])
    return result

# Setting route for "/map" in order to send post or get request to python method and retrieve heatmap data
@app.route('/map', methods=['GET', 'POST'])
def map():
    result = returnCoordinates(request.form['dispatch'])
    return result

# Setting route for "/barData" in order to send post or get request to python method and retrieve data for bar graph
@app.route('/barData', methods=['GET', 'POST'])
def barData():
    result = returnBarData()
    return result

# Setting route for "/avgData" in order to send post or get request to python method and retrieve data for total average graph
@app.route('/avgData', methods=['GET', 'POST'])
def avgData():
    result = returnAvgTimeData()
    return result

# Setting route for "/boxData" in order to send post or get request to python method and retrieve data for total amount of boxes graph
@app.route('/boxData', methods=['GET', 'POST'])
def boxData():
    result = returnBoxData()
    return result

# Setting route for "/zipData" in order to send post or get request to python method and retrieve the data needed for the combo graph
@app.route('/zipData', methods=['GET', 'POST'])
def zipData():
    result = returnZipData()
    return result

if __name__ == '__main__':
    app.run(debug=True)
