import ee
import pandas as pd

# Authenticate and initialize the Earth Engine API
# ee.Authenticate()
# ee.Initialize()

# # Define the service account and credentials
# service_account = 'id-0123-project@ee-kevin123.iam.gserviceaccount.com'
# credentials = ee.ServiceAccountCredentials(service_account, '/Users/kd6801/Desktop/ee-kevin123-e9fbc09a5476.json')
# ee.Initialize(credentials)

# Path to your service account key file
key_file_path = '/Users/kd6801/Desktop/Mining-Project/ee-kevin123-e9fbc09a5476.json'
service_account = 'id-0123-project@ee-kevin123.iam.gserviceaccount.com'

# Authenticate and initialize
credentials = ee.ServiceAccountCredentials(service_account, key_file_path)
ee.Initialize(credentials)

# Define the time period and region
start_year = 1990
end_year = 2021
start_day = '01-01'
end_day = '12-31'
observe_country = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_co', 'DRC'))

# Set our boundary box for the region of interest
xmin = 14.63463833
ymin = 3.21977333
xmax = 17.89385167
ymax = 6.947788

# Define the number of rows and columns to partition the area
num_rows = 4
num_cols = 4

# Calculate the width and height of each partition
x_step = (xmax - xmin) / num_cols
y_step = (ymax - ymin) / num_rows

# Initialize a list to store the results
dfs = []

# Harmonization function for Landsat 8 to Landsat 7
def harmonizationRoy(oli):
    slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949])
    itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029])
    y = oli.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'])\
        .resample('bicubic')\
        .subtract(itcp.multiply(10000)).divide(slopes)\
        .set('system:time_start', oli.get('system:time_start'))
    return y.toShort()

# Collect and harmonize Landsat imagery
def getSRcollection(year, start_day, end_year, end_day, sensor):
    sr_collection = ee.ImageCollection(f'LANDSAT/{sensor}/C01/T1_SR')\
        .filterDate(f'{year}-{start_day}', f'{end_year}-{end_day}')
    if sensor == 'LC08':
        sr_collection = sr_collection.map(lambda img: harmonizationRoy(img.unmask()))
    else:
    return sr_collection
        sr_collection = sr_collection.map(lambda img: img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7']).unmask().resample('bicubic').set('system:time_start', img.get('system:time_start')))

# Combine data from different Landsat missions
def getCombinedSRcollection(start_year, start_day, end_year, end_day):
    le5 = getSRcollection(start_year, start_day, end_year, end_day, 'LT05')
    le7 = getSRcollection(start_year, start_day, end_year, end_day, 'LE07')
    lc8 = getSRcollection(start_year, start_day, end_year, end_day, 'LC08')
    return ee.ImageCollection(le7.merge(lc8).merge(le5))

# Add NDVI band to each image
def addNDVI(image):
    ndvi = image.normalizedDifference(['B4', 'B3']).rename('NDVI')
    return image.addBands(ndvi)

# Function to fetch coordinates and convert them to a DataFrame
def fetch_coords_and_save_csv(loss, region, scale=150, max_pixels=1e13):
    coords = ee.Image.pixelLonLat().addBands(loss)
    loss_coords = coords.updateMask(loss).reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=region,
        scale=scale,
        maxPixels=max_pixels
        #bestEffort=True
    )

    lon = loss_coords.get('longitude').getInfo()
    lat = loss_coords.get('latitude').getInfo()
    data = {'Longitude': lon, 'Latitude': lat}
    df = pd.DataFrame(data)
    return df

# Pre-filter the image collection
preFilteredSR = getCombinedSRcollection(start_year, start_day, end_year, end_day)

# Loop through each partition
for row in range(num_rows):
    for col in range(num_cols):
        # Calculate the bounding box for the current partition
        x_start = xmin + col * x_step
        y_start = ymin + row * y_step
        x_end = xmin + (col + 1) * x_step
        y_end = ymin + (row + 1) * y_step
        region = ee.Geometry.Rectangle([x_start, y_start, x_end, y_end])

        # Main image collection with NDVI
        # collectionSR = getCombinedSRcollection(start_year, start_day, end_year, end_day)
        collectionSR_wIndex = preFilteredSR.map(addNDVI)

        # Calculate NDVI anomalies
        reference = collectionSR_wIndex.filterDate('2002-01-01', '2005-12-31').mean()
        observation = collectionSR_wIndex.filterDate('2010-01-01', '2019-12-31').map(lambda image: image.select('NDVI').subtract(reference.select('NDVI')).set('system:time_start', image.get('system:time_start')))
        anomaly = observation.select('NDVI').sum()
        num_images = observation.select('NDVI').count()
        anom_mean = anomaly.divide(num_images)
        loss = anom_mean.lte(-0.15).selfMask()

        # Call the function to process and save CSV for the current partition
        coords_df = fetch_coords_and_save_csv(loss, region)
        dfs.append(coords_df)

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dfs)

# Save the final DataFrame to a CSV file
final_df.to_csv('/Users/kd6801/Desktop/anomaly_coords_combined.csv', index=False)
print("Combined CSV file has been saved locally.")
