import ee
import pandas as pd

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
observe_country = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_co', 'COD'))

# Set our boundary box for the region of interest
xmin = 14.63463833
ymin = 3.21977333
xmax = 17.89385167
ymax = 6.947788



# Define the number of rows and columns to partition the area
num_rows = 6
num_cols = 6

# Calculate the width and height of each partition
x_step = (xmax - xmin) / num_cols
y_step = (ymax - ymin) / num_rows

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
        sr_collection = sr_collection.map(lambda img: img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7']).unmask().resample('bicubic').set('system:time_start', img.get('system:time_start')))
    return sr_collection

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
# def fetch_coords_and_save_csv(loss, region, scale=30, max_pixels=1e13):
#     coords = ee.Image.pixelLonLat().addBands(loss)
#     loss_coords = coords.updateMask(loss).reduceRegion(
#         reducer=ee.Reducer.toList(),
#         geometry=region,
#         scale=scale,
#         maxPixels=max_pixels
#         #bestEffort=True
#     )

#     lon = loss_coords.get('longitude').getInfo()
#     lat = loss_coords.get('latitude').getInfo()
#     data = {'Longitude': lon, 'Latitude': lat}
#     df = pd.DataFrame(data)
#     return df
def fetch_coords_and_save_csv(loss, region, scale=70, max_pixels=1e11, bestEffort=True):
    coords = ee.Image.pixelLonLat().addBands(loss)
    loss_coords = coords.updateMask(loss).reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=region,
        scale=scale,
        maxPixels=max_pixels,
        bestEffort=bestEffort
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

        # Use the pre-filtered collection with NDVI
        collectionSR_wIndex = preFilteredSR.map(addNDVI)

        # Calculate NDVI anomalies
        reference = collectionSR_wIndex.filterDate('2002-01-01', '2005-12-31').mean()
        observation = collectionSR_wIndex.filterDate('2010-01-01', '2019-12-31').map(lambda image: image.select('NDVI').subtract(reference.select('NDVI')).set('system:time_start', image.get('system:time_start')))
        anomaly = observation.select('NDVI').sum()
        num_images = observation.select('NDVI').count()
        anom_mean = anomaly.divide(num_images)
        loss = anom_mean.lte(-0.15).selfMask()

        # Process and save CSV for the current partition
        coords_df = fetch_coords_and_save_csv(loss, region)
        file_name = f'/Users/kd6801/Desktop/Mining-Project/anomaly_coords_row{row}_col{col}.csv'
        coords_df.to_csv(file_name, index=False)
        print(f"CSV file for partition row {row}, column {col} has been saved as {file_name}.")
