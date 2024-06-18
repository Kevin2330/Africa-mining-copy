import logging
import boto3
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_gcps
from rasterio.session import AWSSession
from lxml import etree  # Replaced xml.etree.ElementTree with lxml
import io
import csv

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client("s3")
bucket_name = "usgs-landsat"
aws_session = AWSSession(requester_pays=True)

logging.basicConfig(level=logging.ERROR)

def lambda_handler(event, context):
    all_coords_dict = {}    

    # Define the bucket name
    bucket_name = 'usgs-landsat'

    # Define the periods for calculation
    periods = {
        '2000_2005': {
            'years': [ '2003','2004' '2005'],
            'prefix': 'collection02/level-2/standard/etm/'
        },
        '2018_2023': {
            'years': [ '2021','2022','2023'],
            'prefix': 'collection02/level-2/standard/oli-tirs/'
        }
    }

    for i in event:
        path = i["path"]
        row = i["row"]

        # Initialize dictionaries for both periods
        ndvi_2000_2005 = []
        ndvi_2018_2023 = []
        bands_2023 = {'band1': None, 'band2': None, 'band3': None, 'band6': None, 'band10': None}
        transform = None
        crs = None
        
        for year in periods['2000_2005']['years']:
            ndvi, _, transform, crs = process_scene(path, row, year, periods['2000_2005']['prefix'])
            if ndvi is not None:
                ndvi_2000_2005.append(ndvi)
        
        for year in periods['2018_2023']['years']:
            ndvi, bands, transform, crs = process_scene(path, row, year, periods['2018_2023']['prefix'])
            if ndvi is not None:
                ndvi_2018_2023.append(ndvi)
                if year == '2023':  # Only take band values from 2023
                    bands_2023 = bands
        
        if ndvi_2000_2005 and ndvi_2018_2023:
            max_shape_2000_2005 = np.max([ndvi.shape for ndvi in ndvi_2000_2005],axis = 0)
            max_shape_2018_2023 = np.max([ndvi.shape for ndvi in ndvi_2018_2023],axis = 0)
            target_shape = (max(max_shape_2000_2005[0], max_shape_2018_2023[0]),
                            max(max_shape_2000_2005[1], max_shape_2018_2023[1]))
            
            resized_ndvi_2000_2005 = [resize_array(ndvi, target_shape) for ndvi in ndvi_2000_2005]
            resized_ndvi_2018_2023 = [resize_array(ndvi, target_shape) for ndvi in ndvi_2018_2023]
            
            mean_ndvi_2000_2005 = np.mean(resized_ndvi_2000_2005, axis=0)
            mean_ndvi_2018_2023 = np.mean(resized_ndvi_2018_2023, axis=0)
            ndvi_loss = mean_ndvi_2000_2005 - mean_ndvi_2018_2023
            
            resized_bands_2023 = {band: resize_array(bands_2023[band][0], target_shape) for band in bands_2023.keys()}
            
            get_coordinates(ndvi_loss, resized_bands_2023, transform, crs, all_coords_dict)

    # Save the filtered coordinates to a CSV file
    output_csv = 'NDVI_loss_filtered_coordinates-{}.csv'.format(event[0])
    # Prepare the data
    data_tuples = [
        (
            k[0], k[1], v['NDVI_Loss'], 
            v['Band1_2023'], v['Band2_2023'], v['Band3_2023'], v['Band6_2023'], v['Band10_2023']
        ) for k, v in all_coords_dict.items()
    ]
    # Define column headers
    columns = ['Longitude', 'Latitude', 'NDVI_Loss', 
            'Band1_2023', 'Band2_2023', 'Band3_2023', 'Band6_2023', 'Band10_2023']

    # Write to CSV using csv module
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)

    # Write the header
    csv_writer.writerow(columns)

    # Write the data rows
    csv_writer.writerows(data_tuples)

    # Get CSV content
    csv_content = csv_buffer.getvalue()
    
    # Get bucket name and file name from the event
    bucket_name = "africamining"
    file_name = f"NDVI_loss_filtered_coordinates-{event[0]}.csv"
    
    # Upload CSV to S3
    s3.put_object(
        Bucket=bucket_name,
        Key=file_name,
        Body=csv_buffer.getvalue()
    )

def calculate_ndvi(red_band, nir_band):
    '''
    Calculates the Normalized Difference Vegetation Index (NDVI) from red and near-infrared (NIR) bands.

    NDVI is a measure of vegetation health and is calculated using the following formula:
    NDVI = (NIR - Red) / (NIR + Red)

    Parameters:
    - red_band (numpy.ndarray): A 2D array representing the red band of the imagery.
    - nir_band (numpy.ndarray): A 2D array representing the near-infrared (NIR) band of the imagery.

    Returns:
    - numpy.ndarray: A 2D array representing the NDVI values, with the same shape as the input bands.
    '''
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi

def get_coordinates(ndvi_loss_array, bands_2023, transform, crs, all_coords_dict, threshold=0.15):
    '''
    Extracts coordinates with significant NDVI loss and their corresponding band values.

    This function iterates over an NDVI loss array, identifies cells where the NDVI loss exceeds 
    a specified threshold, and retrieves their geographic coordinates. The coordinates are 
    transformed to the EPSG:4326 coordinate reference system (latitude and longitude). It also 
    collects corresponding band values for each coordinate and stores this information in a 
    dictionary.

    Parameters:
    - ndvi_loss_array (numpy.ndarray): A 2D array representing the NDVI loss values.
    - bands_2023 (dict): A dictionary containing 2D arrays of band values for the year 2023. 
                         Expected keys are 'band1', 'band2', 'band3', 'band6', and 'band10'.
    - transform (affine.Affine): The affine transformation associated with the NDVI loss array.
    - crs (rasterio.crs.CRS): The coordinate reference system of the NDVI loss array.
    - all_coords_dict (dict): A dictionary to store the extracted coordinates and their 
                              corresponding NDVI loss and band values.
    - threshold (float, optional): The NDVI loss threshold for selecting significant coordinates. 
                                   Defaults to 0.15.

    Returns:
    - list: A list of tuples, each containing the longitude and latitude of coordinates with 
            significant NDVI loss.
    '''
    coords = []
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    for y in range(ndvi_loss_array.shape[0]):
        for x in range(ndvi_loss_array.shape[1]):
            ndvi_loss_v = ndvi_loss_array[y, x]
            if ndvi_loss_v > threshold: 
                proj_x, proj_y = rasterio.transform.xy(transform, y, x)
                lon, lat = transformer.transform(proj_x, proj_y)
                coords.append((lon, lat))
                all_coords_dict[(lon, lat)] = {
                    'NDVI_Loss': ndvi_loss_v,
                    'Band1_2023': bands_2023['band1'][y, x],
                    'Band2_2023': bands_2023['band2'][y, x],
                    'Band3_2023': bands_2023['band3'][y, x],
                    'Band6_2023': bands_2023['band6'][y, x],
                    'Band10_2023': bands_2023['band10'][y, x]
                }
    print(ndvi_loss_v)
    return coords

def get_transform(src):
    '''
    Retrieves the affine transformation of a raster dataset.

    This function checks if Ground Control Points (GCPs) are present in the source raster dataset.
    If GCPs are available, it computes and returns the affine transformation derived from these GCPs.
    Otherwise, it returns the default affine transformation of the raster dataset.

    Parameters:
    - src (rasterio.io.DatasetReader): The source raster dataset from which to retrieve the affine transformation.

    Returns:
    - affine.Affine: The affine transformation of the raster dataset.
    '''
    if src.gcps[0]:
        gcps, gcp_transform = src.gcps
        return from_gcps(gcps)
    else:
        return src.transform

def get_cloud_cover(metadata_key):
    '''
    Resizes a 2D numpy array to the specified shape using nearest-neighbor interpolation.

    This function takes a 2D numpy array and resizes it to the given shape using nearest-neighbor
    interpolation. The resulting array has the same data type as the input array.

    Parameters:
    - arr (numpy.ndarray): The input 2D array to be resized.
    - shape (tuple): A tuple (new_rows, new_cols) specifying the shape of the resized array.

    Returns:
    - numpy.ndarray: The resized 2D array with the specified shape.
    '''
    obj = s3.get_object(Bucket=bucket_name, Key=metadata_key, RequestPayer='requester')
    metadata_content = obj['Body'].read()
    tree = etree.parse(io.BytesIO(metadata_content))
    root = tree.getroot()
    for elem in root.iter():
        if 'CLOUD_COVER' in elem.tag:
            return float(elem.text)
    return float('inf')  # If CLOUD_COVER is not found, return infinity

def process_scene(path, row, year, prefix):
    '''
    Processes a Landsat scene to calculate NDVI and retrieve values for specified bands.

    This function searches for a Landsat scene in the specified S3 bucket, selects the scene with 
    the least cloud cover, and calculates the NDVI (Normalized Difference Vegetation Index) 
    using the red and near-infrared (NIR) bands. It also retrieves values for other specified 
    bands (band1, band2, band3, band6, and band10).

    Parameters:
    - path (int): The path number of the Landsat scene.
    - row (int): The row number of the Landsat scene.
    - year (int): The year of the Landsat scene.
    - prefix (str): The prefix to use for constructing the S3 bucket path.

    Returns:
    - tuple: A tuple containing:
        - ndvi (list of numpy.ndarray): The calculated NDVI values.
        - bands_values (dict of lists of numpy.ndarray): The retrieved values for the specified bands.
        - transform (affine.Affine): The affine transformation of the red band.
        - crs (rasterio.crs.CRS): The coordinate reference system of the red band.

    '''
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f'{prefix}{year}/{path:03d}/{row:03d}/', RequestPayer='requester')
    ndvi_values = []
    bands_values = {'band1': [], 'band2': [], 'band3': [], 'band6': [], 'band10': []}
    print(response)
    if 'Contents' in response:
        scenes = {}
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('_MTL.xml'):
                cloud_cover = get_cloud_cover(key)
                scenes[key] = cloud_cover
        
        sorted_scenes = sorted(scenes.items(), key=lambda item: item[1])
        if sorted_scenes:
            selected_scene = sorted_scenes[0][0].rsplit('/', 1)[0]
            file_name = sorted_scenes[0][0].rsplit('/', 1)[1].replace('_MTL.xml', '')
            band_keys = {
                'red': selected_scene + '/' + file_name + "_SR_B3.TIF" if 'etm' in prefix else selected_scene + '/' + file_name + "_SR_B4.TIF",
                'nir': selected_scene + '/' + file_name + "_SR_B4.TIF" if 'etm' in prefix else selected_scene + '/' + file_name + "_SR_B5.TIF",
                'band1': selected_scene + '/' + file_name + "_SR_B1.TIF",
                'band2': selected_scene + '/' + file_name + "_SR_B2.TIF",
                'band3': selected_scene + '/' + file_name + "_SR_B3.TIF",
                'band6': None if 'etm' in prefix else selected_scene + '/' + file_name + "_SR_B6.TIF",
                'band10': None if 'etm' in prefix else selected_scene + '/' + file_name + "_ST_B10.TIF"
            }

            try:
                with rasterio.Env(aws_session):
                    with rasterio.open(f's3://{bucket_name}/{band_keys["red"]}', RequestPayer='requester') as red_src:
                        red_band = red_src.read(1).astype("float32")
                        transform = get_transform(red_src)
                        crs = red_src.crs
                        if crs is None or not crs.is_valid:
                            crs = rasterio.crs.from_epsg(32633)
                    with rasterio.open(f's3://{bucket_name}/{band_keys["nir"]}', RequestPayer='requester') as nir_src:
                        nir_band = nir_src.read(1).astype("float32")
                    ndvi = calculate_ndvi(red_band, nir_band)
                    ndvi_values.append(ndvi)

                    for band in bands_values.keys():
                        if band_keys[band] is not None:
                            with rasterio.open(f's3://{bucket_name}/{band_keys[band]}', RequestPayer='requester') as band_src:
                                bands_values[band].append(band_src.read(1).astype("float32"))
                    
                    return ndvi, bands_values, transform, crs
            except Exception as e:
                print(f'Error processing {year} path {path, row}: {e}')
    return None, None, None, None

def resize_array(arr, shape):
    '''
    Resizes a 2D numpy array to the specified shape using nearest-neighbor interpolation.

    This function takes a 2D numpy array and resizes it to the given shape using nearest-neighbor
    interpolation. The resulting array has the same data type as the input array. The reason why we have this function
    is because the landsat data size for different time period is different.

    Parameters:
    - arr (numpy.ndarray): The input 2D array to be resized.
    - shape (tuple): A tuple (new_rows, new_cols) specifying the shape of the resized array.

    Returns:
    - numpy.ndarray: The resized 2D array with the specified shape.
    '''
    result = np.empty(shape, dtype=arr.dtype)
    row_ratio, col_ratio = arr.shape[0] / shape[0], arr.shape[1] / shape[1]
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i, j] = arr[int(i * row_ratio), int(j * col_ratio)]
    
    return result
