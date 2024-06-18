import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from mpi4py import MPI
import sys

# Load the CSV file
df = pd.read_csv('/home/kaiwen1/30123-Project-Kaiwen/matched_without_distance_1000m.csv')

# Filter the data to only include rows where potential_mining = 0
df = df[df['potential_mining'] == 0]

# Randomly sample 300000 data points from the filtered data
df = df.sample(n=200000, random_state=21)

# Load the GeoJSON and shapefiles
local_roads = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-local-roads.geojson')
main_roads = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-main-roads.geojson')
villages = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-villages.geojson')
protected_areas = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-protected/protected_areas.shp')
waterways = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-water/hotosm_cod_waterways_lines_shp.shp')

target_crs = "EPSG:4326"  # Common CRS (WGS 84)
projected_crs = "EPSG:3857"  # Common projected CRS (Pseudo-Mercator)

# Function to set CRS if it's not set
def set_crs_if_needed(gdf, target_crs):
    """
    Set the CRS (Coordinate Reference System) of a GeoDataFrame if it is not already set.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to check and set the CRS for.
    target_crs (str): The target CRS to set if not already set.

    Returns:
    GeoDataFrame: The GeoDataFrame with the CRS set.
    """
    if gdf.crs is None:
        gdf.set_crs(target_crs, inplace=True)
    return gdf

# Ensure all GeoDataFrames have the same CRS
local_roads = set_crs_if_needed(local_roads, target_crs)
main_roads = set_crs_if_needed(main_roads, target_crs)
villages = set_crs_if_needed(villages, target_crs)
protected_areas = set_crs_if_needed(protected_areas, target_crs)
waterways = set_crs_if_needed(waterways, target_crs)

# Convert the CSV data to a GeoDataFrame
df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
mining_locations = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# Reproject all GeoDataFrames to a projected CRS
mining_locations = mining_locations.to_crs(projected_crs)
villages = villages.to_crs(projected_crs)
waterways = waterways.to_crs(projected_crs)
local_roads = local_roads.to_crs(projected_crs)
main_roads = main_roads.to_crs(projected_crs)
protected_areas = protected_areas.to_crs(projected_crs)

# Function to calculate the closest distance
def calculate_nearest_distance(gdf1, gdf2):
    """
    Calculate the nearest distance from each geometry in gdf1 to the nearest geometry in gdf2.

    Parameters:
    gdf1 (GeoDataFrame): The GeoDataFrame containing the geometries to calculate distances from.
    gdf2 (GeoDataFrame): The GeoDataFrame containing the geometries to calculate distances to.

    Returns:
    list: A list of distances from each geometry in gdf1 to the nearest geometry in gdf2.
    """
    distances = []
    for geom in gdf1.geometry:
        nearest_geom = gdf2.geometry[gdf2.distance(geom).idxmin()]
        nearest_dist = geom.distance(nearest_geom)
        distances.append(nearest_dist)
    return distances

def main():
    """
    Perform parallel processing of mining locations using MPI.

    Divides the mining locations among MPI processes, calculates the nearest distances
    to various features (villages, waterways, roads, protected areas), and saves the results
    to a CSV file.
    """
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide the mining_locations among the processes
    chunk_size = len(mining_locations) // size
    if rank == size - 1:
        chunk = mining_locations.iloc[rank * chunk_size:]
    else:
        chunk = mining_locations.iloc[rank * chunk_size:(rank + 1) * chunk_size]

    # Print the number of data points each process is handling and flush the output
    print(f"Rank {rank}: Processing {len(chunk)} data points", flush=True)

    # Calculate closest distances
    chunk['distance_to_village'] = calculate_nearest_distance(chunk, villages)
    chunk['distance_to_waterway'] = calculate_nearest_distance(chunk, waterways)
    chunk['distance_to_local_road'] = calculate_nearest_distance(chunk, local_roads)
    chunk['distance_to_main_road'] = calculate_nearest_distance(chunk, main_roads)
    chunk['distance_to_protected_area'] = calculate_nearest_distance(chunk, protected_areas)

    # Print statement to track progress and flush the output
    for i in range(len(chunk)):
        print(f"Rank {rank}: Processed point {i + 1}/{len(chunk)}", flush=True)

    # Gather results from all processes
    gathered_chunks = comm.gather(chunk, root=0)

    if rank == 0:
        # Concatenate all chunks
        result_df = pd.concat(gathered_chunks)
        # Save the updated DataFrame to a new CSV file
        result_df.to_csv('/home/kaiwen1/30123-Project-Kaiwen/complete_data_y0_200k.csv', index=False)
        # Display the first few rows of the updated DataFrame
        print(result_df.head(), flush=True)
        print("File saved successfully.", flush=True)

if __name__ == "__main__":
    main()
