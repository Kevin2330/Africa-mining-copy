import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from mpi4py import MPI

# Load the CSV files
#mines_df = pd.read_csv('/home/kaiwen1/30123-Project-Kaiwen/congo_existing_mines.csv')

# Try specifying a different encoding
try:
    mines_df = pd.read_csv('/home/kaiwen1/30123-Project-Kaiwen/congo_existing_mines.csv', encoding='latin1')
except UnicodeDecodeError:
    print("UnicodeDecodeError encountered, trying with different encoding.")
    mines_df = pd.read_csv('/home/kaiwen1/30123-Project-Kaiwen/congo_existing_mines.csv', encoding='utf-16', errors='ignore')

# only use a fraction (10%)
# ndvi_loss_data = pd.read_csv('/home/kaiwen1/30123-Project-Kaiwen/sample_full.csv').sample(frac=0.1, random_state=42)
ndvi_loss_data = pd.read_csv('/home/kaiwen1/30123-Project-Kaiwen/sample_full.csv')


# Rename columns for consistency
mines_df = mines_df.rename(columns={
    'longitude': 'Longitude',
    'latitude': 'Latitude'
})

# Convert dataframes to GeoDataFrames
mines_gdf = gpd.GeoDataFrame(
    mines_df, 
    geometry=gpd.points_from_xy(mines_df.Longitude, mines_df.Latitude),
    crs="EPSG:4326"
)

ndvi_loss_df = pd.DataFrame(ndvi_loss_data)
ndvi_loss_gdf = gpd.GeoDataFrame(
    ndvi_loss_df, 
    geometry=gpd.points_from_xy(ndvi_loss_df.Longitude, ndvi_loss_df.Latitude),
    crs="EPSG:4326"
)

# Convert the CRS to EPSG:3857 for distance calculations in meters
mines_gdf = mines_gdf.to_crs(epsg=3857)
ndvi_loss_gdf = ndvi_loss_gdf.to_crs(epsg=3857)

# Buffer radius in meters
radius = 5000  # 5000 meters 

def main():
    """
    Perform parallel processing of NDVI loss locations using MPI.

    Divides the NDVI loss locations among MPI processes, buffers mine locations by a specified radius,
    checks if NDVI loss points are within any of the buffered mine locations, and saves the results to a CSV file.
    """
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide the NDVI loss locations among the processes
    chunk_size = len(ndvi_loss_gdf) // size
    if rank == size - 1:
        chunk = ndvi_loss_gdf.iloc[rank * chunk_size:]
        print(f"Rank {rank}: Processing {len(chunk)} data points (remaining chunk)", flush=True)
    else:
        chunk = ndvi_loss_gdf.iloc[rank * chunk_size:(rank + 1) * chunk_size]
        print(f"Rank {rank}: Processing {len(chunk)} data points", flush=True)

    # Buffer the mine locations by 10 km
    mines_buffered = mines_gdf.buffer(radius)

    # Check if NDVI loss points are within any of the buffered mine locations
    chunk['potential_mining'] = chunk.apply(
        lambda row: mines_buffered.contains(row.geometry).any(), axis=1
    ).astype(int)

    # Gather results from all processes
    gathered_chunks = comm.gather(chunk, root=0)

    if rank == 0:
        # Concatenate all chunks
        result_gdf = pd.concat(gathered_chunks)
        # Save the updated DataFrame to a new CSV file
        result_gdf.to_csv('/home/kaiwen1/30123-Project-Kaiwen/matched_without_distance_5000m.csv', index=False)
        # Display the first few rows of the updated DataFrame
        print(result_gdf.head(), flush=True)
        print("File saved successfully.", flush=True)

if __name__ == "__main__":
    main()
