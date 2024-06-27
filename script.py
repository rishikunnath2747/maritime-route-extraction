import pandas as pd
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from itertools import combinations
import math
import json
from sklearn.cluster import DBSCAN


def find_route(start_lat, start_lon, end_lat, end_lon):
    columns_to_load = ['# Timestamp', 'Type of mobile', 'MMSI', 'Latitude', 'Longitude',
        'Navigational status', 'ROT', 'SOG', 'COG']
    points = pd.read_csv('./ais.csv', usecols=columns_to_load)
    df_sorted = points.sort_values(by=['MMSI', '# Timestamp'])

    unique_mmsi_array = points['MMSI'].unique()
    unique_mmsi_list = list(unique_mmsi_array)
    unique_mmsi_values = unique_mmsi_list

    def extract_start_end_points(ship_data):
        result = pd.DataFrame(columns=['MMSI', 'STARTLAT', 'STARTLON', 'ENDLAT', 'ENDLON'])

        # Group by MMSI and aggregate start and end points
        for mmsi, group in ship_data.groupby('MMSI'):
            start_lat = group.iloc[0]['Latitude']
            start_lon = group.iloc[0]['Longitude']
            end_lat = group.iloc[-1]['Latitude']
            end_lon = group.iloc[-1]['Longitude']

            result = result._append({'MMSI': mmsi,
                                    'STARTLAT': start_lat,
                                    'STARTLON': start_lon,
                                    'ENDLAT': end_lat,
                                    'ENDLON': end_lon}, ignore_index=True)

        return result

    startenddataset = extract_start_end_points(df_sorted)

    def find_closest_rows(start_lat, start_lon, end_lat, end_lon, dataset):
        distances = []

        # Calculate distances for each row in the dataset
        for index, row in dataset.iterrows():
            start_dist = ((row['STARTLAT'] - start_lat)**2 + (row['STARTLON'] - start_lon)**2)**0.5
            end_dist = ((row['ENDLAT'] - end_lat)**2 + (row['ENDLON'] - end_lon)**2)**0.5
            total_dist = start_dist + end_dist
            distances.append((index, total_dist))

        # Sort distances in ascending order
        sorted_distances = sorted(distances, key=lambda x: x[1])

        # Get the sorted rows from the dataset
        sorted_rows = [dataset.iloc[index] for index, _ in sorted_distances]

        return sorted_rows

    closest_rows = find_closest_rows(start_lat, start_lon, end_lat, end_lon, startenddataset)
    sorted_dataset = pd.DataFrame(closest_rows)
    first_mmsi_value = int(sorted_dataset['MMSI'].iloc[0])



    def calculate_distance(lat1, lon1, lat2, lon2):
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

    def check_intersection(t_main, t_close):
        # Calculate distances between start and end points of both trajectories
        dist_t_main = calculate_distance(t_main['STARTLAT'], t_main['STARTLON'], t_main['ENDLAT'], t_main['ENDLON'])
        dist_t_close = calculate_distance(t_close['STARTLAT'], t_close['STARTLON'], t_close['ENDLAT'], t_close['ENDLON'])

        # Calculate radius for circles around start and end points of trajectories
        radius_t_main = dist_t_main / 2
        radius_t_close = dist_t_close / 2

        # Calculate center coordinates of circles
        center_t_main = ((t_main['STARTLAT'] + t_main['ENDLAT']) / 2, (t_main['STARTLON'] + t_main['ENDLON']) / 2)
        center_t_close = ((t_close['STARTLAT'] + t_close['ENDLAT']) / 2, (t_close['STARTLON'] + t_close['ENDLON']) / 2)

        # Calculate distance between centers of circles
        dist_centers = calculate_distance(center_t_main[0], center_t_main[1], center_t_close[0], center_t_close[1])

        # Check if circles intersect
        return dist_centers <= (radius_t_main + radius_t_close)
    


    def find_closest_trajectories(t_main, sorted_dataset):
        aoi_list = []
        t_close = t_main
        for index, row in sorted_dataset.iterrows():
            if index == 0:
                continue  # Skip the first row since it's t_main itself

            if check_intersection(t_main, row):
                aoi_list.append(row['MMSI'])
            else:
                break  # Stop checking further trajectories if no intersection is found

        return aoi_list

    t_main = sorted_dataset.iloc[0]
    aoi_list = find_closest_trajectories(t_main, sorted_dataset)
    aoi_list = [int(x) for x in aoi_list]
    print(aoi_list)

    df_aoi = df_sorted[df_sorted['MMSI'].isin(aoi_list)]
    df_aoi = df_aoi.sort_values(by=['MMSI', '# Timestamp'])

    unique_mmsi_array = df_aoi['MMSI'].unique()
    unique_mmsi_list = list(unique_mmsi_array)

    

    finaldf = pd.DataFrame([])

    for i, mmsival in enumerate(unique_mmsi_list, 1):
        #print(f"{i}. Clustering for MMSI: {mmsival}")

        # Step 1: Take all the rows where df['MMSI'] == mmsival and store in a tempdf
        tempdf = df_aoi[df_aoi['MMSI'] == mmsival][['Latitude', 'Longitude','MMSI']]

        # Step 2: Apply DBSCAN to cluster the points
        dbscan = DBSCAN(eps=0.1, min_samples=3)  # Adjust eps and min_samples based on your data
        tempdf['Cluster'] = dbscan.fit_predict(tempdf)

        # Step 3: Select the cluster with the highest number of points
        cluster_counts = tempdf['Cluster'].value_counts()
        dominant_cluster = cluster_counts.idxmax()

        # Step 4: Filter out the points belonging to the dominant cluster
        dominant_cluster_data = tempdf[tempdf['Cluster'] == dominant_cluster]

        # Step 5: Append the dominant cluster data to the final dataframe
        finaldf = pd.concat([finaldf, dominant_cluster_data])

    # 2. Function to Calculate Hausdorff Distance and COG Difference
    def calculate_hausdorff_distance_cog(points_a, cog_a, points_b, cog_b):
        distance_hausdorff = max(
            directed_hausdorff(points_a, points_b)[0],
            directed_hausdorff(points_b, points_a)[0]
        )
        cog_difference = abs(cog_a - cog_b)
        return distance_hausdorff, cog_difference

    # 3. Form DataFrame for Distances
    distance_data = {'MMSI_1': [], 'MMSI_2': [], 'Distance': [], 'COG_Difference': []}

    # Get unique MMSI values
    unique_mmsi_values = df_aoi['MMSI'].unique()

    # Iterate over unique MMSI pairs
    for idx, (mmsi_a, mmsi_b) in enumerate(combinations(unique_mmsi_values, 2)):
        # Extract points and COG values for each trajectory
        data_a = df_aoi[df_aoi['MMSI'] == mmsi_a][['Latitude', 'Longitude', 'COG']].values
        data_b = df_aoi[df_aoi['MMSI'] == mmsi_b][['Latitude', 'Longitude', 'COG']].values

        points_a = data_a[:, :2]
        cog_a = data_a[0, 2]

        points_b = data_b[:, :2]
        cog_b = data_b[0, 2]

        # Calculate Hausdorff distance and COG difference
        distance_hausdorff, cog_difference = calculate_hausdorff_distance_cog(points_a, cog_a, points_b, cog_b)

        # Append data to the distance_data dictionary
        distance_data['MMSI_1'].append(mmsi_a)
        distance_data['MMSI_2'].append(mmsi_b)
        distance_data['Distance'].append(distance_hausdorff)
        distance_data['COG_Difference'].append(cog_difference)

        # Print statement for every 100 iterations
        if idx % 100 == 0:
            remaining_pairs = len(list(combinations(unique_mmsi_values, 2))) - len(distance_data['MMSI_1'])


    distance_df = pd.DataFrame(distance_data)



    def dbscan(data, eps, min_samples):
        """
        Performs DBSCAN clustering on the given data.

        Args:
            data: The data points to be clustered.
            eps: The maximum distance between two points to consider them neighbors.
            min_samples: The minimum number of neighbors required for a point to be considered a core point.

        Returns:
            A list of cluster labels, where the label for each point is the index of its cluster.
        """

        clusters = []
        core_points=[]
        visited = set()

        for point in data:
            if point in visited:
                continue

            if is_core_point(point, data, eps, min_samples):
                core_points.append(point)
                cluster = expand_cluster(point, data, eps, min_samples, visited)
                clusters.append(cluster)
            else:
                visited.add(point)
                clusters.append(point)  # Mark point as noise
        return clusters


    def is_core_point(point, data, eps, min_samples):
        """
        Checks if a point is a core point.

        Args:
            point: The point to check.
            data: The data points to be clustered.
            eps: The maximum distance between two points to consider them neighbors.
            min_samples: The minimum number of neighbors required for a point to be considered a core point.

        Returns:
            True if the point is a core point, False otherwise.
        """

        neighbors = get_neighbors(point, data, eps)
        return len(neighbors) >= min_samples


    def expand_cluster(point, data, eps, min_samples, visited):
        """
        Expands a cluster by recursively adding neighboring points.

        Args:
            point: The starting point of the cluster.
            data: The data points to be clustered.
            eps: The maximum distance between two points to consider them neighbors.
            min_samples: The minimum number of neighbors required for a point to be considered a core point.
            visited: A set of visited points to avoid duplicates.

        Returns:
            A list of points belonging to the expanded cluster.
        """

        cluster = [point]
        visited.add(point)

        for neighbor in get_neighbors(point, data, eps):
            if neighbor not in visited:
                cluster.extend(expand_cluster(neighbor, data, eps, min_samples, visited))
                visited.add(neighbor)

        return cluster


    def get_neighbors(point, data, eps,max_cog_difference=45):
        """
        Gets the neighbors of a point within a given distance.

        Args:
            point: The point to find neighbors for.
            data: The data points to be clustered.
            eps: The maximum distance between two points to consider them neighbors.

        Returns:
            A list of neighbors within the given distance.
        """

        neighbors = []

        for other_point in data:
            if point != other_point and distance(point, other_point) <= eps and cog_distance(point, other_point, max_cog_difference):
                neighbors.append(other_point)

        return neighbors


    def distance(point1, point2):
        """
        Calculates the distance between two points.

        (Assuming Euclidean distance in this example)

        Args:
            point1: The first point.
            point2: The second point.

        Returns:
            The distance between the two points.
        """
        distance_row = distance_df.loc[(distance_df['MMSI_1'] == point1) & (distance_df['MMSI_2'] == point2), 'Distance']
        if not distance_row.empty:
            distance = distance_row.values[0]
            return distance
        else:
            # Check the reversed order
            distance_row_reversed = distance_df.loc[(distance_df['MMSI_1'] == point2) & (distance_df['MMSI_2'] == point1), 'Distance']
            if not distance_row_reversed.empty:
                distance_reversed = distance_row_reversed.values[0]
                return distance_reversed
            else:
                return None

    def cog_distance(point1, point2,max_cog_difference=45):
        """
        Calculates the COG distance between two points.

        Args:
            point1: The first point.
            point2: The second point.

        Returns:
            The COG distance between the two points.
        """
        cog_row = distance_df.loc[(distance_df['MMSI_1'] == point1) & (distance_df['MMSI_2'] == point2), 'COG_Difference']
        if not cog_row.empty:
            cog_value = cog_row.values[0]
            return cog_value is None or cog_value <= max_cog_difference
        else:
            # Check the reversed order
            cog_row_reversed = distance_df.loc[(distance_df['MMSI_1'] == point2) & (distance_df['MMSI_2'] == point1), 'COG_Difference']
            if not cog_row_reversed.empty:
                cog_value_reversed = cog_row_reversed.values[0]
                return cog_value_reversed is None or cog_value_reversed <= max_cog_difference
            else:
                return None
    clusters = dbscan(aoi_list,1,2)
    print(clusters)

    flag=0
    final_cluster = []
    
    for sublist in clusters:
        if (type(sublist) == list):
            if first_mmsi_value in sublist:
                final_cluster.append(sublist)
                flag=1
    if flag == 0:
        coordinates = df_sorted[df_sorted['MMSI'] == first_mmsi_value][['Latitude', 'Longitude']].values.tolist()
        json_coordinates = json.dumps(coordinates)
        #print(json_coordinates)
        return json_coordinates
    else:
        # Extract coordinates for all MMSIs in final_cluster
        coordinates_list = []
        final_cluster=final_cluster[0]
        for mmsi_value in final_cluster:
            coordinates = df_sorted[df_sorted['MMSI'] == mmsi_value][['Latitude', 'Longitude']].values.tolist()
            coordinates_list.extend(coordinates)

        json_coordinates = json.dumps(coordinates_list)
        return json_coordinates


    return clusters

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5:
        lat1, lon1, lat2, lon2 = map(float, sys.argv[1:])
        clusters = find_route(lat1, lon1, lat2, lon2)
        #print(clusters)
        sys.stdout.flush()

