import numpy as np
from math import radians, sin, cos, sqrt, asin

def dist_on_sphere(pos0, pos1):
    '''
    distance based on Haversine formula
    pos: [latitude, longitude]
    output: distance(km)
    Ref: https://en.wikipedia.org/wiki/Haversine_formula
    '''
    radius = 6378.137
    latang1, lngang1 = pos0
    latang2, lngang2 = pos1
    phi1, phi2 = radians(latang1), radians(latang2)
    lam1, lam2 = radians(lngang1), radians(lngang2)
    term1 = sin((phi2 - phi1) / 2.0) ** 2
    term2 = sin((lam2 - lam1) / 2.0) ** 2
    term2 = cos(phi1) * cos(phi2) * term2
    wrk = sqrt(term1 + term2)
    wrk = 2.0 * radius * asin(wrk)
    return wrk

MESH_INFOS = [
    None,
    {"length": 80000, "parent": 1, "ratio": 1, "lat": 1 / 1.5, "lon": 1},
    {"length": 10000, "parent": 1, "ratio": 8, "lat": 5 / 60, "lon": 7.5 / 60},
    {"length": 1000, "parent": 2, "ratio": 10, "lat": 30 / 3600, "lon": 45 / 3600},
    {"length": 500, "parent": 3, "ratio": 2, "lat": 15 / 3600, "lon": 22.5 / 3600},
    {"length": 250,"parent": 4, "ratio": 2, "lat": 7.5 / 3600, "lon": 11.25 / 3600},
    {"length": 125,"parent": 5, "ratio": 2, "lat": 3.75 / 3600, "lon": 5.625 / 3600},
    {"length": 100,"parent": 3, "ratio": 10, "lat": 3 / 3600, "lon": 4.5 / 3600},
    {"length": 50,"parent": 7, "ratio": 2, "lat": 1.5 / 3600, "lon": 2.25 / 3600},
    {"length": 25,"parent": 7, "ratio": 10, "lat": 0.75 / 3600, "lon": 1.125 / 3600}
]

class MeshCodeUtility:

    @staticmethod
    def get_meshcode(lat, lon, m):
        """
        Get meshcode for the given latitude, longitude, and mesh level.
        """
        if m in ["80km", 1]:
            return MeshCodeUtility._handle_80km_mesh(lat, lon)
        elif m in ["10km", 2]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 1, 5, 7.5)
        elif m in ["1km", 3]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 2, 30, 45)
        elif m in ["500m", 4]:
            return MeshCodeUtility._handle_special_mesh(lat, lon, 3, 15, 22.5)
        elif m in ["250m", 5]:
            return MeshCodeUtility._handle_special_mesh(lat, lon, 4, 7.5, 11.25)
        elif m in ["125m", 6]:
            return MeshCodeUtility._handle_special_mesh(lat, lon, 5, 3.75, 5.625)
        elif m in ["100m", 7]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 3, 3, 4.5)
        elif m in ["50m", 8]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 7, 1.5, 2.25)
        elif m in ["25m", 9]:
            return MeshCodeUtility._handle_interval_mesh(lat, lon, 8, 0.75, 1.125)
        else:
            raise ValueError('Invalid mesh degree')
    
    @staticmethod
    def _handle_80km_mesh(lat, lon):
        """
        Handle the generation of an 80km mesh code from given latitude and longitude.

        Parameters:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

        Returns:
        dict: A dictionary containing the mesh code, and the remaining latitude and longitude
              after calculating the 80km mesh.
        """
        lat_base, dest_lat = divmod(lat * 60, 40)
        lat_base = int(lat_base)
        lon_base = int(lon - 100)
        dest_lon = lon - 100 - lon_base
        return {
            "mesh_code": f"{int(lat_base):02d}{int(lon_base):02d}",
            "lat": dest_lat,
            "lon": dest_lon
        }

    @staticmethod
    def _handle_interval_mesh(lat, lon, parent_degree, lat_interval, lon_interval):
        """
        Handle the generation of a mesh code for specified intervals from given latitude and longitude.

        Parameters:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        parent_degree (int): The parent mesh level.
        lat_interval (float): The interval for latitude in the mesh.
        lon_interval (float): The interval for longitude in the mesh.

        Returns:
        dict: A dictionary containing the mesh code, and the remaining latitude and longitude
              after calculating the mesh for the specified intervals.
        """
        base_data = MeshCodeUtility.get_meshcode(lat, lon, parent_degree)
        if parent_degree == 1:
            base_data["lon"] *= 60
            
        if parent_degree == 2:
            base_data["lat"] *= 60
            base_data["lon"] *= 60

        left_operator, dest_lat = divmod(
            base_data["lat"], lat_interval)
        right_operator, dest_lon = divmod(
            base_data["lon"], lon_interval)
        return {
            "mesh_code": f"{base_data['mesh_code']}{int(left_operator):01d}{int(right_operator):01d}",
            "lat": dest_lat,
            "lon": dest_lon
        }

    @staticmethod
    def _handle_special_mesh(lat, lon, parent_degree, lat_interval, lon_interval):
        """
        Handle the generation of a special mesh code based on a specific pattern from given latitude and longitude.

        Parameters:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        parent_degree (int): The parent mesh level.
        lat_interval (float): The interval for latitude in the mesh.
        lon_interval (float): The interval for longitude in the mesh.

        Returns:
        dict: A dictionary containing the mesh code, and the remaining latitude and longitude
              after calculating the special mesh.
        """
        base_data = MeshCodeUtility.get_meshcode(lat, lon, parent_degree)
        left_index, dest_lat = divmod(base_data["lat"], lat_interval)
        right_index, dest_lon = divmod(
            base_data["lon"], lon_interval)
        operator = int(2 * left_index + right_index) + 1
        return {
            "mesh_code": f"{base_data['mesh_code']}{int(operator):01d}",
            "lat": dest_lat,
            "lon": dest_lon
        }


    @staticmethod
    def get_mesh_coords(meshcode, m):
        """
        Retrieve the coordinate data for a given mesh code and mesh level.

        Parameters:
        meshcode (str): The mesh code.
        m (int): Mesh level.

        Returns:
        dict: A dictionary containing the mesh code, degree, southwest latitude/longitude, 
                center latitude/longitude, and geometry (coordinates of the mesh's corners).
        """
        coords_data = {
            "mesh_code": meshcode,
            "degree": m,
            "south_west_latlon": MeshCodeUtility.get_mesh_latlon(meshcode, m),
            "center_latlon": MeshCodeUtility.get_mesh_center_latlon(meshcode, m),
            "geometry": MeshCodeUtility.get_mesh_geometry(meshcode, m),
        }
        coords_data["latlons"] = [(lat, lon)
                                    for lon, lat in coords_data["geometry"][0][:-1]]
        return coords_data

    @staticmethod
    def get_mesh_latlon(meshcode, m):
        """
        Calculate the southwest corner latitude and longitude for a given mesh code and mesh level.

        Parameters:
        meshcode (str): The mesh code.
        m (int): Mesh level.

        Returns:
        tuple: Latitude and longitude of the southwest corner.
        """
        if m == 1:
            lat = int(meshcode[0:2]) / 1.5
            lon = int(meshcode[2:4]) + 100
            return lat, lon
        if m in [2, 3, 7, 8, 9]:
            lat, lon = MeshCodeUtility.get_mesh_latlon(
                meshcode[:-2], MESH_INFOS[m]["parent"])
            lat += int(meshcode[-2]) * MESH_INFOS[m]["lat"]
            lon += int(meshcode[-1]) * MESH_INFOS[m]["lon"]
            return lat, lon
        elif m in [4, 5, 6]:
            lat, lon = MeshCodeUtility.get_mesh_latlon(
                meshcode[:-1], MESH_INFOS[m]["parent"])
            lat += ((int(meshcode[-1]) - 1) // 2) * MESH_INFOS[m]["lat"]
            lon += ((int(meshcode[-1]) - 1) % 2) * MESH_INFOS[m]["lon"]
            return lat, lon
        else:
            raise ValueError('Invalid mesh degree')

    @staticmethod
    def get_mesh_center_latlon(meshcode, m):
        """
        Calculate the center latitude and longitude for a given mesh code and mesh level.

        Parameters:
        meshcode (str): The mesh code.
        m (int): Mesh level.

        Returns:
        tuple: Latitude and longitude of the center.
        """
        lat, lon = MeshCodeUtility.get_mesh_latlon(meshcode, m)
        lat_center = lat + MESH_INFOS[m]["lat"]/2
        lon_center = lon + MESH_INFOS[m]["lon"]/2
        return lat_center, lon_center

    @staticmethod
    def get_mesh_geometry(meshcode, m):
        """
        Retrieve the geometry (coordinates of the mesh's corners) for a given mesh code and mesh level.

        Parameters:
        meshcode (str): The mesh code.
        m (int): Mesh level.

        Returns:
        list: A list of lists containing the coordinates of the mesh's corners.
        """
        lat, lon = MeshCodeUtility.get_mesh_latlon(meshcode, m)
        return [[[lon, lat],
                 [lon, lat + MESH_INFOS[m]["lat"]],
                 [lon + MESH_INFOS[m]["lon"], lat + MESH_INFOS[m]["lat"]],
                 [lon + MESH_INFOS[m]["lon"], lat],
                 [lon, lat]]]