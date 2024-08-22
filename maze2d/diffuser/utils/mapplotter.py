import csv
from shapely.geometry import Point
from shapely.geometry import Polygon,MultiPolygon
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import cv2
import sys
from PIL import Image
import numpy as np
import yaml
import matplotlib.image as mpimg

class MapConverter():
    def __init__(self, map_dir, name_env, threshold=105):
        
        self.threshold = threshold
        self.map_dir = map_dir
        self.name_env = name_env
        self.vertices = []
        self.map_callback()
        if 'rls' in name_env.lower():
            ext = [(0, 0), (0, 4), (6, 4), (6, 6),(12,6),(12,0)]
        elif 'ssi' in name_env.lower():
            ext = [(-4.9, -0.239), (-0.85, -5.25), (6.13, -5.13), (6.87, -2.45),(3.35,-0.2),(5.12,3.22), (0.47, 5.81), (4.15, 12.5), (2.30,12.5)]
        polygon = Polygon(ext)
        polygons = MultiPolygon(self.vertices)
        self.polydiff = polygon
        for poly in polygons.geoms:
            self.polydiff = self.polydiff.difference(poly)

    def plot_all(self):
        """
            plot all trajectories of one file for test and visualization
        """
        plt.clf()
        mply = gpd.GeoSeries([self.polydiff])
        mply.plot()
        fig = plt.gcf()
        x = []
        y = []
        # with open('datasets/2024-01-24-14-14-58-robot_pose.csv') as csv_file:  # rls
        with open('datasets/ssi_130k.csv') as csv_file:  # ssi
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    x.append(float(row[".position.x"]))
                    y.append(float(row[".position.y"]))
        plt.plot(x,y, color = "#FF0000")
        # plt.savefig(f'map.pdf', bbox_inches='tight')
        return fig

    def plot(self, x, y):
        plt.clf()
        mply = gpd.GeoSeries([self.polydiff])
        mply.plot()
        fig = plt.gcf()

        path_length = len(x)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.plot(x, y, c='black', zorder=10)
        plt.scatter(x, y, c=colors, zorder=20)

        return fig

    def map_callback(self):
        
        map_array = cv2.imread(self.map_dir)
        map_array = cv2.flip(map_array, 0)
        print(f'loading map file: {self.map_dir}')
        try:
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)
        except cv2.error as err:
            print(err, "Conversion failed: Invalid image input, please check your file path")    
            sys.exit()
        info_dir = self.map_dir.replace('pgm','yaml')

        with open(info_dir, 'r') as stream:
            map_info = yaml.load(stream, Loader=yaml.FullLoader) 
        
        # set all -1 (unknown) values to 0 (unoccupied)
        map_array[map_array < 0] = 0
        contours = self.get_occupied_regions(map_array)
        print('Processing...')
        self.vertices = [self.contour_to_verts(c, map_info) for c in contours]

    def get_occupied_regions(self, map_array):
        """
        Get occupied regions of map
        """
        map_array = map_array.astype(np.uint8)
        _, thresh_map = cv2.threshold(
                map_array, self.threshold, 100, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
                thresh_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0]
        output_contours = []
        for idx, contour in enumerate(contours):
            output_contours.append(contour) if 0 not in contour else print('Remove image boundary')
            
        return output_contours

    def contour_to_verts(self, contour, metadata):
        for point in contour:
            x, y = point[0]
            vertices = []
            if 'rls' in self.name_env.lower():
                new_vertices = Polygon([self.coords_to_loc((x - 10, y - 10), metadata),self.coords_to_loc((x + 10, y - 10), metadata),self.coords_to_loc((x+ 10, y + 10), metadata),self.coords_to_loc((x - 10, y + 10), metadata)])
            elif 'ssi' in self.name_env.lower():
                new_vertices = Polygon([self.coords_to_loc((x - 7, y - 7), metadata),self.coords_to_loc((x + 7, y - 7), metadata),self.coords_to_loc((x+ 7, y + 7), metadata),self.coords_to_loc((x - 7, y + 7), metadata)])
            vertices.append(new_vertices)
        return vertices

    def coords_to_loc(self,coords, metadata):
        x, y = coords
        loc_x = x * metadata['resolution'] + metadata['origin'][0]
        loc_y = y * metadata['resolution'] + metadata['origin'][1]
        # TODO: transform (x*res, y*res, 0.0) by Pose map_metadata.origin
        # instead of assuming origin is at z=0 with no rotation wrt map frame
        return (loc_x, loc_y)


if __name__ == "__main__":
    # map = MapConverter('datasets/rls_wbmp.pgm')
    map = MapConverter('datasets/ssi_map.pgm', threshold=10)
    map.plot_all()