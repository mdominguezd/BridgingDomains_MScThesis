import ee
import geemap
import logging
import multiprocessing
import os
import requests
import shutil
from retry import retry
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

def perform_KMeans(geo_fn, chunks, get_iter = 0, split = 'train'):
    """
        Function to divide a geojson using the K-Means clustering algorithm and return one of the clusters.

        Inputs: 
            - geo_fn: (str) Filename of the geojson
            - chunks: (int) Number of chunks in which the geojson will be divided.
            - get_iter: (int) Number of the chunk that will be returned by the function.
            - split: (str) Split of the dataset. It can be either train, validation or test.

        Output:
            - gdf: (geopandas.GeoDataFrame) GeoDataFrame subset of the original geojson.
            
    """
    gdf = gpd.read_file(geo_fn)

    gdf = gdf[gdf['split'] == split]
    
    kmeans = KMeans(chunks, random_state = 10)
    centroids = pd.DataFrame([gdf.centroid.x, gdf.centroid.y]).T
    labels = kmeans.fit_predict(centroids)
    
    gdf = gdf[labels == get_iter]

    gdf['cashew'] = 1

    return gdf

def get_img_n_bounds(gdf):
    """
        Function to get the ee.Image and the area of interest. These are the ones that will be downloaded in smaller patches.

        Inputs:
            - gdf: (geopandas.GeoDataFrame) GeoDataFrame subset of the original geojson.

        Outputs:
            - image: (ee.Image) Image that will be downloaded in smaller patches.
            - bound: (ee.Geometry) Region of the image that will be downloaded. This geometry will be used to get the central points of the patches.
    """

    params = pd.read_table('Track_params.txt', delimiter = ', ', engine='python')
    
    for i in range(len(params)):
        globals()[params['Param'].iloc[i]] = params['value'].iloc[i]
    
    Pl = (ee.ImageCollection(platform)
          .filterDate(date_beg, date_end)
          .select(['B','G','R', 'N'])
          )
    
    bound = ee.Geometry.BBox(*tuple(gdf.dissolve().bounds.iloc[0])).buffer(2000)
    
    images = Pl.map(lambda img : img.clip(bound))

    proj = images.first().projection()
    
    image = images.median().reproject(proj.getInfo()['crs'], proj.getInfo()['transform'])

    return image, bound

def append_label(img, gdf):
    """
        Function to append the labels (Ground truths) as an additional band to the image that will be downloaded.

        Inputs:
            - img: (ee.Image) Image gotten from GEE that will be downloaded. 
            - gdf: (geopandas.GeoDataFrame) Geodataframe with the labels that will be appended.

        Outputs:
            - image_with_labels: (ee.Image) Image with an extra band in which the labels have been appended.
    """
    feat = geemap.geopandas_to_ee(gdf)

    rasterized = feat.reduceToImage(['cashew'], ee.Reducer.max())

    reprojected = rasterized.reproject(img.projection().getInfo()['crs'], img.projection().getInfo()['transform'])

    image_with_labels = img.addBands(reprojected)
    
    return image_with_labels
    
def get_point_distance(overlap = 0.5):
    """
        Function to get distance between centroids of patches that will be downloaded. This value will depend on the percentage of overlap desired.

        Inputs:
            - overlap: (float) Percentage of overlap (Number between 0 and 1)

        Outputs:
            - distance: (float) Distance between centroids of patches.
            
    """
    
    distance = 4.77*256*(1-overlap)
    
    return distance

############################################################################################################
############################################################################################################
############################################################################################################
geo_fn = '../Data/Vector/Cashew_Polygons_CIV_splitted_KM_fold_1.geojson'
split = 'train'

init_ind = 0

if 'TNZ' in geo_fn:
    domain = 'Tanzania'
else:
    domain = 'IvoryCoast'

fold = geo_fn.split('.')[-2][-1]

gdf = perform_KMeans(geo_fn, 4, 0, split)

feat = geemap.geopandas_to_ee(gdf)

image, region = get_img_n_bounds(gdf)

image = append_label(image, gdf)

distance = get_point_distance(0.8)

params = {
'buffer': (4.77*256)//2,  # The buffer distance (m) around each point
'scale': 100,  # The scale to do stratified sampling
'seed': 1,  # A randomization seed to use for subsampling.
'dimensions': '256x256',  # The dimension of each image chip
'format': "GEO_TIFF",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
'prefix': domain+'_'+split+'_imgs_',  # The filename prefix
'processes': 16,  # How many processes to used for parallel processing
'out_dir': 'Chips/'+domain+'/fold'+fold+'/'+split,  # The output directory. Default to the current working directly
}

############################################################################################################
############################################################################################################
############################################################################################################

def getRequests():
    """
        Function to get the centroids of all of the patches that will be downloaded.

        Inputs:
            -

        Outputs:
            - point: (dict) Dictionary with the coordinates of the centroids
    """
    proj = image.select([0]).projection()

    latlon = ee.Image.pixelLonLat()
    
    coords = latlon.select(['longitude', 'latitude']).reduceRegion(ee.Reducer.toList(), region, distance)
    lat = ee.List(coords.get('latitude'))
    lon = ee.List(coords.get('longitude'))
    
    point_list = lon.zip(lat)
    points = ee.FeatureCollection(point_list.map(lambda point : ee.Feature(ee.Geometry.Point(point))))

    buffs = points.map(lambda point : point.buffer(params['buffer'])/1.5).filterBounds(feat)

    points = points.filterBounds(buffs)
    points = points.aggregate_array('.geo').getInfo()
    
    return points

@retry(tries=10, delay=1, backoff=2)
def getResult(index, point):
    point = ee.Geometry.Point(point['coordinates'])
    region = point.buffer(params['buffer']).bounds()

    if params['format'] in ['png', 'jpg']:
        url = image.getThumbURL(
            {
                'region': region,
                'dimensions': params['dimensions'],
                'format': params['format'],
            }
        )
    else:
        url = image.getDownloadURL(
            {
                'region': region,
                'dimensions': params['dimensions'],
                'format': params['format'],
            }
        )

    if params['format'] == "GEO_TIFF":
        ext = 'tif'
    else:
        ext = params['format']

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    out_dir = os.path.abspath(params['out_dir'])
    basename = str(index+init_ind).zfill(5)
    filename = f"{out_dir}/{params['prefix']}{basename}.{ext}"
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", basename)

if __name__ == '__main__':
    logging.basicConfig()
    items = getRequests()
    
    pool = multiprocessing.Pool(params['processes'])
    pool.starmap(getResult, enumerate(items))
    
    pool.close()