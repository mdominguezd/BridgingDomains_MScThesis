import geopandas as gpd
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import contextily as cnx
import seaborn as sns

def split_data(gdf, seed, clusters = 10, splits = [50,25,25]):
    """
        Function to split both datasets into train, validation and test spatially. The function uses K-Means clustering with x, y coordinates to split the data in groups of polygons roughly as (60% Training, 20% Validation and 20% Test.)

        Input:
            - gdf: GeoDataFrame with the complete dataset.
            - **seed:** (int) Seed to ensure replicabilty of kmeans clustering.
            - **cluster:*** (int) Number of clusters to get from K-means algorithm
            - **splits:** (list) List with values of approximate percentage wanted to get per split. Order goes \[train, validation, test\].

        Output:
            - gdf: GeoDataFrame with an additional feature ('split') with the split to which the polygon belongs (Train, Validation or Test).
    """
    gdf.geometry = gdf.make_valid()

    # Get centroids 
    centroids = pd.DataFrame([gdf.centroid.x, gdf.centroid.y]).T

    # Get the clusters with K-Means.
    kmeans = KMeans(clusters, random_state = seed)
        
    labels = kmeans.fit_predict(centroids)
    
    clusters = np.max(labels) + 1
    
    gdf['split'] = 'train' # Roughly 60% of the data
    gdf['split'][labels > clusters//(100/splits[0])] = 'validation' # Roughly 20% of the data
    gdf['split'][labels > clusters//(100/(splits[0] + splits[1]))] = 'test' # Roughly 20% of the data

    return gdf


def plot_splits(gdf, filename = '', title = 'Title', kde = False):
    """
        Function to plot the splits created using k-means clustering.

        Inputs:
            - gdf: (geopandas.GeoDataFrame) DataFrame with the data splitted in train, validation and test.
            - filename: (str) Name of the output file where the plot will be saved.
            - title: (str) Title to be shown in the figure.
            - kde: (boolean) Boolean to determine if kernel density of the polygons will be plotted or not.

        Outputs:
    """
    
    fig, ax = plt.subplots(1,1, figsize = (15,15))

    col = {'train' : 'red', 'validation' : 'blue', 'test' : 'green'}
    
    gdf.plot(color = [col[i] for i in gdf['split']],
             legend = True, 
             ax = ax)
    
    legend_elements = [Patch(facecolor='red', label='Color Patch'),
                       Patch(facecolor='blue', label='Color Patch'),
                       Patch(facecolor='green', label='Color Patch')]

    if kde:
        sns.kdeplot(x = gdf.centroid.x, y = gdf.centroid.y,
            hue = gdf.split, thresh = 0.05, 
            fill = True, levels = 40, 
            palette = ['red', 'blue', 'green'])
    
    ax.legend(legend_elements, 
              ['train', 'validation', 'test'], 
              fontsize = 12,
              title  = 'split',
              loc = 'upper right')
    
    cnx.add_basemap(ax = ax, 
                    crs = gdf.crs, 
                    source = cnx.providers.Esri.WorldStreetMap)

    ax.set_title(title)
    
    if len(filename) != 0:
        fig.savefig(filename)