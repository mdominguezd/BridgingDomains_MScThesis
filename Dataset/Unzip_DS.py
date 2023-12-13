from zipfile import ZipFile
import os 

def UnzipFolders(domain):
    """
        Function to unzip the three folds for each dataset.

        Input:
            - domain: string with the name of the domain. Options: "Tanzania" or "IvoryCoast"
    """
    if (domain != "Tanzania") & (domain != "IvoryCoast"):
        raise Exception("domain needs to be Tanzania or IvoryCoast")

    if len([i for i in os.listdir('.') if '.zip' in i]) != 0: 
        for i in range(1): # NEEEDS TO BE CHANGED
            with ZipFile(domain + str(i+1) + ".zip", 'r') as zipped:
                zipped.extractall(path="./")
            os.remove(domain + str(i+1) + ".zip")