from zipfile import ZipFile
import os 
import warnings

def UnzipFolders(domain):
    """
        Function to unzip the three folds for each dataset.

        Input:
            - domain: string with the name of the domain. Options: "Tanzania" or "IvoryCoast"
    """
    if (domain != "Tanzania") & (domain != "IvoryCoast"):
        raise Exception("Domain needs to be Tanzania or IvoryCoast")

    if len([i for i in os.listdir('.') if '.zip' in i]) != 0: 
        for i in range(3): 
            if (len([f for f in os.listdir('.') if domain + str(i+1) + ".zip" in f]) != 0):
                with ZipFile(domain + str(i+1) + ".zip", 'r') as zipped:
                    zipped.extractall(path="./")
                os.remove(domain + str(i+1) + ".zip")
            else:
                raise warnings.warn("No zipped file found (" + domain + str(i+1) + ".zip" + ")")