from urllib import request


def download_DS(urls, paths):
    """
        Simple function to download the Datasets stored as zip files
    """

    if len(urls) != len(paths):
        raise Exception("Length of urls and paths must be the same.")

    for i in range(len(urls)):
        u = request.urlopen(urls[i])
        data = u.read()
        u.close()
         
        with open(paths[i], "wb") as f :
            f.write(data)
            
        # request.urlretrieve(urls[i], paths[i])
        

