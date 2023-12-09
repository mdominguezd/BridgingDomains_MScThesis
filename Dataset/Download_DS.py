# from urllib import request


# def download_DS(urls, paths):
#     """
#         Simple function to download the Datasets stored as zip files
#     """

#     if len(urls) != len(paths):
#         raise Exception("Length of urls and paths must be the same.")

#     for i in range(len(urls)):
#         u = request.urlopen(urls[i])
#         data = u.read()
#         u.close()
         
#         with open(paths[i], "wb") as f :
#             f.write(data)

import requests

def download_DS(urls, paths):
    if len(urls) != len(paths):
        raise Exception("Length of urls and paths must be the same.")

    for i in range(len(urls)):
        # Modify the shared link to get the direct download link
        direct_download_link = urls[i].replace("www.dropbox.com", "dl.dropboxusercontent.com").split('?')[0]
    
        # Download the file using the direct download link
        with requests.get(direct_download_link, stream=True) as response:
            with open(paths[i], 'wb') as local_file:
                for chunk in response.iter_content(chunk_size=8192):
                    local_file.write(chunk)
        # request.urlretrieve(urls[i], paths[i])
        

