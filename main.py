import subprocess

# Download datasets
subprocess.call(['sh','./Dataset/DS_Download.sh'])

# Unzip and remove zip folders
subprocess.call(['sh','./Dataset/DS_Unzip.sh'])

