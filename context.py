"""
Context file for the project.
"""
import os
import getpass

fsps_dir = "/home/kadu/repos/fsps"
os.environ["SPS_HOME"] = fsps_dir

# Add home directories in different computers
if getpass.getuser() == "kadu":
    home_dir = "/home/kadu/Dropbox/INSPIRE"

data_dir = os.path.join(home_dir, "data")

