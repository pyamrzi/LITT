# This script extracts vessel masks iteratively from OMERO api and saves them as a pkl file
# Made by M. Pouya Mirzaei for the LaViolette Lab

# INSTALL DEPENDENCIES
print("Installing dependencies...")
import numpy as np
import math
import sys
from omero.gateway import BlitzGateway
import getpass
from littFunctions import create_filled_polygon_mask_skimage, get_combined_shape_points
from tqdm import tqdm
import pickle
import os

# CONNECT TO OMERO
print("Connecting to OMERO:")
username = input("Enter your OMERO username: ")
password = getpass.getpass("Enter your OMERO password: ")
conn = BlitzGateway(username, password, host="wss://wsi.lavlab.mcw.edu/omero-wss", port=443, secure=True)
conn.connect()
conn.c.enableKeepAlive(30)

if conn.connect():
    conn.c.enableKeepAlive(30)
    print("Connection successful")
    print("Current user:")
    user = conn.getUser()
    print("   ID:", user.getId())
    print("   Username:", user.getName())
    print("   Full Name:", user.getFullName())
else:
    print("Connection failed")

del user, username, password  # clean memory

# EXTRACT DATA FROM MASK AND CSV

## Initialize variables 
combmasks = []

## Set parameters
SOX2_image_id = 4113
SOX2_image = conn.getObject("Image", SOX2_image_id)

ds = 1
vessel_color = (255, 16, 0) # RGB color for SOX2
vessel_out_color = (255, 122, 0) # RGB color for vessel out

combmasks.extend(get_combined_shape_points(SOX2_image, vessel_color, ds, name="Vessel In"))
combmasks.extend(get_combined_shape_points(SOX2_image, vessel_out_color, ds, name="Vessel Out"))

os.makedirs("data", exist_ok=True)

print("Saving combined masks as a pickle file...")
with open("data/vessel_masks.pkl", "wb") as f:
    pickle.dump(combmasks, f)
print("Combined masks saved successfully.")
