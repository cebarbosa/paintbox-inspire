"""
Context file for the project.
"""
import os
import getpass
import platform

import astropy.units as u
import matplotlib.pyplot as plt

fsps_dir = "/home/kadu/repos/fsps"
os.environ["SPS_HOME"] = fsps_dir

# Add home directories in different computers
if getpass.getuser() == "kadu":
    home_dir = "/home/kadu/Dropbox/INSPIRE"
elif platform.node() in ["uv100", "alphacrucis", "yaci.iag.usp.br"]:
    home_dir = "/sto/home/cebarbosa/INSPIRE"

data_dir = os.path.join(home_dir, "data")

# Matplotlib settings
plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# set tick width
width = 0.5
majsize = 4
minsize = 2
plt.rcParams['xtick.major.size'] = majsize
plt.rcParams['xtick.major.width'] = width
plt.rcParams['xtick.minor.size'] = minsize
plt.rcParams['xtick.minor.width'] = width
plt.rcParams['ytick.major.size'] = majsize
plt.rcParams['ytick.major.width'] = width
plt.rcParams['ytick.minor.size'] = minsize
plt.rcParams['ytick.minor.width'] = width
plt.rcParams['axes.linewidth'] = width

colwidth = 3.54399 # inches - A&A template
txtwidth = 7.24551

flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz

