# Quickstart Guide

Here's what you need to make tile mosaic images ASAP:

![](/dat/images/tile_mosaic_example.jpg)

## Steps

1. Make sure you satisfy the [Software Requirements](#software-requirements).
2. Clone the repository to a local directory on your machine.
3. Put your desired image to print into `image/Run/` and make sure no other images are present.
4. Use Python to run the script called `create_tile_mosaic.py`, adjusting any arguments as desired.
5. Follow the [In-Game](#in-game) steps.


## Software Requirements


* python 3.12
* python libraries:
  * os, base64, json, zlib, enum (included with python)
  * numpy
  * pandas
  * luadata
  * pillow (PIL fork)
* Factorio 2.0
* Factorio Mods:
  * Technicolor Lab Tiles [LINK](https://mods.factorio.com/mod/tech-tiles)



## In-Game

1. Open Factorio with Technicolor Lab Tiles loaded. In startup settings, make sure you are set to 125 unique tiles (this is the default).

2. Alt+tab to `Blueprint Out/`, and drag the `.txt` file directly into the map. It may take a few seconds, but then the game should say 'Blueprint imported successfully' and you will be holding the blueprint. Click a spot in your quickbar to save the blueprint there and make Factorio the active screen. Note: you may need to place the game in windowed mode to do this, especially on a single screen.
