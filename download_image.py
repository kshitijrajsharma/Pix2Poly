import asyncio
import os

from geomltoolkits.downloader import tms as TMSDownloader

# Define area of interest
ZOOM = 18
WORK_DIR = "banepa"
TMS = "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
BBOX = [85.514668, 27.628367, 85.528875, 27.638514]

# Create working directory
os.makedirs(WORK_DIR, exist_ok=True)

# Download tiles
asyncio.run(
    TMSDownloader.download_tiles(
        tms=TMS,
        zoom=ZOOM,
        out=WORK_DIR,
        bbox=BBOX,
        georeference=True,
        dump=True,
        prefix="OAM",
    )
)
