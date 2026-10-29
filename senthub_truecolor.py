import datetime
import pytz
import os
import getpass
from datetime import date
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
# import s3fs
# import botocore.session
import boto3

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from datetime import datetime
from dateutil.relativedelta import relativedelta


from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    # MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)


def find_folder_with_tiff(base_directory="/tmp/"):
    for folder, _, filenames in os.walk(base_directory):
        for filename in filenames:
            if filename.endswith('.tiff'):
                folder = folder.split("/")[-1]
                return folder
    return None  # Return None if no folder with a .tiff file is found




# Initialize a session using the profile
session = boto3.Session(profile_name='personal')
s3_client = session.client('s3')

# Retrieve credentials
credentials = session.get_credentials()
AWS_SECRET = credentials.secret_key
AWS_ACCESS_KEY = credentials.access_key

# Sentinel Hub (Copernicus Data Space) credentials come from .env (gitignored), never this file
def read_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value.strip().strip('"').strip("'"))


read_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
ID = os.environ["SENTINELHUB_CLIENT_ID"]
SECRET = os.environ["SENTINELHUB_CLIENT_SECRET"]

config = SHConfig()
config.sh_client_id = ID
config.sh_client_secret = SECRET

config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
# No config.save(): that writes the secret in plain text to ~/.config/sentinelhub/config.toml

# Create a session
client = BackendApplicationClient(client_id=ID)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=SECRET)



def proc_sent_tc(date, lon1, lat1, lon2, lat2, fp, sent_sat, resolution=60):

    coords_wgs84 = (lon1, lat1, lon2, lat2 )

    bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)

    # print(f"Image shape at {resolution} m resolution: {betsiboka_size} pixels")

    evalscript_true_color_s2 = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """


    evalscript_true_color_s3 = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B08", "B06", "B03"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B08, sample.B06, sample.B03];
    }
    """

    if sent_sat == "s2l1c":
        request_true_color = SentinelHubRequest(
            data_folder='/tmp',
            evalscript=evalscript_true_color_s2,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C.define_from(
                        "s2l1c", service_url=config.sh_base_url
                    ),
                    time_interval=(date, date),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=bbox_size,
            config=config,
        )

    elif sent_sat == "s2l2a":
        request_true_color = SentinelHubRequest(
            data_folder='/tmp',
            evalscript=evalscript_true_color_s2,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A.define_from(
                        "s2l2a", service_url=config.sh_base_url
                    ),
                    time_interval=(date, date),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=bbox_size,
            config=config,
        )

    elif sent_sat == "s3l1c":
        request_true_color = SentinelHubRequest(
            data_folder='/tmp',
            evalscript=evalscript_true_color_s3,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL3_OLCI.define_from(
                        "s3l1c", service_url=config.sh_base_url
                    ),
                    time_interval=(date, date),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=bbox_size,
            config=config,
        )


    true_color_imgs = request_true_color.get_data(save_data=True)
    
    folder = find_folder_with_tiff("/tmp/")

    df = xr.open_dataset(f"/tmp/{folder}/response.tiff").to_dataframe().reset_index()

    os.system(f"rm -rfv /tmp/{folder}")

    if df['band_data'].sum() == 0:
        print(f"No data found in bounds for {date}")
        return pd.DataFrame()

    df_pivot = df.pivot_table(index=['x', 'y'], columns='band', values='band_data').reset_index()

    # Normalize the band data if necessary (assuming 0-255 scale)
    # If your data is already in the correct scale, adjust or omit this step
    df_pivot[[1, 2, 3]] = df_pivot[[1, 2, 3]] / df_pivot[[1, 2, 3]].max() * 255

    # Convert to integer values
    df_pivot[[1, 2, 3]] = df_pivot[[1, 2, 3]].astype(int)

    # Create a hex color column
    df_pivot['color'] = '#' + df_pivot[1].apply(lambda x: f'{x:02x}') + df_pivot[2].apply(lambda x: f'{x:02x}') + df_pivot[3].apply(lambda x: f'{x:02x}')

    df_pivot.columns = ['x', 'y', 'red', 'green', 'blue', 'color']

    df_pivot.insert(0, 'date', date)

    # Convert 'date' to datetime if it's not already
    df_pivot['date'] = pd.to_datetime(df_pivot['date'])

    # Set 'date', 'lon', and 'lat' as indices
    df_pivot = df_pivot.set_index(['date', 'x', 'y'])

    return df_pivot


# Large view of east end
# lat1 = 46.22996034602979
# lon1 = -81.2505942984692
# lat2 = 46.18929459055976
# lon2 = -81.17794381038998

# Small view of east end
# lat1= 46.195450219342
# lon1 = -81.19299434094617
# lat2 = 46.20294240570666
# lon2 = -81.20270168110038

# Lake Penage
# lat1 = 46.288955366070184
# lon1 = -81.44353366986864
# lat2 = 46.17949449297964
# lon2 = -81.17363838649905

# date = "2024-12-24"
# fp = None
# sent_sat = "s2l2a"

# start_date = datetime.strptime("2024-08-01", "%Y-%m-%d")
# end_date = datetime.today()

# # Generate list of dates
# date_list = [(start_date + timedelta(days=x)).strftime("%Y-%m-%d") 
#              for x in range((end_date - start_date).days + 1)]

# [proc_daily(x) for x in date_list]


# proc_daily( "2024-12-19")

def proc_daily(date, sent_sat="s2l2a"):

    # Initialize a session using the profile
    session = boto3.Session(profile_name='personal')
    s3_client = session.client('s3')

    # Setup coords
    lat1 = 46.288955366070184
    lon1 = -81.44353366986864
    lat2 = 46.17949449297964
    lon2 = -81.17363838649905
    fp = None

    # Proc data
    outdat = proc_sent_tc(date, lon1, lat1, lon2, lat2, fp, sent_sat)
    outdat = outdat.reset_index()
    
    if len(outdat) > 0:
        outdat.to_csv(f"processed/{date}.csv", index=False)


        # Example data (replace with your actual data)
        # dat should be a DataFrame with columns x, y, and color

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the raster (scatterplot-like in this case for simplicity)
        sc = ax.scatter(outdat['x'], outdat['y'], c=outdat['color'], marker='s', s=100)

        # Add red points
        ax.scatter(-81.34968490328954, 46.27268309258747, color='red', label="Panache Bay Marina")
        ax.scatter(-81.20146454620291, 46.19863156087819, color='red', label="Camp")

        # Add annotations
        ax.text(-81.34968490328954, 46.28, "Panache Bay \n Marina", color="red", 
                fontsize=12, ha='center', va='center', fontweight='bold')
        ax.text(-81.19825731781991, 46.195, "Camp", color="red", 
                fontsize=12, ha='center', va='center', fontweight='bold')

        # Remove all axis labels, ticks, and the background grid
        ax.set_axis_off()

        # Save the figure
        plt.savefig(f"images/{date}.png", dpi=600, bbox_inches='tight', format='png')

        s3_client.upload_file(Filename=f"images/{date}.png", Bucket="penage-true-color-images", Key=f"{date}.png")
        print(f"Saved: s3://penage-true-color-images/{date}.png")


def proc_daily_point(date, sent_sat="s2l2a"):

    session = boto3.Session(profile_name='personal')
    s3_client = session.client('s3')

    # 0.5km x 0.5km box centered on (46.198907336078946, -81.20123776310807)
    # 0.25km ≈ 0.00225 deg lat; ≈ 0.00325 deg lon at lat 46.2°
    lat1 = 46.196660
    lon1 = -81.204488
    lat2 = 46.201155
    lon2 = -81.197988
    fp = None

    outdat = proc_sent_tc(date, lon1, lat1, lon2, lat2, fp, sent_sat, resolution=10)
    outdat = outdat.reset_index()

    if len(outdat) > 0:
        outdat.to_csv(f"processed/point_{date}.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 8))
        sc = ax.scatter(outdat['x'], outdat['y'], c=outdat['color'], marker='s', s=150)

        ax.scatter(-81.20123776310807, 46.198907336078946, color='red', label="Camp")
        ax.text(-81.20123776310807, 46.19940, "Camp", color="red",
                fontsize=12, ha='center', va='center', fontweight='bold')

        ax.set_axis_off()

        plt.savefig(f"images/point_{date}.png", dpi=600, bbox_inches='tight', format='png')
        plt.close(fig)

        s3_client.upload_file(Filename=f"images/point_{date}.png",
                              Bucket="penage-true-color-images",
                              Key=f"point/{date}.png")
        print(f"Saved: s3://penage-true-color-images/point/{date}.png")


if __name__ == "__main__":

    # Get today's date
    end_date = datetime.today()

    # Get the date one month before today
    start_date = end_date - relativedelta(months=1)

    # # Loop through each date in the range
    current_date = start_date
    while current_date <= end_date:
        proc_daily(current_date.strftime("%Y-%m-%d"), sent_sat="s2l2a")
        proc_daily_point(current_date.strftime("%Y-%m-%d"), sent_sat="s2l2a")
        current_date += timedelta(days=1)  # Move to the next day

    # cd /Users/andrewwoodill/Repos/penage-satellite-maps/ && source env/bin/activate && ipython senthub_truecolor.py

