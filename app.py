import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError
from PIL import Image
from io import BytesIO

# AWS S3 configuration
BUCKET_NAME = "penage-true-color-images"  # Change to your bucket name

# Available views: label -> S3 prefix
VIEWS = {
    "Planet 3m (2km)": "planet/",
    "Planet camp (600m)": "planet_point/",
    "Lake view": "",
    "Camp point view (0.5km)": "point/",
}

# Initialize AWS session using credentials from Streamlit secrets
session = boto3.Session(
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name=st.secrets["aws"]["region_name"]
)
s3_client = session.client('s3')

# Function to list files in the S3 bucket with a given prefix
def list_dates_in_bucket(bucket_name, prefix):
    try:
        kwargs = {"Bucket": bucket_name, "Prefix": prefix}
        # Empty prefix = root-only listing (skip subprefixes like point/)
        if not prefix:
            kwargs["Delimiter"] = "/"
        response = s3_client.list_objects_v2(**kwargs)
        if 'Contents' in response:
            # Extract file names, strip the prefix, and return as a sorted list
            return sorted(
                [obj['Key'].replace(prefix, '').replace('.png', '') for obj in response['Contents']],
                reverse=True  # Sort in descending order (most recent first)
            )
        else:
            return []
    except NoCredentialsError:
        st.error("AWS credentials not found.")
        return []

# Function to download and display image
def download_and_display_image(bucket_name, key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        img_data = response['Body'].read()
        img = Image.open(BytesIO(img_data))
        return img
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return None

# Streamlit app
st.title("Lake Panache Satellite Images")

# View selector
view_label = st.radio("View", list(VIEWS.keys()), horizontal=True)
prefix = VIEWS[view_label]

# Dropdown for available dates
dates = list_dates_in_bucket(BUCKET_NAME, prefix)

if dates:
    # Default to the most recent date; reset if previous selection isn't in this view
    if "selected_date" not in st.session_state or st.session_state["selected_date"] not in dates:
        st.session_state["selected_date"] = dates[0]

    selected_date = st.selectbox("Select a date", dates, index=dates.index(st.session_state["selected_date"]))

    # Update session state when a new selection is made
    if selected_date != st.session_state["selected_date"]:
        st.session_state["selected_date"] = selected_date
        st.rerun()  # Force rerun to update the image immediately

    # Construct the S3 key for the selected date
    s3_key = f"{prefix}{st.session_state['selected_date']}.png"

    # Download and display the image
    image = download_and_display_image(BUCKET_NAME, s3_key)
    if image:
        st.image(image, caption=f"{view_label} — {st.session_state['selected_date']}", use_container_width=True)

else:
    st.warning("No images found in the bucket for this view.")
