import subprocess
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Tuple, Optional
from uuid import uuid4

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from ifdo.models import (
    ImageAcquisition,
    ImageCaptureMode,
    ImageData,
    ImageDeployment,
    ImageFaunaAttraction,
    ImageIllumination,
    ImageMarineZone,
    ImageNavigation,
    ImagePixelMagnitude,
    ImageQuality,
    ImageSpectralResolution, ImagePI,
)

from marimba.core.pipeline import BasePipeline
from marimba.lib import image
from marimba.main import __version__


class MRITCDemoPipeline(BasePipeline):

    @staticmethod
    def get_pipeline_config_schema() -> dict:
        return {
            "voyage_id": "IN2018_V06",
            "voyage_pi": "Alan Williams",
            "start_date": "2018-11-23",
            "end_date": "2018-12-19",
            "platform_id": "MRITC",
        }

    @staticmethod
    def get_collection_config_schema() -> dict:
        return {}

    def _import(self, data_dir: Path, source_path: Path, config: Dict[str, Any], **kwargs: dict):
        # Log the start of the import operation
        self.logger.info(f"Importing data from {source_path=} to {data_dir}")

        # Iterate over all files in the source path recursively
        for source_file in source_path.rglob("*"):
            # Filter for files with specific extensions
            if source_file.is_file() and source_file.suffix.lower() in [".csv", ".jpg", ".mp4"]:
                # Copy files to the destination directory, skipping if in dry run mode
                if not self.dry_run:
                    copy2(source_file, data_dir)
                # Log each file copied
                self.logger.debug(f"Copied {source_file.resolve().absolute()} -> {data_dir}")

    def get_image_output_file_name(self, file_path: Path) -> str:
        try:
            image = Image.open(file_path)
            exif_data = image._getexif() if hasattr(image, "_getexif") else None

            index = int(str(file_path).split("_")[-1].split(".")[0])

            if exif_data:
                # Extract DateTime from EXIF if available
                date_str = next((value for tag, value in exif_data.items() if TAGS.get(tag) == "DateTime"), None)
                if date_str:
                    date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    iso_timestamp = date.strftime("%Y%m%dT%H%M%SZ")

                    # Safely get config data with defaults
                    platform_id = self.config.get("platform_id", "UNKNOWN")
                    voyage_parts = self.config.get("voyage_id", "UNK_UNK").split("_")
                    deployment_id = str(file_path).split("/")[-3].split("_")[2]

                    # Construct and return new filename
                    return (
                        f"{platform_id}_SCP_"
                        f"{voyage_parts[0]}_{voyage_parts[1]}_{deployment_id}_"
                        f"{iso_timestamp}_{index:04d}.JPG"
                    )
                else:
                    self.logger.error(f"No EXIF DateTime tag found in image {file_path}")
            else:
                self.logger.error(f"No EXIF data found in image {file_path}")

        except IOError:
            self.logger.error(f"Error: Unable to open {file_path}. Are you sure it's an image?")

        # Return a default or error filename if necessary
        return "default_filename.JPG"

    def get_mp4_timestamp(self, file_path: Path) -> str:
        """Extract timestamp from an MP4 file using ffprobe."""
        try:
            # Use ffprobe to get the creation_time of the MP4 file
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'format_tags=creation_time', '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Parse the creation_time output
            creation_time_str = result.stdout.strip()
            if creation_time_str:
                creation_time = datetime.strptime(creation_time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                return creation_time.strftime('%Y%m%dT%H%M%SZ')
            else:
                self.logger.error(f"No creation time found in MP4 metadata for {file_path}")
                return "00000000T000000Z"
        except Exception as e:
            self.logger.error(f"Error extracting timestamp from MP4: {e}")
            return "00000000T000000Z"

    def _process(self, data_dir: Path, config: Dict[str, Any], **kwargs: dict):
        # Define directories for each type of file
        paths = {
            "images": data_dir / "images",
            "video": data_dir / "video",
            "data": data_dir / "data",
            "thumbs": data_dir / "thumbnails"
        }

        # Ensure all directories exist
        for path in paths.values():
            path.mkdir(exist_ok=True)

        # Initialize lists for tracking processed files
        jpg_list = []
        thumb_list = []

        # Extract common identifiers from config
        voyage_id = self.config.get("voyage_id")
        platform_id = self.config.get("platform_id")

        # Process each file in the data directory
        for file in data_dir.glob("*"):
            # Skip directories
            if not file.is_file():
                continue

            # Get deployment ID from the file path
            deployment_id = str(file).split("/")[-3].split("_")[2]
            file_ext = file.suffix.lower()  # Normalize file extension

            try:
                # Process image files
                if file_ext == ".jpg":
                    output_file_path = paths["images"] / self.get_image_output_file_name(file)
                    file.rename(output_file_path)
                    self.logger.info(f"Renamed image {file.name} -> {output_file_path}")
                    jpg_list.append(output_file_path)

                # Process MP4 files
                elif file_ext == ".mp4":
                    iso_timestamp = self.get_mp4_timestamp(file)
                    new_mp4_name = f"{platform_id}_{voyage_id}_{deployment_id}_{iso_timestamp}.mp4"
                    output_file_path = paths["video"] / new_mp4_name
                    file.rename(output_file_path)
                    self.logger.info(f"Renamed MP4 {file.name} -> {output_file_path}")

                # Process CSV files
                if file_ext == ".csv":
                    new_csv_name = f"{platform_id}_{voyage_id}_{deployment_id}.CSV"
                    output_file_path = paths["data"] / new_csv_name
                    file.rename(output_file_path)
                    self.logger.info(f"Renamed CSV {file.name} -> {output_file_path}")

            except (FileNotFoundError, IOError) as e:
                self.logger.error(f"Error processing file {file.name}: {str(e)}")
                continue

        # Generate thumbnails for processed images
        for jpg in jpg_list:
            output_filename = f"{jpg.stem}_THUMB{jpg.suffix}"
            output_path = paths["thumbs"] / output_filename
            self.logger.info(f"Generating thumbnail image: {output_path}")

            try:
                image.resize_fit(jpg, 300, 300, output_path)
                thumb_list.append(output_path)
            except Exception as e:
                self.logger.error(f"Error creating thumbnail for {jpg.name}: {str(e)}")

        # Create an overview image if thumbnails exist
        if thumb_list:
            overview_path = data_dir / "overview.jpg"
            self.logger.info(f"Creating thumbnail overview image: {overview_path}")

            try:
                image.create_grid_image(thumb_list, overview_path)
            except Exception as e:
                self.logger.error(f"Error creating overview image: {str(e)}")

    def _package(
            self,
            data_dir: Path,
            config: Dict[str, Any],
            **kwargs: dict
    ) -> Dict[Path, Tuple[Path, Optional[ImageData], Optional[Dict[str, Any]]]]:

        # Initialise an empty dictionary to store file mappings
        data_mapping: Dict[Path, Tuple[Path, Optional[List[ImageData]], Optional[Dict[str, Any]]]] = {}

        # Recursively gather all file paths from the data directory
        file_paths = data_dir.rglob("*")

        # Read the sensor data CSV file and parse the 'FinalTime' column as datetime, flooring to the nearest second for matching timestamps
        sensor_data_df = pd.read_csv(next((data_dir / "data").glob("*.CSV")))
        sensor_data_df["FinalTime"] = pd.to_datetime(
            sensor_data_df["FinalTime"],
            format="%Y-%m-%d %H:%M:%S.%f"
        ).dt.floor("s").astype(str)

        for file_path in file_paths:

            # Extract the deployment ID from the data directory's path
            deployment_id = str(data_dir).split("/")[-2]

            # Define the output path relative to the deployment ID
            output_file_path = deployment_id / file_path.relative_to(data_dir)

            # Process only valid image files (JPGs), excluding thumbnails and overview images
            if (
                    file_path.is_file()
                    and file_path.suffix.lower() == ".jpg"
                    and "_THUMB" not in file_path.name
                    and "overview" not in file_path.name
            ):
                # Extract the ISO timestamp from the filename
                iso_timestamp = file_path.name.split("_")[5]
                # Convert the ISO timestamp to a datetime object
                target_datetime = pd.to_datetime(iso_timestamp, format="%Y%m%dT%H%M%SZ")

                # Fetch the first matching row from the sensor data
                matching_row = sensor_data_df.loc[sensor_data_df["FinalTime"] == target_datetime]

                if not matching_row.empty:
                    first_row = matching_row.iloc[0]

                    # Construct the ImageData instance with necessary metadata
                    image_data = ImageData(
                        # iFDO core
                        image_datetime=datetime.strptime(iso_timestamp, "%Y%m%dT%H%M%SZ"),
                        image_latitude=float(first_row["UsblLatitude"]),
                        image_longitude=float(first_row["UsblLongitude"]),
                        image_altitude=float(first_row["Altitude"]),
                        image_coordinate_reference_system="EPSG:4326",
                        # image_coordinate_uncertainty_meters=None,
                        # image_context=None,
                        # image_project=None,
                        image_event=str(first_row["Operation"]),
                        image_platform=self.config.get("platform_id"),
                        image_sensor=str(first_row["Camera"]),
                        image_uuid=str(uuid4()),
                        image_pi=ImagePI(name="Keiko Abe", orcid="0000-0000-0000-0000"),
                        image_creators=[ImagePI(name="Keiko Abe", orcid="0000-0000-0000-0000")],
                        image_license="CC BY 4.0",
                        image_copyright="CSIRO",
                        # image_abstract=None,
                        # Note: Marimba automatically calculates and injects the SHA256 hash during packaging
                        # image_hash_sha256=image_hash_sha256,

                        # # iFDO capture (optional)
                        image_acquisition=ImageAcquisition.PHOTO,
                        image_quality=ImageQuality.PRODUCT,
                        image_deployment=ImageDeployment.SURVEY,
                        image_navigation=ImageNavigation.SATELLITE,
                        # image_scale_reference=ImageScaleReference.NONE,
                        image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
                        image_pixel_mag=ImagePixelMagnitude.CM,
                        image_marine_zone=ImageMarineZone.SEAFLOOR,
                        image_spectral_resolution=ImageSpectralResolution.RGB,
                        image_capture_mode=ImageCaptureMode.TIMER,
                        image_fauna_attraction=ImageFaunaAttraction.NONE,
                        # image_area_square_meter=None,
                        # image_meters_above_ground=None,
                        # image_acquisition_settings=None,
                        # image_camera_yaw_degrees=None,
                        image_camera_pitch_degrees=first_row["Pitch"],
                        image_camera_roll_degrees=first_row["Roll"],
                        # image_overlap_fraction=0,
                        image_datetime_format="%Y-%m-%d %H:%M:%S.%f",
                        # image_camera_pose=None,
                        # image_camera_housing_viewport=None,
                        # image_flatport_parameters=None,
                        # image_domeport_parameters=None,
                        # image_camera_calibration_model=None,
                        # image_photometric_calibration=None,
                        # image_objective=None,
                        image_target_environment="Benthic habitat",
                        # image_target_timescale=None,
                        # image_spatial_constraints=None,
                        # image_temporal_constraints=None,
                        # image_time_synchronization=None,
                        image_item_identification_scheme="<platform_id>_<camera_id>_<voyage_id>_<deployment_number>_<datetimestamp>_<image_id>.<ext>",
                        image_curation_protocol=f"Processed with Marimba v{__version__}",

                        # # iFDO content (optional)
                        # Note: Marimba automatically calculates and injects image_entropy and image_average_color during packaging
                        # image_entropy=0.0,
                        # image_particle_count=None,
                        # image_average_color=[0, 0, 0],
                        # image_mpeg7_colorlayout=None,
                        # image_mpeg7_colorstatistics=None,
                        # image_mpeg7_colorstructure=None,
                        # image_mpeg7_dominantcolor=None,
                        # image_mpeg7_edgehistogram=None,
                        # image_mpeg7_homogenoustexture=None,
                        # image_mpeg7_stablecolor=None,
                        # image_annotation_labels=None,
                        # image_annotation_creators=None,
                        # image_annotations=None,
                    )

                    # Add the image file, metadata (ImageData), and ancillary metadata to the data mapping
                    data_mapping[file_path] = (output_file_path, [image_data], first_row.to_dict())

            # For non-image files, add them without metadata
            elif file_path.is_file():
                data_mapping[file_path] = (output_file_path, None, None)

        # Return the constructed data mapping for all files
        return data_mapping
