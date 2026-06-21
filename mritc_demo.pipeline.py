"""Marimba Pipeline for processing MRITC data."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any
from uuid import uuid4

import exiftool
import pandas as pd
from ifdo.models import (
    ImageAcquisition,
    ImageCaptureMode,
    ImageContext,
    ImageCreator,
    ImageData,
    ImageDeployment,
    ImageFaunaAttraction,
    ImageIllumination,
    ImageLicense,
    ImageMarineZone,
    ImageNavigation,
    ImagePI,
    ImagePixelMagnitude,
    ImageQuality,
    ImageSpectralResolution,
)
from marimba.core.pipeline import BasePipeline
from marimba.core.schemas.ifdo import iFDOMetadata
from marimba.lib import image
from marimba.main import __version__
from PIL import Image
from PIL.ExifTags import TAGS


class MRITCDemoPipeline(BasePipeline):
    """
    Marimba Pipeline for processing MRITC data.

    This class extends BasePipeline to provide specific functionality for handling MRITC data. It includes methods for
    importing, processing, and packaging data from marine surveys, handling image and video files with associated
    metadata.

    Methods:
        get_pipeline_config_schema(): Get the pipeline configuration schema.
        get_collection_config_schema(): Get the collection configuration schema.
        get_image_output_file_name(file_path): Generate standardized output filename for image files.
        get_mp4_timestamp(file_path): Extract timestamp from MP4 file metadata.
        _import(data_dir, source_path, config, **kwargs): Import data from source to destination directory.
        _process(data_dir, config, **kwargs): Process the imported data.
        _package(data_dir, config, **kwargs): Package the processed data with metadata.
    """

    CREATION_TIME_TAG = "QuickTime:CreationTime"

    def __init__(
        self,
        root_path: str | Path,
        config: dict[str, Any] | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize a new Pipeline instance.

        Args:
            root_path (str | Path): Base directory path where the pipeline will store its data and configuration files.
            config (dict[str, Any] | None, optional): Pipeline configuration dictionary. If None, default configuration
             will be used. Defaults to None.
            dry_run (bool, optional): If True, prevents any filesystem modifications. Useful for validation and testing.
             Defaults to False.
        """
        super().__init__(
            root_path,
            config,
            dry_run=dry_run,
            metadata_class=iFDOMetadata,
        )

    @staticmethod
    def get_pipeline_config_schema() -> dict:
        """
        Get the pipeline configuration schema for the MRITC pipeline.

        Returns:
            dict: Configuration parameters for the pipeline
        """
        return {
            "voyage_id": "IN2018_V06",
            "voyage_pi": "Alan Williams",
            "start_date": "2018-11-23",
            "end_date": "2018-12-19",
            "platform_id": "MRITC",
        }

    @staticmethod
    def get_collection_config_schema() -> dict:
        """
        Get the collection configuration schema for the MRITC pipeline.

        Returns:
            dict: Configuration parameters for the collection
        """
        return {}

    def get_metadata_header(
        self,
        context: str,
        collection_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return header metadata for the given context."""
        metadata = {
            "copyright": "CSIRO",
            "license_name": "CC BY-NC-SA 4.0",
            "license_uri": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        }

        # Context-specific names (optional - smart defaults will be used if not specified)
        if context == "dataset":
            voyage_id = self.config.get('voyage_id', 'UNKNOWN')
            metadata["name"] = f"Complete {voyage_id} Dataset"
        elif context == "pipeline":
            platform_id = self.config.get('platform_id', 'UNKNOWN')
            metadata["name"] = f"{platform_id} Camera System"
        elif context == "collection":
            # Set custom name for collection
            # If there's a deployment_id in the config, use it; otherwise use a generic collection name
            deployment_id = collection_config.get("deployment_id") if collection_config else None
            if deployment_id:
                metadata["name"] = f"{self.config['platform_id']} Deployment {deployment_id}"
            else:
                metadata["name"] = f"{self.config['platform_id']} Collection"

        return metadata

    def _import(
        self,
        data_dir: Path,
        source_path: Path,
        config: dict[str, Any],  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> None:
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
        """
        Generate a standardized output filename for an image file based on its metadata.

        This method extracts EXIF data from the image file to determine the timestamp and combines it with
        configuration parameters to create a standardized filename following the format:
        <platform_id>_SCP_<voyage_prefix>_<voyage_suffix>_<deployment_id>_<timestamp>_<index>.JPG

        Args:
            file_path (Path): Path to the source image file.

        Returns:
            str: Standardized filename for the image.

        Raises:
            OSError: If the image file cannot be opened or read.
        """
        try:
            image = Image.open(file_path)
            exif_data = getattr(image, "_getexif", lambda: None)()

            index = int(str(file_path).split("_")[-1].split(".")[0])

            if exif_data:
                # Extract DateTime from EXIF if available
                date_str = next((value for tag, value in exif_data.items() if TAGS.get(tag) == "DateTime"), None)
                if date_str:
                    date = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
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
                self.logging.exception(f"No EXIF DateTime tag found in image {file_path}")
            else:
                self.logging.exception(f"No EXIF data found in image {file_path}")

        except OSError:
            self.logging.exception(f"Error: Unable to open {file_path}. Are you sure it's an image?")

        # Return a default or error filename if necessary
        return "default_filename.JPG"

    def get_mp4_timestamp(self, et: exiftool.ExifToolHelper, file_path: Path) -> str:
        """Extract timestamp from an MP4 file using exiftool."""
        try:
            result: list[dict[str, str]] = et.get_tags([file_path], tags=[self.CREATION_TIME_TAG])
            creation_time_str = result[0].get(self.CREATION_TIME_TAG, "00:00:00 00:00:00.000000Z")
            creation_time = datetime.strptime(creation_time_str, "%Y:%m:%d %H:%M:%S.%fZ")
            return creation_time.strftime("%Y%m%dT%H%M%SZ")
        except Exception as e:
            self.logging.exception(f"Error extracting timestamp from MP4: {e}")
            return "000000T000000Z"

    def _process(
        self,
        data_dir: Path,
        config: dict[str, Any],  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> None:
        # Define directories for each type of file
        paths = {
            "images": data_dir / "images",
            "video": data_dir / "video",
            "data": data_dir / "data",
            "thumbs": data_dir / "thumbnails",
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
        with exiftool.ExifToolHelper() as et:
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
                        iso_timestamp = self.get_mp4_timestamp(et, file)
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

                except (OSError, FileNotFoundError) as e:
                    self.logging.exception(f"Error processing file {file.name}: {e!s}")
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
                self.logging.exception(f"Error creating thumbnail for {jpg.name}: {e!s}")

        # Create an overview image if thumbnails exist
        if thumb_list:
            overview_path = data_dir / "overview.jpg"
            self.logger.info(f"Creating thumbnail overview image: {overview_path}")

            try:
                image.create_grid_image(thumb_list, overview_path)
            except Exception as e:
                self.logging.exception(f"Error creating overview image: {e!s}")

    def _package(
        self,
        data_dir: Path,
        config: dict[str, Any],  # noqa: ARG002
        **kwargs: dict,  # noqa: ARG002
    ) -> dict[Path, tuple[Path, ImageData | None, dict[str, Any] | None]]:

        # Initialise an empty dictionary to store file mappings
        data_mapping: dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]] = {}

        # Recursively gather all file paths from the data directory
        file_paths = data_dir.rglob("*")

        # Read the sensor data CSV file and parse the 'FinalTime' column as datetime, flooring to the nearest second
        # for matching timestamps
        sensor_data_df = pd.read_csv(next((data_dir / "data").glob("*.CSV")))
        sensor_data_df["FinalTime"] = pd.to_datetime(
            sensor_data_df["FinalTime"],
            format="%Y-%m-%d %H:%M:%S.%f",
        ).dt.floor("s")

        for file_path in file_paths:

            # Extract the deployment ID from the data directory's path
            deployment_id = str(data_dir).split("/")[-2]

            # Define the output path relative to the deployment ID
            output_file_path = deployment_id / file_path.relative_to(data_dir)

            # Process only valid image files (JPGs) and videos (MP4s), excluding thumbnails and overview images
            if (
                file_path.is_file()
                and file_path.suffix.lower() in [".jpg", ".mp4"]
                and "_THUMB" not in file_path.name
                and "overview" not in file_path.name
            ):
                # Extract the ISO timestamp from the filename
                index_map = {".jpg": 5, ".mp4": 4}
                iso_timestamp = file_path.stem.split("_")[index_map.get(file_path.suffix.lower(), -1)]

                # Convert the ISO timestamp to a datetime object
                target_datetime = pd.to_datetime(iso_timestamp, format="%Y%m%dT%H%M%SZ")

                # Check file type and perform the appropriate matching
                if file_path.suffix.lower() == ".jpg":
                    # For jpgs, find the perfect match
                    matching_row = sensor_data_df.loc[sensor_data_df["FinalTime"] == target_datetime]
                elif file_path.suffix.lower() == ".mp4":
                    # For mp4s, find the closest match
                    time_diffs = abs(sensor_data_df["FinalTime"] - target_datetime)
                    matching_row = sensor_data_df.loc[time_diffs.idxmin()]
                else:
                    raise ValueError("Unsupported file type")

                if not matching_row.empty:
                    if isinstance(matching_row, pd.DataFrame):
                        first_row = matching_row.iloc[0].copy()
                    elif isinstance(matching_row, pd.Series):
                        first_row = matching_row.copy()
                    else:
                        raise ValueError(f"Unexpected type for matching_row: {type(matching_row)}")

                    # Convert any Timestamp fields in first_row directly to ISO 8601 strings
                    first_row = first_row.map(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)

                    # Construct the ImageData instance with necessary metadata
                    # ruff: noqa: ERA001
                    image_data = ImageData(
                        # iFDO core
                        image_datetime=datetime.strptime(iso_timestamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc),
                        image_latitude=float(first_row["UsblLatitude"]),
                        image_longitude=float(first_row["UsblLongitude"]),
                        image_altitude=float(first_row["Altitude"]),
                        image_coordinate_reference_system="EPSG:4326",
                        # image_coordinate_uncertainty_meters=None,
                        # image_context=None,
                        # image_project=None,
                        image_event=ImageContext(name=deployment_id),
                        image_platform=ImageContext(name=self.config.get("platform_id")),
                        image_sensor=ImageContext(name=str(first_row["Camera"])),
                        image_uuid=str(uuid4()),
                        image_pi=ImagePI(name="Keiko Abe", uri="https://orcid.org/0000-0000-0000-0000"),
                        image_creators=[ImageCreator(name="Keiko Abe", uri="https://orcid.org/0000-0000-0000-0000")],
                        image_license=ImageLicense(
                            name="CC BY-NC-SA 4.0",
                            uri="https://creativecommons.org/licenses/by-nc-sa/4.0/",
                        ),
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
                        # image_area_square_meters=None,
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
                        # image_stereo_camera_calibration_model=None,
                        # image_photometric_calibration=None,
                        # image_objective=None,
                        image_target_environment="Benthic habitat",
                        # image_target_timescale=None,
                        # image_spatial_constraints=None,
                        # image_temporal_constraints=None,
                        # image_time_synchronisation=None,
                        image_item_identification_scheme="<platform_id>_<camera_id>_<voyage_id>_<deployment_number>_<datetimestamp>_<image_id>.<ext>",
                        image_curation_protocol=f"Processed with Marimba v{__version__}",

                        # # iFDO content (optional)
                        # image_entropy=0.0,
                        # image_particle_count=None,
                        # image_average_color=[0, 0, 0],
                        # image_mpeg7_colorlayout=None,
                        # image_mpeg7_colorstatistic=None,
                        # image_mpeg7_colorstructure=None,
                        # image_mpeg7_dominantcolor=None,
                        # image_mpeg7_edgehistogram=None,
                        # image_mpeg7_homogeneoustexture=None,
                        # image_mpeg7_scalablecolor=None,
                        # image_annotation_labels=None,
                        # image_annotation_creators=None,
                        # image_annotations=None,
                    )

                    # Add the image file, metadata (ImageData), and ancillary metadata to the data mapping
                    metadata = self._metadata_class(image_data)
                    data_mapping[file_path] = output_file_path, [metadata], first_row.to_dict()

            # For non-image files, add them without metadata
            elif file_path.is_file():
                data_mapping[file_path] = (output_file_path, None, None)

        # Return the constructed data mapping for all files
        return data_mapping
