"""
Google Earth Engine utility functions.
"""
import ee
import numpy as np
import logging
from typing import Dict, Any, List
from tqdm import tqdm
import time
import sys
sys.path.append('..')
from config.config import GEE_CONFIG, DATASET_CONFIG

def initialize_gee() -> None:
    """Initialize Google Earth Engine with error handling."""
    try:
        ee.Authenticate()
        ee.Initialize(project=GEE_CONFIG['PROJECT_ID'])
        logging.info("Successfully initialized Google Earth Engine")
    except Exception as e:
        logging.error(f"Failed to initialize GEE: {str(e)}")
        raise

def get_region_of_interest(region_name: str = GEE_CONFIG['REGION']) -> ee.Geometry:
    """Get the geometry for the region of interest."""
    try:
        region = ee.FeatureCollection("FAO/GAUL/2015/level1") \
            .filter(ee.Filter.eq("ADM1_NAME", region_name)) \
            .geometry()
        logging.info(f"Successfully retrieved geometry for {region_name}")
        return region
    except Exception as e:
        logging.error(f"Failed to get region geometry: {str(e)}")
        raise

def fetch_gee_data(region: dict) -> Dict[str, ee.ImageCollection]:
    """
    Fetch climate data from Google Earth Engine.
    
    Args:
        region: Region of interest
        
    Returns:
        Dictionary of GEE ImageCollections
    """
    try:
        # Set date range
        start_date = '2015-01-01'
        end_date = '2023-12-31'
        
        # Define datasets
        datasets = {
            'LST': ee.ImageCollection('MODIS/061/MOD11A2')
                .select('LST_Day_1km')
                .filterDate(start_date, end_date),
                
            'Precipitation': ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
                .select('precipitation')
                .filterDate(start_date, end_date),
                
            'Humidity': ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
                .select(['Tair_f_inst', 'Psurf_f_inst', 'Qair_f_inst'])
                .filterDate(start_date, end_date),
                
            'SolarRadiation': ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
                .select('SWdown_f_tavg')
                .filterDate(start_date, end_date),
                
            'SoilMoisture': ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
                .select('SoilMoi0_10cm_inst')
                .filterDate(start_date, end_date),
                
            'WindSpeed': ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
                .select('Wind_f_inst')
                .filterDate(start_date, end_date),
                
            'Evapotranspiration': ee.ImageCollection("MODIS/061/MOD16A2")
                .select('ET')
                .filterDate(start_date, end_date),
                
            'NDVI': ee.ImageCollection('MODIS/061/MOD13Q1')
                .select('NDVI')
                .filterDate(start_date, end_date),
                
            'CloudCover': ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                .select('probability')
                .filterDate(start_date, end_date)
        }
        
        # Filter by region
        for name, collection in datasets.items():
            datasets[name] = collection.filterBounds(region)
        
        logging.info("Successfully fetched all datasets from GEE")
        return datasets
        
    except Exception as e:
        logging.error(f"Failed to fetch GEE data: {str(e)}")
        raise

def batch_process_collection(collection: ee.ImageCollection, 
                           region: ee.Geometry,
                           temporal_reducer: str = 'mean',
                           scale: int = 5000,
                           batch_size: int = 30) -> np.ndarray:
    """
    Process an image collection in batches with temporal reduction.
    
    Args:
        collection: Image collection to process
        region: Region to sample
        temporal_reducer: Reducer to use for temporal aggregation ('mean' or 'sum')
        scale: Scale in meters for sampling
        batch_size: Number of days to process in each batch
        
    Returns:
        Numpy array of processed values
    """
    try:
        # Get date range
        date_range = collection.reduceColumns(ee.Reducer.minMax(), ['system:time_start'])
        start_time = ee.Date(date_range.get('min'))
        end_time = ee.Date(date_range.get('max'))
        
        # Calculate number of months between start and end
        months = end_time.difference(start_time, 'month').round()
        n_months = months.getInfo()
        
        values = []
        
        # Process by monthly chunks
        for i in tqdm(range(0, n_months, batch_size), desc="Processing batches"):
            chunk_start = start_time.advance(i, 'month')
            chunk_end = chunk_start.advance(batch_size, 'month')
            
            # Get chunk of images
            chunk = collection.filterDate(chunk_start, chunk_end)
            
            # Apply temporal reduction
            if temporal_reducer == 'mean':
                reduced = chunk.mean()
            else:  # sum
                reduced = chunk.sum()
            
            # Sample the reduced image
            result = reduced.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=scale,
                maxPixels=1e9
            ).getInfo()
            
            # Get the first (and usually only) band value
            band_value = next(iter(result.values()))
            values.append(float(band_value) if band_value is not None else 0)
            
        return np.array(values, dtype=np.float32)
        
    except Exception as e:
        logging.error(f"Error in batch processing: {str(e)}")
        raise

def convert_gee_data_to_numpy(gee_datasets: Dict[str, ee.ImageCollection], 
                            region: ee.Geometry) -> Dict[str, np.ndarray]:
    """
    Convert GEE ImageCollections to numpy arrays.
    
    Args:
        gee_datasets: Dictionary of GEE ImageCollections
        region: Region to sample
        
    Returns:
        Dictionary of numpy arrays
    """
    try:
        numpy_datasets = {}
        total_datasets = len(gee_datasets)
        logging.info(f"Starting conversion of {total_datasets} datasets to numpy arrays")
        
        # Define temporal reduction strategy for each dataset
        temporal_reducers = {
            'LST': 'mean',
            'Precipitation': 'sum',
            'Humidity': 'mean',
            'SolarRadiation': 'mean',
            'SoilMoisture': 'mean',
            'WindSpeed': 'mean',
            'Evapotranspiration': 'sum',
            'NDVI': 'mean',
            'CloudCover': 'mean'
        }
        
        for idx, (name, collection) in enumerate(gee_datasets.items(), 1):
            start_time = time.time()
            logging.info(f"Converting {name} dataset ({idx}/{total_datasets})...")
            
            try:
                if name == 'Humidity':
                    # Special handling for humidity (multiple bands)
                    temp = batch_process_collection(
                        collection.select('Tair_f_inst'),
                        region,
                        temporal_reducer='mean'
                    )
                    pressure = batch_process_collection(
                        collection.select('Psurf_f_inst'),
                        region,
                        temporal_reducer='mean'
                    )
                    q = batch_process_collection(
                        collection.select('Qair_f_inst'),
                        region,
                        temporal_reducer='mean'
                    )
                    numpy_datasets[name] = {
                        'temperature_2m': temp,
                        'surface_pressure': pressure,
                        'specific_humidity': q
                    }
                else:
                    # Process other datasets normally
                    numpy_datasets[name] = batch_process_collection(
                        collection,
                        region,
                        temporal_reducer=temporal_reducers[name]
                    )
                
                end_time = time.time()
                logging.info(f"Completed {name} conversion in {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                logging.error(f"Error converting {name} dataset: {str(e)}")
                logging.error("Skipping this dataset and continuing with others...")
                continue
        
        logging.info("Successfully converted all datasets to numpy arrays")
        return numpy_datasets
        
    except Exception as e:
        logging.error(f"Failed to convert GEE data to numpy: {str(e)}")
        raise

def sample_region(image: ee.Image, region: ee.Geometry, scale: int = 5000) -> Dict:
    """
    Sample an image over a region using a reasonable scale.
    
    Args:
        image: EE image to sample
        region: Region to sample over
        scale: Scale in meters for sampling
        
    Returns:
        Dictionary of sampled values
    """
    try:
        # Get the mean value for the region
        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e9
        ).getInfo()
        return values
    except Exception as e:
        logging.error(f"Error sampling region: {str(e)}")
        raise

def export_to_drive(image: ee.Image, description: str, region_of_interest: ee.Geometry) -> None:
    """Export GEE image to Google Drive."""
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            fileFormat='GeoTIFF',
            region=region_of_interest.bounds(),
            scale=1000
        )
        task.start()
        logging.info(f"Started export task for {description}")
    except Exception as e:
        logging.error(f"Failed to export {description}: {str(e)}")
        raise
