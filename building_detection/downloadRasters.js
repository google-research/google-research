/*
* This script was made to download the data of the Open Buildings 2.5D Temporal Dataset and to be runned on Google Earth Engine code space: https://code.earthengine.google.com/ 
more info here: https://sites.research.google/gr/open-buildings/temporal/
*/

/*
* Once in Google Earth Engine code editor: https://code.earthengine.google.com/?scriptPath=Examples:Datasets/GOOGLE/GOOGLE_Research_open-buildings-temporal_v1
* First, define a Polygon called "geometryPolygon"
* Then, modify the inputs in this script as years and bands that you want to export 
* Finally, just run the code

* The files will appear in the task tab that is on the right, the files can downloaded in your google drive 
*/

// Inputs
/* Years to export */
var years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023];
/* Bands to export */
var bandCodes = ['B0', 'B1', 'B2']; /*B0 -> building fractional count, B1 -> Building Height, B2 -> Building_presence */
/* Collection and version */
var col = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1'); /* Look this link if a new version is available: https://console.cloud.google.com/storage/browser/open-buildings-temporal-data */

/* 
* Do not modify anything from here 
*/

// Previous 
var bandMapping = {
  'B0': 'building_fractional_count',
  'B1': 'building_height',
  'B2': 'building_presence'
}; /*To just define B0,B1 or B2 in inputs*/

// Function to filter and export data for the given year and band
function exportBandYear(year, bandCode) {
  // Convert the band code (B0, B1, B2) to the actual band name
  var bandName = bandMapping[bandCode];
  
  // Check if the bandCode is valid
  if (!bandName) {
    print('Error: Invalid band code ' + bandCode);
    return;
  }

  // Filter the collection for the given years
  var startDate = year + '-01-01';
  var endDate = year + '-12-31';
  var col_filtered = col.filterDate(startDate, endDate)
                        .filterBounds(geometryPolygon);
  
  // Create a mosaic for the filtered collection
  var mosaic = col_filtered.mosaic();
  
  // Select the desired band using the correct band name
  var band = mosaic.select(bandName).clip(geometryPolygon);
  
  // Visualize the band on the map (optional)
  Map.addLayer(band, {}, bandName + ' - ' + year);
  
  // Export the band as a GeoTIFF to Google Drive
  Export.image.toDrive({
    image: band,
    description: bandName + '_' + year,
    scale: 10,  // Scale in meters/pixel (adjust as necessary)
    region: geometryPolygon,
    fileFormat: 'GeoTIFF',
    maxPixels: 1e13 
  });
}

// Download loop through the years and band codes
years.forEach(function(year) {
  bandCodes.forEach(function(bandCode) {
    exportBandYear(year, bandCode);
  });
});
