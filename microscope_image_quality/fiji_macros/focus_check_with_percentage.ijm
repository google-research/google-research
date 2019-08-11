// This script generates whole-plate montages per channel for inspecting data quality.
// Requirements:
//   - That the images live in a single directory already. 
//inNames = newArray("G:/Export_G/20190301_170231_487/");

// Use forward slash "/" and end the path with a "/" too.
newName = "F:/20190711_122101_017/";
montageFolderName = "OldMacro_AGP/";
inNames = newArray(newName);

min_thresh = 0.5000;
max_thresh = 1000000000000000000000000000000.0000

recompute_all=false;
all_historical_outputs_path = "E:/quality_montages/"; // This shouldn't be edited.


if( ! File.isDirectory(all_historical_outputs_path) ) {
	File.makeDirectory(all_historical_outputs_path);
}

rows = newArray("A","B","C","D","E","F","G","H");
columns = newArray("01","02","03","04","05","06","07","08","09","10","11","12");


channels = newArray("CY3-AGP");

maxChannels = channels.length; 

num_random_sites = 4;
start_site = 0;
end_site = 75;

site_side_length = 2; // For up to 4 sites.

image_height = 2960; //pixels
image_width = 5056; // pixels
crop_side_length = 84*3; // pixels

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

function randint(start, end) {
	// Uniform random, inclusive.
	//a = randint(1,3);
  a=random();
  return round(a*(end-start) + start);
}

function random_site_str(start, end, n) {
	// start: int, starting site number (inclusive)
	// end: int, ending site number (inclusive)
	// n: number of site to choose randomly
	// returns: string, e.g. ((00)|(24)|(43))
	out = "(";
	for (i = 0; i < n; i++) {
		a = randint(start+i*(end-start)/n, start+(i+1)*(end-start)/n);
		while (lengthOf(""+a)<3) a="0"+a; 
		if(i != 0) {
			out = out + "|";
		}
		out = out + "("+a+")";
	}
	return out + ")";
}

function percentage_score(min_threshold, max_threshold) {
	/*
	 * Calculates a percentage score for focus quality
	 * 
	 * Args: minimum and maximum values for thresholding
	 * 
	 * Returns: percentage score of focus quality for plate
	*/
	selectWindow("Probabilities");
	run("Make Substack...", "channels=1-3");
	run("Z Project...", "projection=[Sum Slices]");
	selectWindow("SUM_Probabilities-1");
	setAutoThreshold("Default dark");
	setThreshold(min_threshold, max_max_threshold);
	call("ij.plugin.frame.ThresholdAdjuster.setMode", "B&W");
	setOption("BlackBackground", true);
	run("Convert to Mask");
	getRawStatistics(nPixels, mean, min, max, std, histogram);
	
	print("Percentage patches in-focus: " + round(100*mean/255) +"%");
	
	close();
	selectWindow("Probabilities");
	close();
	selectWindow("Probabilities-1");
	close();
	
	print("Done with quality");
	setBatchMode(true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

run("Close All");
for(scan=0; scan<inNames.length; scan++) {
	inName = inNames[scan];
	print(inName);
	pathName = inName + montageFolderName;
	if( ! File.isDirectory(pathName) ) {
		File.makeDirectory(pathName);
	}
	else {
		print("Directory already exists!");
	}
	random("seed",0);
	setBatchMode(true); //so we don't flash all the images 

	print("starting Zoom montage creation...");
	for(c=0;c<maxChannels;++c) {
		endXYloop = false;
		ch = channels[c];
		for (x=0;x<rows.length;++x) {
			print("Processing row: " +rows[x] );
			for(y=0;y<columns.length;++y) {
				well = rows[x] + columns[y];
				savename = ""+pathName+"Well_zoom_"+well+"_"+ch+".tif";
				if( !File.exists( savename ) | recompute_all  ) {
          site_str = "Site"+random_site_str(start_site, end_site, num_random_sites);
          run("Image Sequence...", "open="+inName+"FOO.tif file=(.*"+well+"_"+site_str+"_"+ch+") sort");
					inFile = getImageID();
		
					// Make a zoomed-in crop montage
					// random crop:
          makeRectangle(randint(0, image_width - crop_side_length), randint(0,image_height - crop_side_length), crop_side_length, crop_side_length);//
					// center crop:
          //makeRectangle(image_width/2 - crop_side_length/2, image_height/2 - crop_side_length/2, crop_side_length, crop_side_length);
					run("Crop");
          run("Make Montage...", "columns="+site_side_length+" rows="+site_side_length+" scale=1");
					montage = getImageID();
					selectImage(inFile);
					close();
					selectImage(montage);
					saveAs("Tiff", savename);
					close();
				}
			}
		} 
		print("Building Plate montage...");
		montage_types = newArray("zoom");
		for(m=0;m<montage_types.length;++m) {
			montage_type = montage_types[m];
			savename = ""+pathName+"Plate_"+montage_type+"_"+ch+".tif"; 
			savename_jpg = ""+pathName+"Plate_"+montage_type+"_"+ch+".jpg";
      if( !File.exists( savename )  || !File.exists(savename_jpg) || recompute_all ) {
        run("Image Sequence...", "open="+pathName+"FOO.tif file=(.*Well_"+montage_type+".*"+ch+".tif*) sort");
				inFile = getImageID();
				// Unfortunately having a border prevents auto-contrast from working.
				run("Make Montage...", "columns=12 rows=8 scale=1");//0301CY
				montage = getImageID();
				selectImage(inFile);
				close();
				selectImage(montage);
				saveAs("Tiff",savename);
				run("Enhance Contrast", "saturated=0.35");
				print("Computing focus quality...");
				setBatchMode(false);
        run("Microscope Image Focus Quality", "originalimage="+savename+" createprobabilityimage=true overlaypatches=true solidpatches=false borderwidth=4");
				// Also save a smaller-file-size jpeg for upload/e-mail sharing.
				selectWindow("Plate_zoom_CY3-AGP.tif");
				saveAs("Jpeg", savename_jpg);
				close();
        
        // The probabilities is a stack with 11 slices, corresponding to probability of 1, 4, ..., 31 pixel blur.
        // We sum the probabilities corresponding to 1, 4 and 7 pixel blurs here, as the acceptable focus threshold.
				
        percentage_score(min_thresh, max_thresh);

			}
		}
    File.copy(savename, all_historical_outputs_path + replace(replace(savename,":/","_"),"/","_"));
    File.copy(savename_jpg, all_historical_outputs_path + replace(replace(savename_jpg,":/","_"),"/","_"));
		
	}
}	
setBatchMode(false); 
for(scan=0; scan<inNames.length; scan++) {
	inName = inNames[scan];
	print(inName);
	pathName = inName + montageFolderName;
	
	type = ".jpg"; // faster load, auto-scaled compared with .tif
	montage_types = newArray("zoom");
	
	for(m=0;m<montage_types.length;++m) {
		montage_type = montage_types[m];
      run("Image Sequence...", "open="+pathName+"foo.jpg file=("+montage_type+".*"+type+") sort");
	    rename(inName+" "+montage_type+" montage");
	}
}

print("Done");