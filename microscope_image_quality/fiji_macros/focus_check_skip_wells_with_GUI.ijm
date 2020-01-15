// This script generates whole-plate montages per channel for inspecting data quality.
// Requires that the images already live in a single directory.

/* 
*  Due to the nature of the FIJI focus plugin, we oversample the focus score when
*  we include blank tiles. We use the final precentage score as a means to calculate
*  the actual focus score sans the blank tiles.
*/

// Use forward slash "/" and end the path with a "/" too.
//newName = "E:/20190717_194041_709/";


/*
 * Creates and shows Dialog for easier use of macro
 * 
 * Args: Drive Name (ex. E)
 * 		 Folder name (ex. 20190717_194041_709)
 * 		 Total number of tiles - # of tiles you'd like to analyze. Does not need to match number of tiles acquired but cannot exceed
 * 		 Throw out blank tiles? - Check box if you'd like to use the random_crop_site function, 
 * 		 						  which "throws out" blank tiles if they do not meet standards
 * 		 
 */
 
Dialog.create("Focus Quality Check");
Dialog.addString("Date of Analysis: ", "YYYY");
Dialog.addToSameRow();
Dialog.addString(" - ", "MM");
Dialog.addToSameRow();
Dialog.addString(" - ", "DD");
Dialog.addString("Data Directory: ", "E",1);
Dialog.addToSameRow();
Dialog.addString(": / ", "20190717_194041_709",17);
Dialog.addToSameRow();
Dialog.addMessage("/");
Dialog.addNumber("Total Number of Sites:", 76);
Dialog.addCheckbox("Throw out blank tiles", false);
Dialog.show();

// Get information from GUI and assign as constants
year = Dialog.getString();
month = Dialog.getString();
day = Dialog.getString();
analysis_date = year+month+day;
ssd = Dialog.getString();
folder_name = Dialog.getString();
newName = ssd + ":/" + folder_name + "/";
num_of_sites = Dialog.getNumber();
no_blank_tiles = Dialog.getCheckbox();

montageFolderName = analysis_date + "_Montages";

if (no_blank_tiles == true) {
	montage_fileName = d2s(num_of_sites,0) + "-Sites_NoBlankTiles_Montage";
} else {
	montage_fileName = d2s(num_of_sites,0) + "-Sites_Montage";
}

inNames = newArray(newName);
random_iterations = 200;
recompute_all=false;
all_historical_outputs_path = "E:/quality_montages/"; // This shouldn't be edited.

rows = newArray("A","B","C","D","E","F","G","H");
columns = newArray("01","02","03","04","05","06","07","08","09","10","11","12");

function append(arr, value) {
  /*
   * Appends value to existing array
   * 
   * Args: arr - array
   * 		   value - element to add to array
   * 		 
   */
	arr2 = newArray(arr.length+1);
	for (i=0; i<arr.length; i++) {
		arr2[i] = arr[i];
	}
    arr2[arr.length] = value;
    return arr2;
}

function array_contains(arr, element) {
   /*
   * Checks if a given array contains a given element
   * 
   * Args: arr - array
   * 		   element - element
   * 		 
   */
	for (i=0; i<arr.length; i++) {
		if (element == arr[i]) {
			return true;
		} if (i == arr.length-1 && element != arr[i]) {
			return false;
		}
	}
}

//Select wells to analyze
wells = newArray;
Dialog.create("Well Chooser");

for (i=0; i<rows.length; i++) {
	for (j=0; j<columns.length; j++) {
		well = rows[i]+columns[j];
		wells = append(wells, well);
	}
}

default_list = newArray(96);
for (k=0; k<default_list.length; k++) {
	default_list[k] = true;
}

Dialog.addCheckboxGroup(8,12, wells, default_list);
Dialog.show();

skip_wells = newArray;

for (i=0; i<default_list.length; i++) {
	default_list[i] = Dialog.getCheckbox();
	if (default_list[i] == false) {
		skip_wells = append(skip_wells, wells[i]);
	}
}

//This file is used in place of skipped wells, allowing a plate montage
blank_file = "C:/Users/Nikon/Desktop/BlankTile_DONOTDELETE.tif";

channels = newArray("CY3-AGP");
maxChannels = channels.length;

min_thresh = 0.5000;
max_thresh = 1000000000000000000000000000000.0000;

if (num_of_sites < 4) {
	num_random_sites = 1;
	site_side_length = 1;
} else {
	num_random_sites = 4;
	site_side_length = 2;
}
getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
random_seed = second;
start_site = 0;
end_site = num_of_sites - 1;

image_height = 2960; //pixels
image_width = 5056; // pixels
crop_side_length = 84*3; // pixels

function make_dir(path_name, make_new) {
	/*
	 * Checks to see if directory exists and creates new one if doesn't
	 * 
	 * Args: path_name - directory name and path
	 * 
	 * Returns: New directory if does not exist, otherwise prints "Directory already exists!"
	 */
	new_path_base = substring(path_name, 0, lengthOf(path_name)-1);
	new_path = new_path_base + "/";
	counter = 0;
	if (make_new == false) {
		if (!File.isDirectory(new_path)) {
			File.makeDirectory(new_path);
		} else {
			print("Directory already exists!");
		}
	} else {
		while (File.isDirectory(new_path)) {
			counter++;
			new_path = new_path_base + "-" + counter + "/";
		}
		File.makeDirectory(new_path);
		return new_path;
	}
}

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

function make_random_rectangle(imageID) {
	counter = 0;
	selectImage(imageID);
	makeRectangle(randint(0, image_width - crop_side_length), randint(0,image_height - crop_side_length), crop_side_length, crop_side_length); //random crop
	getRawStatistics(nPixels, mean, min, max, std, histogram);
	while(min/max > 0.1 && mean < 1000) {
		if (counter == random_iterations) {
			break;
		} else {
			counter++;
			makeRectangle(randint(0, image_width - crop_side_length), randint(0,image_height - crop_side_length), crop_side_length, crop_side_length); //random crop
			getRawStatistics(nPixels, mean, min, max, std, histogram);
		}
	}
	run("Crop");
	run("Enhance Contrast", "saturated=0.35");
	//print("Num of randomly selected rectangles: " + counter);
}

function random_site_crop(well, channel, savepath, fileName,  blank_tiles, start, end, n) {
	/*
	 * Creates random ROI per well and measures intensity — if intensity is less than 90%, reselect a different random ROI
	 * 
	 * Args: well - well of interest
	 * 		 channel - channel/OC of interest
	 * 		 savepath - path of file to be saved
	 * 		 fileName - data directory (ex. E:/20190717_194041_709/)
	 * 		 counter - # of times function has recursed. If over 200, take current ROI and analyze
	 * 		 
	 * Returns: randomly chosen ROI that meets standards and creates montage of all wells
	 */
	
	if (blank_tiles == true) {
		if (num_of_sites == 1) {
			site_str = "Site000";
			run("Image Sequence...", "open="+fileName+"FOO.tif file=(.*"+well+"_"+site_str+"_"+channel+") sort");
			inFile = getImageID();
			make_random_rectangle(inFile);
		} else {
			for (i=0; i < num_random_sites; i++) {
				a = randint(start+i*(end-start)/n, start+(i+1)*(end-start)/n);
				while (lengthOf(""+a) < 3) {
					a = "0" + a;
				}
				site_str = "Site" + a;
				run("Image Sequence...", "open="+fileName+"FOO.tif file=(.*"+well+"_"+site_str+"_"+channel+") sort");
				inFile = getImageID();
				make_random_rectangle(inFile);
			}
		}
	} else {
		if (num_of_sites == 1) {
			site_str = "Site000";
		} else {
			site_str = "Site"+random_site_str(start_site, end_site, num_random_sites);
		}
		run("Image Sequence...", "open="+fileName+"FOO.tif file=(.*"+well+"_"+site_str+"_"+channel+") sort");
		inFile = getImageID();
		makeRectangle(randint(0, image_width - crop_side_length), randint(0,image_height - crop_side_length), crop_side_length, crop_side_length); //random crop
		run("Crop");
		run("Enhance Contrast", "saturated=0.35");
	}
	if (num_of_sites < 4){
		image_ID = inFile;
	} else {
		if (blank_tiles == true) {
			run("Images to Stack");
			inFile = getImageID();
		}
		run("Make Montage...", "columns="+site_side_length+" rows="+site_side_length+" scale=1");
		image_ID = getImageID();
		selectImage(inFile);
		close();
	}
	selectImage(image_ID);
	saveAs("Tiff",savepath);
	close();
}

function percentage_score(path_name, save_name, save_name_jpg, montageType, channel, min_threshold, max_threshold) {
	/*
	 * Calculates a percentage score for focus quality
	 * 
	 * Args: minimum and maximum values for thresholding
	 * 
	 * Returns: percentage score of focus quality for plate
	*/
	run("Image Sequence...", "open="+path_name+"FOO.tif file=(.*Well_"+montageType+".*"+channel+".tif*) sort");
	inFile = getImageID();
	// Unfortunately having a border prevents auto-contrast from working.
	run("Make Montage...", "columns=12 rows=8 scale=1");//0301CY
	montage = getImageID();
	selectImage(inFile);
	close();
	selectImage(montage);
	saveAs("Tiff",save_name);
	run("Enhance Contrast", "saturated=0.35");
	print("Computing focus quality...");
	setBatchMode(false);
	run("Microscope Image Focus Quality", "originalimage="+save_name+" createprobabilityimage=true overlaypatches=true solidpatches=false borderwidth=4");
	// Also save a smaller-file-size jpeg for upload/e-mail sharing.
	selectWindow(montage_fileName + ".tif");
	saveAs("Jpeg", save_name_jpg);
	close();
	// The probabilities is a stack with 11 slices, corresponding to probability of 1, 4, ..., 31 pixel blur.
	// We sum the probabilities corresponding to 1, 4 and 7 pixel blurs here, as the acceptable focus threshold.
	selectWindow("Probabilities");
	run("Make Substack...", "channels=1-3");
	run("Z Project...", "projection=[Sum Slices]");
	selectWindow("SUM_Probabilities-1");
	setAutoThreshold("Default dark");
	setThreshold(min_threshold, max_threshold);
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

make_dir(all_historical_outputs_path, false);

run("Close All");
for(scan=0; scan<inNames.length; scan++) {
	inName = inNames[scan];
	print(inName);
	pathName = make_dir(inName + montageFolderName + "/", true);
	random("seed",random_seed);
	
	setBatchMode(true); //so we don't flash all the images 

	print("starting Zoom montage creation...");
	for(c=0;c<maxChannels;++c) {
		endXYloop = false;
		ch = channels[c];
		//counter = 0;
		for (x=0;x<rows.length;++x) {
			print("Processing row: " +rows[x] );
			for(y=0;y<columns.length;++y) {
				well = rows[x] + columns[y];
				skip = array_contains(skip_wells, well);
				savename = ""+pathName+"Well_zoom_"+well+"_"+ch+".tif";
				if (skip == true) {
					run("Image Sequence...", "open="+blank_file+" file=BlankTile sort");
					image_ID = getImageID();
					selectImage(image_ID);
					saveAs("Tiff",savename);
					close();
				} else {
					if( !File.exists( savename ) | recompute_all ) {
						random_site_crop(well, ch, savename, inName, no_blank_tiles, start_site, end_site, num_of_sites);
					}
				}
			}
		}
		
		print("Building Plate montage...");
		montage_types = newArray("zoom");
		for(m=0;m<montage_types.length;++m) {
			montage_type = montage_types[m];
			savename = ""+pathName+montage_fileName+".tif"; 
			savename_jpg = ""+pathName+montage_fileName+".jpg";
			if( !File.exists( savename )  || !File.exists(savename_jpg) || recompute_all ) {
				percentage_score(pathName, savename, savename_jpg, montage_type, ch, min_thresh, max_thresh);
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
	pathName = inName + montageFolderName + "/";
	type = ".jpg"; // faster load, auto-scaled compared with .tif
	run("Image Sequence...", "open="+pathName+"foo.jpg file=("+type+") sort");
	rename(inName+" "+montage_fileName);
}

print("Done");