#############
# ORI_ALIGN #
#############


For each point separately:
	- all images across time with classification "worm" are loaded
	- they are aligned to relative to each other, such that the output is a timeseries for a single individual where all images are facing the same way. At this stage, where the images are facing is unknown, just that they all face the same way.
	- the 'relatively' aligned timeseries is then checked against an atlas. The atlas acts as a datum to make a decision on an absolute orientation, with the goal being to flip the timeseries such that the timeseries and the atlas face the same way.
	- because the orientation of the atlas is known (head LEFT, vulva UP), the orientations of the original str images can be inferred and recorded
	- these orientations are then output as a csv

To run:
	- set the parameters at the top of the ori_align.py script
	- take care to pick the correct atlas. each atlas is specific to the type of microscopy image (e.g. GFP body, or mCherry pharynx)
		- the CHANNEL settings refer to the channel within the image files, NOT the channels of the microscopy experiment
			- i.e. if you were to open an image from analysis/ch2_raw_str folder, although the experimental channel is 2, the channel in the actual image is 0 since those specific images contain only one channel
		- the RAW_STR_CHANNEL and ATLAS_CHANNEL need to be of the same type of microscopy experiment, e.g. both be mCherry germline
		- this may or not be the same integer
	- then run the "run_ori_prediction.slurm" script

Output csv:
	- Time
	- Point
	- head: L or R label for where the head in raw_str image is facing
	- vulva: U or D label for where the vulva in raw_str image is/would be facing
		- not reliable because of the worms' high rotational symmetry - movies made may have random UD flips
		- and not reliable because the worm could have been imaged halfway in the UD transition - movies will have a smooth transition from U to D
		- the vulva may not actually be visible on the images, its just the inferred location and is used to disambiguate the up-down axis
			- e.g. pharynx images, or male worms
	- head_confidence: ratio of images in the 'relatively-aligned timeseries' that agreed to the final orientation to the atlas datum
	- shift_ax0/1: the translational shifts that best align consecutive images. Required for movie making


####################
# MOVIE_GENERATION #
####################

For each point separately:
	- all images across time with classification "worm" are loaded
	- the "head", "vulva" orientations from ori_align.py are used to flip each image to face LEFT and UP
	- the shift_ax0/1 are used to translate the images to reduce jumps and hence make the movie have less motion jitter

To run:
	- set the parameters at the top of the movie_generation.py script
	- STR_IMG_COLS can be used to specify which image directories to process
		- ["analysis/ch2_raw_str", "analysis/ch2_raw_seg_str"] will only process those directories
		- "All" will process all directories that end with "_str"
	- ALIGNMENT sets how the frames are placed in the movie canvas along the worm's head->tail axis
		- "center" aligns the frames on their middle, so the worm appears to grow in both directions (default, previous behaviour)
		- "left" pins the head end to the left edge, so all the growth happens towards the right
		- either way the frames stay registered to each other via shift_ax0/1, and the dorsoventral axis stays centered
	- WHICH_POINTS can be used to set which points to process
		- range(1, 101) will process points from 1 to 100 (python upper bound is exclusive, so up to and not including 101)
		- [25, 42, 50] will process only points 25, 42, 50
		- "All" will process all points
	- then run the "run_movie_generation.slurm" script

Output:
	- movies for each point, for each STR_IMG_COL
	- e.g.:
		- MOVIE_DIR/ch2_raw_str_movies
		- MOVIE_DIR/ch2_raw_seg_str_movies
	- the subdirectories are generated automatically for each STR_IMG type, only MOVIE_DIR needs to be specified
	