# MusicRecommender
lastFM data used for recommendation of new interprets.

This is software developed under the Apache Licence.
You are free to use it as you see fit.

general Usage: Specify some artists you like in the <bands> global variable.
Running the script will generate <numberRecommendations> recommendations and display them
in the pdf file <outfilename>

if you get the following error:
AttributeError: 'FigureManagerGTK3Cairo' object has no attribute 'canvas'
ignore it, its a bug in matplotlib waiting to be fixed. It does not harm the functionality.

The program uses the lastFM data set. A small portion (the ~3.5k artists with most listeners)
has been distilled into the dMat.p file, but in order to get more artists it is advised to build it yourself.
To do this Just do the following:
grab the dataset from this source:
http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html
extract the content into the folder
delete the existing dMat.p
set the minListeners to a value of your choice (200 listeners correspondes to 10k artists and about 1 Gig Memory with a O(n^2) dependency on the artists)
run the script
enjoy ;)

