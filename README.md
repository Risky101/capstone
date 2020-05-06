# capstone

The learn.py file pulls information from the CSV and makes a prediciton by means of a random forest algorithm.

The fraudlearn.csv file needs to be downloaded to your machine and the pd.read_csv (currently on line 9) of learn.py needs 
to have the directory information in place of "----ENTER CSV HERE----" in order to read it.

scratch.py is the code in progress to connect to our postgres database and pull larger datasets into the algorithm to increase
prediction accuracy.  It has issues with transformation of the data and thus does not work in its current state.

The frontend system is developed through Tableau. You must use the LEIE Data Set csv file to access it.
