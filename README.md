# ToxTree
ToxTree is a machine learning based model to predict hERG and Nav1.5 cardiotoxicity of a molecular compound at different potency levels:

The model predicts toxicities in one of the four classes(Potencies are in μM):
 - Strong blocker: 0<IC50<=1
 - Moderate blocker: 1<IC50<=10
 - Week blocker: 10<IC50<=30
 - Non-blocker: 30<IC50


:exclamation:Download first the whole toxtree package and follw the steps bellow.


## Package structure

- 'toxtree' package is the main package of the software
- 'toxtree.py' is the main entry point to the software
- 'hERG_models' package contains all hERG train models 
- 'nav1.5_models' package contains all Nav1.5 train models 
- 'inducers' package contains hERG and Nav1.5 toxtree inference pipeline (makes the final decisions)
- 'PaDEL' package contains PaDEL software to generate descriptors from canonical SMILE format
- 'test_bed' contains some csv files containining descriptors of molecular compounds for test purposes. Can also be used
for future predictions by just copying the csv descriptors file for a molecular compound of your concern and use the 
command as described below

## Prerequisites
To run the software, make sure you have:

- Java 6 or higher
- Python 3.5 or higher

## Run software
### Two steps
##### Step one(Generate descriptors):

- Install Java 6 or higher 
- Download Padel from: http://www.yapcwsoft.com/dd/padeldescriptor/ OR (In "PaDEL" folder, unzip the compressed file name "PaDEL-Descriptor" containing the software downloaded for you)
- After decompression, the folder will contain the ".jar" file of the software named (PaDEL-Descriptor.jar)
- Now, get the canonical smile format of a molecular compound of your concern
- Paste the smile line in a file having the extension (.smi). An exeample is pasted in the file with path "PaDEL/input/Sample_Canonical_SMILES.smi".
	 
One can put as many smile formats to be processed as he wishes in the same file, each one in a row. Molecular compounds will then be processed sequentially.

- Now, in the "PaDEL" folder containing the jar file, run the following command to generate 1D and 2D descriptors:
		java -jar PaDEL-Descriptor.jar -2d -retainorder -dir input/ -file output/descriptor_values.scv
- You may have to add "-maxruntime 100000" to the command in case of a very long smile format of a molecular compound that will take too much time to converge for final values.
- The descriptors of the the smile formats will be output in the file "output/descriptor_values.scv", each line coresponding to one molecular compund in the input.
	
#### Step two(Make predictions):
- Make sure you have "Python 3.5 or higher" installed => (You may follw this link for Macos users: http://dioskurn.com/installing-scikit-learn-in-macos/ )

		(Check version by typing: $python -V)
- Make sure you have "Pip3" installed

		(Check version by typing: $pip3 -V)
- Install scikit-learn by typing the following commands: 

		$pip install -U scikit-learn
- cd to "toxtree" folder
- Once everything is set up, run: 

		$python toxtree-py -help
- run the software for predictions as follow:
=> For hERG liability predictions: 

		python toxtree.py -i test_bed/result.csv --hERG
=> For nAv1.5 liability predictions: 

		python toxtree.py -i test_bed/result.csv --Nav1.5
- If you want to output the results in a file rather than the console append "> <file_name>" to the end of the command
=> Example:

		python toxtree.py -i test_bed/herg_test.csv --hERG > myPreds.txt
		
Enjoy your predictions :)
