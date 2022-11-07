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
- 'hERG_models' package contains all hERG trained models 
- 'nav1.5_models' package contains all Nav1.5 trained models 
- 'inducers' package contains hERG and Nav1.5 toxtree inference pipeline
- 'PaDEL' package contains PaDEL software to generate descriptors from canonical SMILES strings

## Prerequisites
To run the software, make sure you have the following dependencies and packages installed:

- Java 6 or higher
- Python 3.5 or higher
- scikit-learn 0.21.3

## Run software
### Two steps
##### Step one(Generate descriptors):

- Download Padel from: http://www.yapcwsoft.com/dd/padeldescriptor/ OR (In "PaDEL" folder, unzip the compressed file name "PaDEL-Descriptor" containing the software downloaded for you)
- After decompression, the folder will contain the ".jar" file of the software named (PaDEL-Descriptor.jar)
- Get the canonical SMILES of a molecular compound of your concern
- Paste the SMILES in a file with the extension ".smi". An example of the hERG external evaluation set is available in "PaDEL/input/hERG_external_set.smi".
	 
One can put as many SMILES strings to be processed as he wishes in the same file, each one in a row. Molecular compounds will then be processed sequentially.

- Now, in the "PaDEL" folder containing the jar file, run the following command to generate 1D and 2D descriptors (eaxmple of the hERG external evaluation set):
		java -jar PaDEL-Descriptor.jar -2d -retainorder -dir PaDEL/input/hERG/ -file output/hERG/descriptor_values.csv

- The descriptors of the the smile formats will be written in the file "output/hERG/descriptor_values.csv", each line coresponding to one molecular compund in the input file.
	
#### Step two(Make predictions):
- Make sure you have "Python 3.5 or higher" installed => (You may follw this link for Macos users: http://dioskurn.com/installing-scikit-learn-in-macos/ )

		(Check version by typing: $python -V)
- Make sure you have "Pip" installed

		(Check version by typing: $pip -V)
- Install scikit-learn by typing the following commands: 

		$pip install -U scikit-learn==0.21.3
- cd to "toxtree" folder
- Once everything is set up, run: 

		$python toxtree-py -help
- run the software for predictions as follow:

	=> For hERG liability predictions: 

		python toxtree.py -i output/hERG/descriptor_values.csv --hERG

	=> For Nav1.5 liability predictions: 

		python toxtree.py -i output/Nav1.5/descriptor_values.csv --Nav1.5
