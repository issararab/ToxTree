from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from math import sqrt
import pickle
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

### Load model for training dataset - IC50 < 1 microMol tagging criterion
filename = 'nav1.5_models/PIC50_6_imbalanced_best_model.sav'
loaded_model_PIC50_6_imbalanced = pickle.load(open(filename, 'rb'))

### Load model for training dataset - IC50 < 10 tagging criterion
# load the model from disk
filename = 'nav1.5_models/PIC50_5_balanced_best_model.sav'
loaded_model_PIC50_5_balanced = pickle.load(open(filename, 'rb'))

###Load the model for training dataset - IC50 < 30 tagging criterion
# load the model from disk
filename = 'nav1.5_models/PIC50_4o5_imbalanced_best_model.sav'
loaded_model_PIC50_4o5_imbalanced = pickle.load(open(filename, 'rb'))

### Load scaling and projection
# load the model from disk
filename = 'nav1.5_models/standard_scaling.sav'
sc = pickle.load(open(filename, 'rb'))
# load the model from disk
filename = 'nav1.5_models/PCA_transformation_model.sav'
pca = pickle.load(open(filename, 'rb'))

def toxTree(input_compound):
    if loaded_model_PIC50_6_imbalanced.predict(input_compound) == 1:
        return 3
    elif loaded_model_PIC50_5_balanced.predict(input_compound) == 1:
        return 2
    elif loaded_model_PIC50_4o5_imbalanced.predict(input_compound)  == 1:
        return 1
    else:
        return 0


def Nav_infer(file_name):
	#Load data
	test_data = pd.read_csv(file_name)
	relevant_features= ['nAcid','ALogP','ALogp2','AMR','apol','naAromAtom','nAromBond','nAtom','nHeavyAtom','nH','nC','nN','nO','nS','nF','nX','ATS0m','ATS1m','ATS2m','ATS3m','ATS4m','ATS5m','ATS6m','ATS7m','ATS8m','ATS0v','ATS1v','ATS2v','ATS3v','ATS4v','ATS5v','ATS6v','ATS7v','ATS8v','ATS0e','ATS1e','ATS2e','ATS3e','ATS4e','ATS5e','ATS6e','ATS7e','ATS8e','ATS0p','ATS1p','ATS2p','ATS3p','ATS4p','ATS5p','ATS6p','ATS7p','ATS8p','ATS0i','ATS1i','ATS2i','ATS3i','ATS4i','ATS5i','ATS6i','ATS7i','ATS8i','AATS0m','AATS1m','AATS2m','AATS3m','AATS4m','AATS5m','AATS6m','AATS7m','AATS8m','AATS0v','AATS1v','AATS2v','AATS3v','AATS4v','AATS5v','AATS6v','AATS7v','AATS8v','AATS0e','AATS1e','AATS2e','AATS3e','AATS4e','AATS5e','AATS6e','AATS7e','AATS8e','AATS0p','AATS1p','AATS2p','AATS3p','AATS4p','AATS5p','AATS6p','AATS7p','AATS8p','AATS0i','AATS1i','AATS2i','AATS3i','AATS4i','AATS5i','AATS6i','AATS7i','AATS8i','ATSC0c','ATSC1c','ATSC2c','ATSC3c','ATSC4c','ATSC5c','ATSC6c','ATSC7c','ATSC8c','ATSC0m','ATSC1m','ATSC2m','ATSC3m','ATSC4m','ATSC5m','ATSC6m','ATSC7m','ATSC8m','ATSC0v','ATSC1v','ATSC2v','ATSC3v','ATSC4v','ATSC5v','ATSC6v','ATSC7v','ATSC8v','ATSC0e','ATSC1e','ATSC2e','ATSC3e','ATSC4e','ATSC5e','ATSC6e','ATSC7e','ATSC8e','ATSC0p','ATSC1p','ATSC2p','ATSC3p','ATSC4p','ATSC5p','ATSC6p','ATSC7p','ATSC8p','ATSC0i','ATSC1i','ATSC2i','ATSC3i','ATSC4i','ATSC5i','ATSC6i','ATSC7i','ATSC8i','AATSC0c','AATSC1c','AATSC2c','AATSC3c','AATSC4c','AATSC5c','AATSC6c','AATSC7c','AATSC8c','AATSC0m','AATSC1m','AATSC2m','AATSC3m','AATSC4m','AATSC5m','AATSC6m','AATSC7m','AATSC8m','AATSC0v','AATSC1v','AATSC2v','AATSC3v','AATSC4v','AATSC5v','AATSC6v','AATSC7v','AATSC8v','AATSC0e','AATSC1e','AATSC2e','AATSC3e','AATSC4e','AATSC5e','AATSC6e','AATSC7e','AATSC8e','AATSC0p','AATSC1p','AATSC2p','AATSC3p','AATSC4p','AATSC5p','AATSC6p','AATSC7p','AATSC8p','AATSC0i','AATSC1i','AATSC2i','AATSC3i','AATSC4i','AATSC5i','AATSC6i','AATSC7i','AATSC8i','MATS1c','MATS2c','MATS3c','MATS4c','MATS5c','MATS6c','MATS7c','MATS8c','MATS1m','MATS2m','MATS3m','MATS4m','MATS5m','MATS6m','MATS7m','MATS8m','MATS1v','MATS2v','MATS3v','MATS4v','MATS5v','MATS6v','MATS7v','MATS8v','MATS1e','MATS2e','MATS3e','MATS4e','MATS5e','MATS6e','MATS7e','MATS8e','MATS1p','MATS2p','MATS3p','MATS4p','MATS5p','MATS6p','MATS7p','MATS8p','MATS1i','MATS2i','MATS3i','MATS4i','MATS5i','MATS6i','MATS7i','MATS8i','MATS1s','MATS2s','MATS3s','MATS4s','MATS5s','MATS6s','MATS7s','MATS8s','GATS1c','GATS2c','GATS3c','GATS4c','GATS5c','GATS6c','GATS7c','GATS8c','GATS1m','GATS2m','GATS3m','GATS4m','GATS5m','GATS6m','GATS7m','GATS8m','GATS1v','GATS2v','GATS3v','GATS4v','GATS5v','GATS6v','GATS7v','GATS8v','GATS1e','GATS2e','GATS3e','GATS4e','GATS5e','GATS6e','GATS7e','GATS8e','GATS1p','GATS2p','GATS3p','GATS4p','GATS5p','GATS6p','GATS7p','GATS8p','GATS1i','GATS2i','GATS3i','GATS4i','GATS5i','GATS6i','GATS7i','GATS8i','GATS1s','GATS2s','GATS3s','GATS4s','GATS5s','GATS6s','GATS7s','GATS8s','nBase','BCUTw-1l','BCUTw-1h','BCUTc-1l','BCUTc-1h','BCUTp-1l','BCUTp-1h','nBonds','nBonds2','nBondsS','nBondsS2','nBondsS3','nBondsD','nBondsD2','nBondsM','bpol','C1SP2','C2SP2','C3SP2','C1SP3','C2SP3','C3SP3','Sv','Sse','Spe','Sare','Sp','Si','Mv','Mse','Mpe','Mare','Mp','Mi','CrippenLogP','CrippenMR','ECCEN','fragC','nHBAcc','nHBAcc2','nHBAcc3','nHBAcc_Lipinski','nHBDon','nHBDon_Lipinski','HybRatio','IC0','IC1','IC2','IC3','IC4','IC5','TIC0','TIC1','TIC2','TIC3','TIC4','TIC5','SIC0','SIC1','SIC2','SIC3','SIC4','SIC5','CIC0','CIC1','CIC2','CIC3','CIC4','CIC5','BIC0','BIC1','BIC2','BIC3','BIC4','BIC5','MIC0','MIC1','MIC2','MIC3','MIC4','MIC5','ZMIC0','ZMIC1','ZMIC2','ZMIC3','ZMIC4','ZMIC5','Kier1','Kier2','Kier3','nAtomLC','nAtomP','nAtomLAC','MLogP','McGowan_Volume','MDEC-11','MDEC-12','MDEC-13','MDEC-22','MDEC-23','MDEC-24','MDEC-33','MDEC-34','MDEO-11','MDEO-12','MDEN-22','MDEN-23','MDEN-33','MLFER_A','MLFER_BH','MLFER_BO','MLFER_S','MLFER_E','MLFER_L','MPC2','MPC3','MPC4','MPC5','MPC6','MPC7','MPC8','MPC9','MPC10','TPC','piPC1','piPC2','piPC3','piPC4','piPC5','piPC6','piPC7','piPC8','piPC9','piPC10','TpiPC','R_TpiPCTPC','PetitjeanNumber','nRing','n5Ring','n6Ring','nFRing','nF9Ring','nF10Ring','nTRing','nT5Ring','nT6Ring','nT9Ring','nT10Ring','nHeteroRing','n5HeteroRing','n6HeteroRing','nF9HeteroRing','nF10HeteroRing','nT5HeteroRing','nT6HeteroRing','nT9HeteroRing','nT10HeteroRing','nRotB','RotBFrac','nRotBt','RotBtFrac','LipinskiFailures','topoRadius','topoDiameter','topoShape','GGI1','GGI2','GGI3','GGI4','GGI5','GGI6','GGI7','GGI8','GGI9','GGI10','JGI1','JGI2','JGI3','JGI4','JGI5','JGI6','JGI7','JGI8','JGI9','JGI10','JGT','SpMax_D','SpDiam_D','SpAD_D','SpMAD_D','EE_D','VE1_D','VE2_D','VE3_D','VR1_D','VR2_D','VR3_D','TopoPSA','VABC','VAdjMat','MWC2','MWC3','MWC4','MWC5','MWC6','MWC7','MWC8','MWC9','MWC10','TWC','SRW2','SRW4','SRW5','SRW6','SRW7','SRW8','SRW9','SRW10','TSRW','MW','AMW','WPATH','WPOL','XLogP','Zagreb']
	for col in test_data.columns:
		if col not in relevant_features:
			test_data.drop(columns=[col], inplace=True)
	
	#Scale data
	XX_test = sc.transform(test_data)
	#Transform data
	X_test = pca.transform(XX_test)
	print("|++++++++++++++++++++++++++++++++++++++++++++++|")
	print("|++++ Legend: Nav1.5 liability predictions ++++|")
	print("|++++++++++++++++++++++++++++++++++++++++++++++|")
	print("|++++++ IC50 is reproted in microMolar ++++++++|")
	print("|++++++++++++++++++++++++++++++++++++++++++++++|")
	print("|    Strongblocker    =>  0<IC50<=1            |")
	print("|    Moderate blocker => 1<IC50<=10            |")
	print("|    Weakblocker      => 10<IC50<=30           |")
	print("|    Non-blocker      => 30<IC50               |")
	print("+++++++++++++++++++++++++++++++++++++++++++++++|")
	print()
	predictions = []
	for i in range(X_test.shape[0]):
		predc = toxTree(np.array([X_test[i]]))
		if predc == 3:
			print("MC"+str(i+1)+": Strong blocker")
			predictions.append('Strong blocker')
		elif predc == 2:
			print("MC"+str(i+1)+": Moderate blocker")
			predictions.append('Moderate blocker')
		elif predc == 1:
			print("MC"+str(i+1)+": Weak blocker")
			predictions.append('Weak blocker')
		elif predc == 0:
			print("MC"+str(i+1)+": Non-blocker")
			predictions.append('Non-blocker')
		else:
			print("MC"+str(i+1)+": Indetermined")
			predictions.append('Indetermined')
	return predictions