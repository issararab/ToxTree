from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

### Load model for training dataset - IC50 < 1 microMol tagging criterion
# load the model from disk
filename = 'hERG_models/PIC50_6_balanced_best_model.sav'
loaded_model_PIC50_6_balanced = pickle.load(open(filename, 'rb'))

### Load model for training dataset - IC50 < 10 tagging criterion
# load the model from disk
filename = 'hERG_models/PIC50_5_balanced_best_model.sav'
loaded_model_PIC50_5_balanced = pickle.load(open(filename, 'rb'))

###Load the model for training dataset - IC50 < 30 tagging criterion (Combined)
# load the model from disk
filename = 'hERG_models/PIC50_4o5_best_model.sav'
loaded_model_PIC50_4o5_imbalanced = pickle.load(open(filename, 'rb'))

# load the model from disk
filename = 'hERG_models/PIC50_4o5_balanced_best_model.sav'
loaded_model_PIC50_4o5_balanced = pickle.load(open(filename, 'rb'))

### Load scaling
# load the model from disk
filename = 'hERG_models/normalization.sav'
sc = pickle.load(open(filename, 'rb'))

def toxTree(input_compound):
    if loaded_model_PIC50_6_balanced.predict(input_compound) == 1:
        return 3
    elif loaded_model_PIC50_5_balanced.predict(input_compound) == 1:
        return 2
    elif loaded_model_PIC50_4o5_imbalanced.predict(input_compound)  == loaded_model_PIC50_4o5_balanced.predict(input_compound):
        if loaded_model_PIC50_4o5_balanced.predict(input_compound) == 1:
            return 1
        else:
            return 0
    elif max(loaded_model_PIC50_4o5_imbalanced.predict_proba(input_compound)[0])  > max(loaded_model_PIC50_4o5_balanced.predict_proba(input_compound)[0]):
        return loaded_model_PIC50_4o5_imbalanced.predict(input_compound)
    elif max(loaded_model_PIC50_4o5_imbalanced.predict_proba(input_compound)[0])  < max(loaded_model_PIC50_4o5_balanced.predict_proba(input_compound)[0]):
        return loaded_model_PIC50_4o5_balanced.predict(input_compound)
    else:
        return -1


def hERG_infer(file_name):
	#Load data
	test_data = pd.read_csv(file_name)
	relevant_features= ['ALogP','AATS7p','AATS3i','AATS6i','ATSC7m','ATSC1i','AATSC0p','AATSC1i','MATS1c','MATS2i','GATS2m','GATS1p','GATS2p','GATS1i','GATS2i','GATS2s','BCUTw-1l','BCUTc-1l','nBondsS3','SpMax1_Bhv','SpMin1_Bhi','VCH-7','SC-5','Mi','CrippenLogP','VE1_Dt','SwHBa','SaaCH','minHBint2','minHBint4','minHBint5','mindsCH','minaaaC','minsssN','mindsssP','maxHBint3','maxHssNH','maxHCHnX','maxHCsatu','maxaaCH','maxaasC','maxdO','maxaaO','maxsF','maxsSH','LipoaffinityIndex','ETA_dAlpha_A','ETA_BetaP_s','ETA_BetaP_ns','ETA_BetaP_ns_d','nHBAcc','IC2','MDEO-11','MDEO-12','MDEN-13','MLFER_BH','MLFER_BO','TpiPC','R_TpiPCTPC','n6HeteroRing','nT6HeteroRing','TopoPSA','XLogP','nAcid','ALogP','AATS4p','ATS0p','AATS3i','ATSC5p','AATSC6e','MATS7m','MATS5v','MATS2s','GATS1p','GATS3p','BCUTc-1l','SpMax1_Bhv','SpMin1_Bhi','SpMin2_Bhi','SCH-5','SC-6','SPC-5','CrippenLogP','VE1_Dt','SwHBd','SwHBa','SHaaCH','SHAvin','SaaCH','SdssC','SdsN','SdO','SdsssP','minHBint4','minHBint5','mindsCH','minsNH2','minsOH','minssS','minddssS','maxwHBa','maxHBint2','maxHBint3','maxHssNH','maxaaCH','maxaasC','maxsNH2','maxaaO','LipoaffinityIndex','ETA_dEpsilon_B','ETA_dBetaP','nHBAcc','BIC2','MDEO-11','MDEO-12','MDEO-22','MDEN-13','MLFER_A','MLFER_BO','TpiPC','R_TpiPCTPC','XLogP','WTPT-4','nAcid','ALogP','AATS1p','ATSC5p','AATSC0c','AATSC4c','AATSC5c','AATSC5v','AATSC6e','AATSC0p','MATS1e','MATS2s','MATS4s','MATS8s','GATS8c','GATS5m','GATS8p','GATS3s','nBase','BCUTw-1l','BCUTc-1l','SpMax1_Bhv','SpMax7_Bhv','SpMin2_Bhe','SpMax2_Bhi','SpMin2_Bhs','C2SP2','VCH-7','VC-6','CrippenLogP','VE1_Dt','ndssC','SwHBa','SHBint5','SHBint10','SHaaCH','SaaCH','SaasC','minHBint3','minHBint9','minHsOH','minssCH2','minaaaC','minsNH2','minsssN','mindsssP','minssS','minsCl','maxHBint2','maxHBint5','maxHsOH','maxdsCH','maxaaCH','maxaasC','hmin','LipoaffinityIndex','ETA_dEpsilon_D','ETA_Shape_Y','nHBAcc','SIC1','BIC1','BIC2','MDEC-22','MDEO-11','MDEO-22','MLFER_A','nT4Ring','JGI6','TopoPSA','WTPT-4','XLogP','nAcid','ALogP','ALogp2','AATS3i','AATS5s','ATSC3v','ATSC4v','ATSC5v','ATSC6e','ATSC6i','AATSC4c','AATSC6e','AATSC0p','AATSC4s','AATSC5s','MATS1c','MATS8s','GATS3e','GATS1p','GATS8i','GATS3s','GATS5s','GATS6s','VR3_DzZ','VR1_Dzm','VE3_Dzi','VR2_Dzs','nBase','BCUTw-1l','SpMin2_Bhm','SpMax2_Bhv','SpMax7_Bhv','SpMax2_Bhp','SpMin3_Bhs','SpMin8_Bhs','C2SP2','ASP-3','AVP-4','CrippenLogP','SHBint3','SHBint5','SHBint9','SHsOH','SHaaCH','SHCsatu','SdsCH','SaaCH','SsNH2','SsssN','SssssNp','SdO','SsOm','SssS','minHBd','minHBint4','minHBint5','minHsOH','minHaaNH','minHCHnX','mindssC','minaaaC','minsssN','minsCl','maxHBint8','maxHBint10','maxaasC','maxsNH2','maxsOm','hmin','LipoaffinityIndex','ETA_Shape_Y','ETA_BetaP_s','nHBAcc','SIC2','BIC0','BIC3','ZMIC2','Kier3','MDEC-22','MDEN-23','MLFER_A','nFRing','nF6HeteroRing','nRotBt','JGI2','JGI3','JGI7','VR1_D','WTPT-2','WTPT-5','XLogP']
	for col in test_data.columns:
		if col not in relevant_features:
			test_data.drop(columns=[col], inplace=True)
	
	#Scale data
	X_test = sc.transform(test_data)
	
	print("|++++++++++++++++++++++++++++++++++++++++++++++|")
	print("|+++++ Legend: hERG liability predictions +++++|")
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
	