* Encoding: UTF-8.

DATASET ACTIVATE DataSet1.
SORT CASES  BY Dataset.
SPLIT FILE LAYERED BY Dataset.

EXAMINE VARIABLES=VQC_angle_testing_list VQC_amplitude_testing_list dressed_quantum_testing_list 
    sequent_quantum_testing_list NN_with_compressed_input_testing_list 
    NN_with_original_input_testing_list
  /PLOT BOXPLOT STEMLEAF NPPLOT
  /COMPARE GROUPS
  /STATISTICS DESCRIPTIVES EXTREME
  /CINTERVAL 95
  /MISSING LISTWISE
  /NOTOTAL.


GLM VQC_angle_testing_list VQC_amplitude_testing_list dressed_quantum_testing_list 
    sequent_quantum_testing_list NN_with_compressed_input_testing_list 
    NN_with_original_input_testing_list
  /WSFACTOR=Modelle 6 Polynomial 
  /METHOD=SSTYPE(3)
  /EMMEANS=TABLES(Modelle) COMPARE ADJ(BONFERRONI)
  /CRITERIA=ALPHA(.05)
  /WSDESIGN=Modelle.


*Nonparametric Tests: Related Samples. 
NPTESTS 
  /RELATED TEST(VQC_angle_testing_list VQC_amplitude_testing_list dressed_quantum_testing_list 
    sequent_quantum_testing_list NN_with_compressed_input_testing_list 
    NN_with_original_input_testing_list) FRIEDMAN(COMPARE=PAIRWISE) 
  /MISSING SCOPE=ANALYSIS USERMISSING=EXCLUDE
  /CRITERIA ALPHA=0.05  CILEVEL=95.


