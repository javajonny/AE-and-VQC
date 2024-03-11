﻿* Encoding: UTF-8.
DATASET ACTIVATE DataSet11.
SORT CASES  BY Model.
SPLIT FILE LAYERED BY Model.

EXAMINE VARIABLES=Accuracy
  /PLOT BOXPLOT STEMLEAF NPPLOT
  /COMPARE GROUPS
  /STATISTICS DESCRIPTIVES EXTREME
  /CINTERVAL 95
  /MISSING LISTWISE
  /NOTOTAL.


SPLIT FILE OFF.


*Nonparametric Tests: Independent Samples. 
NPTESTS 
  /INDEPENDENT TEST (Accuracy) GROUP (Model) KRUSKAL_WALLIS(COMPARE=PAIRWISE) 
  /MISSING SCOPE=ANALYSIS USERMISSING=EXCLUDE
  /CRITERIA ALPHA=0.05  CILEVEL=95.