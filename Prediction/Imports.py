{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# imports.py\n",
    "\n",
    "def load_imports():\n",
    "    import pandas as pd\n",
    "    import numpy as np\n",
    "    import matplotlib.pyplot as plt\n",
    "    import seaborn as sns\n",
    "    from scipy import stats \n",
    "    import plotly.express as px\n",
    "    import plotly.graph_objects as go\n",
    "    import nbformat\n",
    "    from sklearn.impute import SimpleImputer\n",
    "    from io import StringIO\n",
    "    from sklearn.ensemble import RandomForestClassifier\n",
    "    from sklearn.feature_selection import SelectKBest, chi2\n",
    "    from imblearn.over_sampling import SMOTE\n",
    "    from sklearn.model_selection import train_test_split\n",
    "    from sklearn.ensemble import GradientBoostingClassifier\n",
    "    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "    from sklearn.naive_bayes import GaussianNB\n",
    "    from sklearn.preprocessing import StandardScaler\n",
    "    from sklearn.preprocessing import LabelEncoder\n",
    "    from sklearn.feature_selection import SelectKBest, f_classif\n",
    "    from sklearn.model_selection import GridSearchCV\n",
    "    from sklearn.neighbors import KNeighborsClassifier\n",
    "    \n",
    "    # return the imported modules to use them if necessary\n",
    "    return {\n",
    "        'pd': pd,\n",
    "        'np': np,\n",
    "        'plt': plt,\n",
    "        'sns': sns,\n",
    "        'stats': stats,\n",
    "        'px': px,\n",
    "        'go': go,\n",
    "        'nbformat': nbformat,\n",
    "        'SimpleImputer': SimpleImputer,\n",
    "        'StringIO': StringIO,\n",
    "        'RandomForestClassifier': RandomForestClassifier,\n",
    "        'SelectKBest': SelectKBest,\n",
    "        'chi2': chi2,\n",
    "        'SMOTE': SMOTE,\n",
    "        'train_test_split': train_test_split,\n",
    "        'GradientBoostingClassifier': GradientBoostingClassifier,\n",
    "        'accuracy_score': accuracy_score,\n",
    "        'classification_report': classification_report,\n",
    "        'confusion_matrix': confusion_matrix,\n",
    "        'GaussianNB': GaussianNB,\n",
    "        'StandardScaler': StandardScaler,\n",
    "        'LabelEncoder': LabelEncoder,\n",
    "        'f_classif': f_classif,\n",
    "        'GridSearchCV': GridSearchCV,\n",
    "        'KNeighborsClassifier': KNeighborsClassifier,\n",
    "    }\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
