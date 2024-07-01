# imports.py

def load_imports():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats 
    import plotly.express as px
    import plotly.graph_objects as go
    import nbformat
    from sklearn.impute import SimpleImputer
    from io import StringIO
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, chi2
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    import logging
    import os

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    
    # return the imported modules to use them if necessary
    return {
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'stats': stats,
        'px': px,
        'go': go,
        'nbformat': nbformat,
        'SimpleImputer': SimpleImputer,
        'StringIO': StringIO,
        'RandomForestClassifier': RandomForestClassifier,
        'SelectKBest': SelectKBest,
        'chi2': chi2,
        'SMOTE': SMOTE,
        'train_test_split': train_test_split,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'accuracy_score': accuracy_score,
        'classification_report': classification_report,
        'confusion_matrix': confusion_matrix,
        'GaussianNB': GaussianNB,
        'StandardScaler': StandardScaler,
        'LabelEncoder': LabelEncoder,
        'f_classif': f_classif,
        'GridSearchCV': GridSearchCV,
        'KNeighborsClassifier': KNeighborsClassifier,
        'logging': logging,
        'os': os,
    }

