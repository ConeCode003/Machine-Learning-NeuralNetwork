# MLP Classifiers (Binary + Multiclass) 

Ovaj repozitorijum sadrži dva Jupyter notebook-a u kojima treniram MLP (Multi-Layer Perceptron) klasifikatore nad datasetima preuzetim direktno sa https://scikit-learn.org/stable/api/sklearn.datasets.html
- binarna klasifikacija (Breast Cancer dataset)
- višeklasna klasifikacija (Iris dataset)

--------------------------------------------------------------------

Šta sam uradio (ukratko)

1) Binary klasifikacija — Breast Cancer (load_breast_cancer)
- Učitao dataset iz sklearn.datasets (load_breast_cancer) i definisao X i y.
- Podelio podatke na train/test (80/20) uz random_state=42.
- Uradio standardizaciju podataka pomoću StandardScaler (fit na train, transform na test).
- Kreirao MLPClassifier (MLP neuronska mreža) sa podešenim hiperparametrima (npr. hidden layer 128, solver='sgd', momentum, max_iter).
- Treniranje modela na train skupu.
- Evaluacija:
  - Accuracy: 97.37%
  - classification_report (precision/recall/f1-score)
  - confusion matrix (vizualizacija)

2) Multiclass klasifikacija — Iris (load_iris)
- Učitao dataset iz sklearn.datasets (load_iris) i definisao X i y.
- Podelio podatke na train/test (80/20).
- Kreirao i istrenirao MLPClassifier (hidden layer 32, max_iter=50000).
- Evaluacija:
  - Accuracy: 93.33%
  - classification_report (precision/recall/f1-score)

--------------------------------------------------------------------

Struktura repozitorijuma
- mlp_binary_classifier.ipynb — kompletan workflow za binarnu klasifikaciju (preprocessing → trening → evaluacija + confusion matrix)
- mlp_multiclass_classifier.ipynb — kompletan workflow za višeklasnu klasifikaciju (trening → evaluacija)

--------------------------------------------------------------------

Korišćene tehnologije
- Python
- Jupyter Notebook
- scikit-learn
  - MLPClassifier
  - load_breast_cancer, load_iris
  - train_test_split
  - StandardScaler
  - accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
- matplotlib (vizualizacija confusion matrix)

--------------------------------------------------------------------

Model i evaluacija

Model
- Feed-forward neuronska mreža (MLP) za klasifikaciju

Korišćene metrike
- Accuracy
- Classification report (precision/recall/F1)
- Confusion matrix (za binarnu klasifikaciju)

Rezultati
- Binary (Breast Cancer): Accuracy = 97.37%
- Multiclass (Iris): Accuracy = 93.33%

--------------------------------------------------------------------

Pokretanje projekta (lokalno)

1) Instaliranje zavisnosti:
#bash
pip install scikit-learn matplotlib jupyter
2) Pokreni:
jupyter notebook

3)Otvori notebook:
mlp_binary_classifier.ipynb ili mlp_multiclass_classifier.ipynb

Sta bih sledece uradio:
Dodao dodatne metrike (npr. ROC-AUC za binary, confusion matrix i za multiclass).
Dodao cv / cross-validation - procena performansi
Uveo GridSearchCV za tuning hiperparametera - broj skrivenih slojeva neurona,learning-rate,regularizacija



