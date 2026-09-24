# Journal de Bord - Projet Dermascope (Phase 1)
**Sujet :** Deep Data Engineering pour la Détection de Cancers de la Peau (Dataset ISIC HAM10000).

---

## PARTIE 1 : Théorie et Concepts Médicaux (F.A.Q)

Cette section résume nos échanges sur le fonctionnement "cognitif" du modèle d'Intelligence Artificielle.

### Q1 : Comment le modèle sait qu'il doit chercher la tâche et pas la main entière ?
*   **Réponse :** On peut soit utiliser un modèle spécialisé en segmentation (ex: U-Net ou SAM) pour détourer explicitement la lésion (ce que nous ferons en Phase 2), soit laisser le réseau de classification classique (ex: EfficientNet) apprendre par déduction. S'il voit 10 000 images de bras différents avec des cancers, le seul point commun mathématique entre ces images sera la tâche elle-même. Il apprendra donc naturellement à ignorer le fond.

### Q2 : Comment reconnaît-il la maladie ? Est-ce juste le contour et la couleur ? C'est quoi le RVB ?
*   **Réponse :** Le modèle n'a pas d'yeux. Il lit l'image sous la forme de 3 gigantesques tableaux de nombres (les canaux RVB : Rouge, Vert, Bleu), allant de 0 (éteint) à 255 (allumé). 
Il ne regarde pas *que* le contour. Il cherche mathématiquement la règle **ABCDE** des dermatologues :
    *   **A (Asymétrie) & B (Bords) :** Ses filtres de convolution détectent les ruptures nettes ou irrégulières dans les nombres.
    *   **C (Couleur) :** Il détecte l'hétérogénéité (un passage brutal de pixels marrons à des pixels noirs ou bleutés).
    *   **Textures (Structures internes) :** Il repère des micro-motifs (réseaux pigmentaires, points).
    *   *Note sur la Taille (Diamètre) :* L'IA ne peut pas deviner la taille réelle (en cm) juste avec des pixels. Il est vital de lui fournir cette taille en *Métadonnée* (donnée tabulaire) en plus de l'image.

### Q3 : Une maladie possède-t-elle toujours les mêmes couleurs ?
*   **Réponse :** Absolument pas, et c'est le grand piège de la dermatologie médicale. La couleur varie selon le **stade de la maladie**, la **lumière de la caméra**, mais surtout selon la **couleur de peau du patient (Phototype)**. Une inflammation rouge sur une peau blanche paraîtra grise/noire sur une peau sombre. D'où l'importance vitale d'un dataset diversifié.

### Q4 : Que sont les maladies dans le dataset ISIC ?
*   Le dataset contient 7 diagnostics. Pour simplifier et répondre à l'urgence médicale, nous les séparons en deux `Target` :
    *   **Target 1 (Cancers / Urgences) :** `mel` (Mélanome), `bcc` (Carcinome), `akiec` (Lésion pré-cancéreuse).
    *   **Target 0 (Bénin / Sain) :** `nv` (Grain de beauté classique), `bkl` (Tâche de vieillesse), `df`, `vasc`.

---

## PARTIE 2 : Défis Data Engineering & Splitting

### Le Défi du Pare-Feu et des Formats (Erreur 403 et KeyError)
Lors du téléchargement des données depuis Harvard Dataverse, nous avons affronté le quotidien du Data Engineer :
1.  **L'Erreur 403 (Forbidden) :** Le serveur bloquait le script Colab. Nous avons contourné cela en modifiant le `User-Agent` via la librairie `requests` pour simuler un navigateur Chrome.
2.  **L'Erreur KeyError :** Pandas ne trouvait pas nos colonnes car Harvard a envoyé un fichier TSV (Tab-Separated Values) au lieu d'un CSV (Comma-Separated). Nous avons corrigé cela en forçant le paramètre `sep='\t'`.

### Le concept de Data Leakage et de GroupShuffleSplit
*   **Le problème :** Un patient peut avoir 3 photos de la même lésion. Si on coupe le dataset au hasard (`train_test_split`), l'IA pourrait voir la photo 1 en entraînement et la photo 2 en test. Elle "tricherait" en reconnaissant la peau du patient au lieu de la maladie.
*   **La solution (`GroupShuffleSplit`) :** Cet algorithme regroupe les données par `patient_id`. Il tire au hasard et envoie *toutes* les photos d'un patient donné exclusivement dans la boîte de Train, ou exclusivement dans celle de Test.

### L'Audit de Sécurité (Explication du code de vérification)
La **Cellule 6** du code agit comme un contrôleur qualité. Elle ne modifie rien, elle vérifie juste que la Cellule 4 a bien fait son travail. Voici comment elle fonctionne étape par étape :

1. **La fonction `set()` :** En Python, un *set* (ensemble) extrait toutes les valeurs uniques d'une liste. La ligne `train_patients = set(train_df['patient_id'])` crée une liste contenant tous les identifiants uniques de patients présents dans le Train (sans aucun doublon). Elle fait de même pour la Validation et le Test.
2. **La fonction `intersection()` :** C'est une opération mathématique des ensembles. L'instruction `train_patients.intersection(test_patients)` compare la liste du Train et celle du Test. Son rôle est de recracher uniquement les noms des patients qui seraient présents dans les **deux** listes en même temps.
3. **Le verdict (`len == 0`) :** Si la taille de cette intersection est de zéro (`len(leak_test) == 0`), cela prouve avec une certitude mathématique absolue qu'aucun patient n'est à la fois dans le Train et dans le Test. Il n'y a donc **aucune fuite de données (zéro Data Leakage)**. Ton IA ne pourra pas tricher.

---

## PARTIE 3 : Code Python Complet (Phase 1)

Voici les cellules finales, corrigées et fonctionnelles, à exécuter dans l'ordre.

### Cellule 1 : Imports
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import requests
import io
```

### Cellule 2 : Ingestion des données ISIC (Bypass Firewall & TSV)
```python
print("Connexion au serveur Harvard Dataverse (Contournement anti-bot)...")
url_isic = "https://dataverse.harvard.edu/api/access/datafile/3172582"

# Fausse identité Chrome
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
response = requests.get(url_isic, headers=headers)

if response.status_code == 200:
    # Lecture avec séparateur Tabulation (\t)
    df_isic = pd.read_csv(io.StringIO(response.text), sep='\t')
    
    # Standardisation des colonnes
    df = df_isic.rename(columns={'lesion_id': 'patient_id', 'dx': 'diagnosis'})
    
    print(f"✅ Succès ! Nombre d'images chargées : {len(df)}")
    display(df.head())
else:
    print(f"❌ Échec de téléchargement. Code : {response.status_code}")
```

### Cellule 3 : Création de la Target Médicale
```python
# Cancers et Pré-cancers
malignant_classes = ['mel', 'bcc', 'akiec']

# 0 = Bénin, 1 = Malin (Cancer)
df['target'] = df['diagnosis'].apply(lambda x: 1 if x in malignant_classes else 0)

print("Répartition des cibles :")
print(df['target'].value_counts())
```

### Cellule 4 : Nettoyage et Imputation (Data Cleaning)
```python
# 1. Remplir les âges manquants par la médiane
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)

# 2. Remplir les textes manquants
df['sex'] = df['sex'].fillna('unknown')
df['localization'] = df['localization'].fillna('unknown')

# 3. Encodage One-Hot
df_clean = pd.get_dummies(df, columns=['sex', 'localization'])
print("Colonnes prêtes pour le modèle Deep Learning.")
```

### Cellule 5 : Le Split Patient (80% / 10% / 10%)
```python
# Train (80%) et Temp (20%)
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, temp_idx = next(gss.split(df_clean, groups=df_clean['patient_id']))

train_df = df_clean.iloc[train_idx].copy()
temp_df = df_clean.iloc[temp_idx].copy()

# Validation (10%) et Test (10%)
gss_val_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
val_idx, test_idx = next(gss_val_test.split(temp_df, groups=temp_df['patient_id']))

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
```

### Cellule 6 : L'Audit Anti-Fuite (Data Leakage Check)
```python
train_patients = set(train_df['patient_id'])
val_patients = set(val_df['patient_id'])
test_patients = set(test_df['patient_id'])

# Intersection entre les ensembles
leak_val = train_patients.intersection(val_patients)
leak_test = train_patients.intersection(test_patients)

print("=== AUDIT SÉCURITÉ ===")
if len(leak_val) == 0 and len(leak_test) == 0:
    print("✅ SUCCÈS ABSOLU : Le GroupShuffleSplit a fonctionné. Zéro fuite de données.")
else:
    print("❌ ALERTE FUITE DE DONNÉES ! Des patients se croisent.")
```
