# 🚀 ROADMAP V2 : Architecture Multimodale (Ingénierie Avancée)

Pour ce projet de niveau "Senior", nous allons construire un réseau de neurones en "Y". L'objectif est de fusionner la vision par ordinateur (Computer Vision) et l'analyse de données structurées (Data Science classique).

---

## 🧠 LES 3 MODÈLES (Qui fait quoi ?)

### 1. L'Extracteur Visuel (Les Yeux)
* **Le Modèle :** `EfficientNetB0` (ou `EfficientNetB3` si la RAM le permet).
* **Son Rôle :** Scanner les pixels de l'image (224x224x3) pour détecter les textures asymétriques, les couleurs suspectes et les bordures.
* **Sa Sortie :** Il compresse l'image en un vecteur de caractéristiques pures (ex: 1280 nombres mathématiques qui résument la lésion).

### 2. L'Extracteur Clinique (Le Contexte)
* **Le Modèle :** Un réseau **MLP** (Multi-Layer Perceptron), composé de couches `Dense` basiques avec `BatchNormalization`.
* **Son Rôle :** Traiter les métadonnées du fichier CSV (`age`, `sex`, `localization`). Il apprend les statistiques médicales (ex: "Un enfant de 10 ans a très peu de chances d'avoir un mélanome", ou "Les carcinomes apparaissent souvent sur le visage").
* **Sa Sortie :** Un petit vecteur de caractéristiques de contexte (ex: 32 ou 64 nombres mathématiques).

### 3. La Tête de Fusion (Le Médecin Chef)
* **Le Modèle :** Un bloc de couches `Dense` avec du `Dropout` (pour éviter le sur-apprentissage).
* **Son Rôle :** Il prend le rapport de "l'Extracteur Visuel" ET le rapport de "l'Extracteur Clinique", il fusionne les deux (grâce à une couche `Concatenate`), pèse le pour et le contre, et prend la décision finale.
* **Sa Sortie :** 7 probabilités (Softmax) pour les 7 maladies.

---

## 🗺️ LES 4 ÉTAPES DE DÉVELOPPEMENT

### ÉTAPE 1 : Data Engineering Avancé (Nouveau !)
C'est ici qu'est la vraie ingénierie. Tu ne peux plus juste charger des images, il faut préparer le tableau de données.
1. **Nettoyage des valeurs manquantes (NaN) :** Dans le HAM10000, l'âge est parfois manquant. Il faudra imputer la moyenne ou la médiane.
2. **Standardisation Numérique :** L'âge (0 à 85 ans) doit être mis à l'échelle (entre 0 et 1) avec un `MinMaxScaler` pour ne pas écraser le réseau de neurones.
3. **One-Hot Encoding :** Transformer le sexe (Male/Female) et la localisation (back, face, trunk...) en colonnes de 0 et de 1 (via `pd.get_dummies`).
4. **Alignement parfait :** Lors du Split (Train/Test), il faut s'assurer que l'image N°1 correspond **absolument** à la ligne N°1 du tableau des métadonnées.

### ÉTAPE 2 : Construction de l'Architecture (Keras Functional API)
On abandonne le `Sequential()`. Tu vas utiliser l'API Fonctionnelle de Keras pour créer les deux branches en parallèle.
* Création de `input_image`.
* Création de `input_meta`.
* Utilisation de `tf.keras.layers.Concatenate()`.
* Déclaration du `tf.keras.Model(inputs=[input_image, input_meta], outputs=predictions)`.

### ÉTAPE 3 : La Data Augmentation "Synchronisée"
Gros défi technique : Quand tu fais une Data Augmentation sur une image rare (pour équilibrer les classes), tu dois **aussi** dupliquer la ligne de tableau correspondante ! L'image tournée à 90° appartient toujours au même patient de 55 ans.

### ÉTAPE 4 : L'Entraînement Multimodal
Lors du `.fit()`, au lieu de donner un tableau d'images, on donne une **liste de deux tableaux** à l'IA :
`model.fit(x=[X_train_images, X_train_meta], y=y_train...)`

---
## ⭐ LE BONUS POUR LE CV : L'Explicabilité (Grad-CAM & SHAP)
Une fois le modèle multimodal terminé, un ingénieur doit prouver **pourquoi** l'IA a pris sa décision.
* Utiliser **Grad-CAM** pour générer une "carte de chaleur" (Heatmap) sur l'image pour montrer où l'IA a regardé.
* Montrer quel poids l'IA a donné à l'âge par rapport à la photo.
