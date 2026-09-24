# Rapport Phase 2 : Pipeline de Traitement d'Image (Projet Dermascope)

Ce document détaille l'ingénierie mathématique et logicielle appliquée aux images brutes avant de les fournir au réseau de neurones profond (CNN). L'objectif absolu de cette phase est de **détruire tous les biais potentiels** pour empêcher l'Intelligence Artificielle d'apprendre des "raccourcis" visuels au lieu d'apprendre à reconnaître le cancer.

---

## 1. L'Algorithme de Nettoyage (DullRazor Universel)

### Le Piège Fondamental
Si l'on entraîne une IA sur des lésions poilues, le réseau convolutif risque d'associer la présence de poils (une texture forte) à la malignité de la lésion. Il fallait donc "gommer" les poils sans altérer l'information clinique de la peau.

### L'Évolution vers le DullRazor Universel (Les pièges surmontés)
L'algorithme DullRazor classique a rapidement montré ses limites face à la réalité clinique. Nous l'avons repensé pour surmonter trois pièges critiques :

*   **Le Piège du Contraste (Les poils blancs) :** Le Black-Hat ne détecte que les éléments sombres sur un fond clair. Face à des poils blancs ou des reflets de flash, l'algorithme était aveugle. 
    *   *Solution :* Nous avons combiné un masque `Top-Hat` et un masque `Black-Hat` via un **OU Logique Matriciel**.
*   **Le Piège de l'Épaisseur (Les intersections) :** Quand des poils se croisent, ils forment un "nœud" épais que les petits filtres ignorent.
    *   *Solution :* Nous avons augmenté la taille du radar (de 9x9 à 17x17) et appliqué une **Dilatation Rectangulaire** (5x5) sur le masque binaire. L'intensité de l'inpainting a été montée à un rayon de 5.
*   **Le Piège du Macro-Shot (L'Invariance d'échelle) :** Si la photo est zoomée, un poil fin devient un tronc d'arbre de 40 pixels.
    *   *Solution :* Le **Resizing Standardisé** (512x512 pixels). L'échelle des éléments est mathématiquement figée, garantissant 100% de fiabilité à nos filtres.

---

## 2. La Segmentation de la Lésion (Segment Anything Model - SAM)

### Les Limites de SAM et le Piège de la Vraie Vie
SAM est un modèle exceptionnel mais il n'a aucune connaissance médicale. Dans un dataset académique, pointer au centre `(256, 256)` fonctionne. Dans la vraie vie, la lésion peut être excentrée. De plus, utiliser une méthode classique (Otsu) pour chercher "le plus sombre" introduit un biais racial gravissime sur les peaux noires.

### L'Ingénierie du Viseur Hybride (L'Auto-Aim)
Pour guider SAM de manière autonome et éthique, nous avons conçu un viseur hybride :

1.  **L'Anomalie Colorimétrique (Indépendante de la pigmentation) :**
    *   Conversion dans l'espace `LAB`. L'algorithme échantillonne les bords (la peau saine du patient). Il calcule la distance Euclidienne de chaque pixel par rapport à cette peau de référence. L'anomalie est ce qui diverge le plus (plus clair, plus foncé, ou plus rouge).
2.  **La Gravité Centrale (Le Biais Spatial pour le Zoom) :**
    *   Si l'image est extrêmement zoomée, la lésion touche les bords. Nous avons généré une carte de Gaussienne 2D créant un fort "Biais Central".
    *   **Fusion :** En multipliant la carte d'Anomalie par la Gravité Centrale (`Anomalie * Gravité`), l'algorithme trouve la coordonnée parfaite `(X, Y)` envoyée comme **Point Prompt** à SAM.

---

## 3. L'Application du Bokeh et L'Atténuation des Contours

### Le Piège Mortel : Le Biais de Contour (Edge Bias)
Coller une lésion nette sur un fond flou crée une "ligne de fracture" d'une netteté mathématique. Le CNN aurait ignoré le cancer pour se contenter de repérer le "coup de ciseaux" de SAM.

### La Solution : Le Feathering (Masque Alpha)
Nous avons transformé le masque binaire rigide (0 ou 1) en un gradient décimal en floutant le **masque lui-même**. 
Par l'opération d'**Alpha Blending** `(Image * Masque_Flou) + (Fond_Flou * Inverse_Masque)`, la transition entre la maladie nette et le fond flou est devenue un dégradé doux et biologique. L'IA est obligée de regarder *à l'intérieur* de la tâche.

---

## 4. Dictionnaire Technique : Concepts des Masques et Modèles

Afin de comprendre la profondeur mathématique de ce pipeline, voici la définition conceptuelle de chaque outil utilisé :

### A. Les Concepts de Masquage (Computer Vision)
*   **Masque Black-Hat (Chapeau Noir) :** Concept de Morphologie Mathématique. L'algorithme effectue une "Fermeture" (Dilatation suivie d'une Érosion) pour effacer les petits détails sombres, puis soustrait l'image originale de ce résultat. **Concept :** Isoler les structures fines et sombres sur un fond clair (Ex: Poils noirs).
*   **Masque Top-Hat (Chapeau Blanc) :** Opération inverse (Image originale moins son "Ouverture"). **Concept :** Isoler les structures fines et très claires sur un fond sombre (Ex: Poils blancs, reflets de flash).
*   **Masque Combiné (OU Logique Matriciel / np.where) :** Plutôt que d'évaluer chaque pixel avec une boucle `if` (très lent), on applique une opération booléenne sur la matrice entière. **Concept :** Si un pixel est "Vrai" dans le Black-Hat OU "Vrai" dans le Top-Hat, il devient "Vrai" dans le masque final. C'est un radar universel.
*   **Masque de Dilatation :** Opération qui "gonfle" les pixels blancs d'un masque binaire. **Concept :** Compenser les "trous" dans la détection. Si trois poils se croisent, ils forment un nœud trop épais pour être détecté. La dilatation force le masque à s'étendre et à avaler ces intersections pour garantir une gomme parfaite.
*   **Masque Binaire (Segmentation) :** Une matrice mathématique ne contenant que des 0 (Noir/Exclu) et des 1 (Blanc/Inclus). **Concept :** Définir une frontière géométrique infranchissable entre un objet (la maladie) et le reste de l'univers (le fond).
*   **Masque Alpha / Dégradé (Feathering) :** Un masque contenant des valeurs décimales continues de 0.0 (Transparence totale) à 1.0 (Opacité totale). **Concept :** Permettre l'Alpha Blending (fusion). Au lieu d'une coupure au couteau, les pixels à 0.5 mélangent 50% de la maladie avec 50% du fond flou, créant une transition douce indétectable par l'IA.

### B. Les Algorithmes et Modèles Deep Learning
*   **Algorithme d'Inpainting (Méthode Fast Marching - Telea) :** Algorithme de reconstruction d'image. **Concept :** Il parcourt les "trous" (le masque des poils) depuis l'extérieur vers l'intérieur, pixel par pixel, en résolvant l'équation d'Eikonal. Il remplace le pixel manquant par une moyenne pondérée des couleurs des pixels sains immédiatement adjacents. C'est une cicatrisation numérique.
*   **Algorithme d'Anomalie (Distance Euclidienne en espace LAB) :** L'espace couleur RGB est mauvais pour évaluer les différences perçues par l'œil humain. L'espace LAB sépare la Luminosité (L) de la couleur (a, b). **Concept :** En mesurant la distance mathématique 3D entre la couleur de la peau saine (échantillonnée sur les bords) et le reste de l'image, on détecte une "Anomalie" physiologique, totalement indépendante de la race ou du niveau de mélanine du patient.
*   **SAM (Segment Anything Model - Modèle Fondationnel) :** Modèle Deep Learning créé par Meta basé sur l'architecture *Vision Transformer (ViT)*. **Concept :** C'est un modèle "Zero-Shot" (Prêt à l'emploi sans entraînement spécifique) qui a appris le concept universel de "ce qu'est un objet" en s'entraînant sur plus d'un milliard de masques. Il ne classe pas l'objet (il ne sait pas ce qu'est un cancer), il comprend juste la géométrie spatiale. Il exige un "Prompt" (un point ou une boîte) pour savoir quelle zone de son immense réseau de neurones il doit activer pour générer le masque binaire.
