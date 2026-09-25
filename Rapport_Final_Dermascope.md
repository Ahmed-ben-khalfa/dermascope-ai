# Rapport Complet : Projet Dermascope AI 🔬

## 1. Introduction et Problématique
Le diagnostic précoce du mélanome est crucial pour la survie du patient. L'objectif de ce projet était de construire un système de vision par ordinateur robuste, capable non seulement d'analyser des images dermatoscopiques, mais également de prendre en compte les antécédents cliniques du patient (âge, sexe, localisation de la lésion).

## 2. Phase 1 : Nettoyage et Préparation des Données (Data Cleaning)
Le dataset initial présentait des défis majeurs :
- **Déséquilibre de classes sévère** (beaucoup de lésions bénignes, peu de malignes).
- **Valeurs manquantes** dans les métadonnées (âge, sexe).
- **Bruit visuel** (poils, bordures du dermatoscope).

**Solutions apportées :**
- Nettoyage des métadonnées (imputation de l'âge médian, encodage One-Hot pour le sexe et la localisation).
- Pipeline d'augmentation d'images agressif (Rotations, Flips, Color Jitter) pour compenser le manque d'images malignes.

## 3. Phase 2 : Entraînement Initial (V1) et Identification des Limites
Nous avons initialement entraîné un ensemble de modèles (EfficientNet, ResNet, DenseNet) par simple concaténation des caractéristiques visuelles et tabulaires.
- **Résultats V1 :** Sensibilité de 85%, AUC de 0.79.

## 4. Phase 3 : Optimisation Architecturale (L'Approche FiLM)
Pour pallier ces défauts, nous avons implémenté **FiLM (Feature-wise Linear Modulation)**.
Plutôt que de concaténer à la fin, les données cliniques (âge, sexe) agissent comme des "filtres" modulant directement les tenseurs visuels de l'image. Le réseau *apprend à regarder différemment* l'image si on lui dit que le patient a 80 ans au lieu de 20.

### Stratégie d'Entraînement : Progressive Resizing & Focal Loss
1. **Focal Loss :** Pour forcer le modèle à se concentrer sur les erreurs difficiles (les mélanomes atypiques).
2. **Phase 1 (Échauffement 256x256) :** Transfer Learning avec le backbone (vision) gelé.
3. **Phase 2 (Fine-Tuning 512x512) :** Dégel complet de l'architecture pour apprendre les micro-détails cellulaires.

## 5. Phase 4 : Résultats Finaux et Fiabilité
Les résultats de la stratégie FiLM + Progressive Resizing ont été spectaculaires :
- **ROC-AUC :** 0.9142 (Amélioration massive)
- **Sensibilité :** 95.42% (Détection de presque tous les cancers)
- **Spécificité :** 70.32% (Très bon compromis médical pour réduire les fausses alertes)

L'implémentation d'un seuil de chaleur (<40%) a permis d'obtenir une précision "Sniper". Le modèle ignore désormais totalement la peau saine et les artefacts de bordure pour se concentrer au millimètre près sur les structures malignes, validant ainsi la fiabilité clinique de l'IA.
