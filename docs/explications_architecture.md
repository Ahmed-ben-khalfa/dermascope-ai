# Comprendre l'Architecture de DermaScope AI

## 🧠 1. C'est quoi un "Neurone" artificiel ?
Imagine un neurone artificiel comme un détective très basique. Son rôle : regarder une petite information (une couleur, une ligne, un pixel), faire un calcul mathématique dessus, et envoyer son résultat au détective suivant.

Quand on met des millions de neurones ensemble, on crée un Réseau de Neurones. Ils se transmettent l'information en cascade. Les premiers neurones détectent des choses simples (les bords, les couleurs). Les neurones du milieu détectent des formes (des cercles, des taches). Les tout derniers neurones assemblent tout ça pour crier : "C'est un mélanome !".

## 🏢 2. La répartition des rôles dans notre architecture
Dans ton code, on a construit notre IA comme un immense immeuble de bureaux. Voici comment on a réparti les rôles, ligne par ligne :

* `base_model = EfficientNetB0(...)` : **L'équipe des Experts de Google.** Ici, on importe 5,3 millions de neurones pré-entraînés par Google sur des images de la vie de tous les jours. C'est la base de la pyramide. Leurs neurones savent parfaitement extraire les textures et les formes.

* `x = layers.GlobalAveragePooling2D()(x)` : **Le Synthétiseur.** Les millions de neurones d'EfficientNet recrachent beaucoup trop d'informations. Cette couche compresse toutes ces informations visuelles en un "résumé" clair et compact.

* `x = layers.Dropout(0.5)(x)` : **Le Professeur strict (Anti-triche).** Pendant l'entraînement, cette couche "éteint" (désactive) aléatoirement 50% des neurones. Pourquoi ? Pour empêcher les neurones de faire les fainéants ! S'ils travaillent toujours avec les mêmes collègues, ils vont juste "apprendre par cœur" les images au lieu de comprendre la maladie. Le Dropout les force à être tous robustes.

* `x = layers.Dense(128, activation='relu')(x)` : **Les Apprentis Dermatologues.** On ajoute 128 nouveaux neurones (complètement vierges) qui vont lire le résumé de l'image. L'activation `relu` signifie que si le neurone trouve un indice pertinent (ex: bord irrégulier), il s'allume très fort. Si l'indice est mauvais, il reste à 0.

* `outputs = layers.Dense(7, activation='softmax')(x)` : **Le Jury Final.** C'est la toute dernière couche. Elle contient exactement 7 neurones, un pour chaque maladie (mel, nv, bcc...). La fonction `softmax` est magique : elle force les 7 neurones à se partager 100%. Par exemple, le neurone 1 dira 10%, le neurone 2 dira 80%, etc.

## ⏳ 3. La Phase 1 (Apprendre à parler)
Le code : `base_model.trainable = False` puis `model.fit(...)`

Dans cette phase, les millions de neurones d'EfficientNetB0 sont gelés. Leurs connexions mathématiques ne peuvent pas changer. Ils regardent l'image de peau et disent : "Je vois une tache marron avec des bords flous". Nous n'entraînons QUE les couches finales (les Apprentis et le Jury). On leur dit : "Quand les experts de Google vous disent 'tache marron à bord flous', sachez que ça correspond à la maladie numéro 4".

Ici, on apprend à nos neurones finaux à interpréter ce que voient les experts.

## 🔓 4. La Phase 2 (Le Fine-Tuning : Devenir Expert)
Le code : `base_model.trainable = True` puis `learning_rate=1e-5`

Maintenant que nos couches finales savent prendre une décision, on dégèle l'ensemble du réseau (les millions de neurones de Google). On veut qu'ils arrêtent d'utiliser leurs connaissances génériques (pour reconnaître des chiens ou des voitures) et qu'ils spécialisent leurs yeux exclusivement sur la peau humaine.

**Pourquoi un `learning_rate` de 1e-5 (très faible) ?** La "vitesse d'apprentissage" (learning rate), c'est la violence avec laquelle on modifie le cerveau à chaque erreur. Si on mettait une vitesse normale, on détruirait en 2 secondes toutes les années d'apprentissage que Google a fait subir au modèle ! En mettant une vitesse minuscule (1e-5), on permet aux neurones de Google de faire des tout petits ajustements microscopiques (Fine-Tuning) pour devenir parfaits sur la peau, sans oublier leur base.

## ⚖️ 5. L'entraînement (Comment ils apprennent concrètement ?)
Pendant les époques (Epochs) appelées par la fonction `.fit()`, voici ce qui se passe des milliers de fois par seconde :

1. L'IA regarde une photo (ex: un Mélanome).
2. Le Jury de 7 neurones donne son verdict (ex: il dit "C'est un grain de beauté normal à 90%").
3. La **Focal Loss** entre en jeu : "FAUX ! C'était un mélanome !".
4. Les **Class Weights** s'en mêlent : "En plus, c'est la classe Mélanome, donc ton erreur est punie 1.62 fois plus fort !".
5. La "punition" (l'Erreur Mathématique) redescend dans tout le réseau. Les mathématiques (la Rétropropagation) modifient très légèrement le "poids" de chaque neurone qui a participé à la mauvaise décision, pour qu'il soit un peu moins bête à la photo suivante.

À la fin de la Phase 2, tes neurones sont des experts mondiaux en dermatologie !
