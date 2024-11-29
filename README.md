# GEI723-APP3

# Réseau de Neurones à Impulsions - Modèle LIF en PyTorch

## Objectif
Ce projet implémente un réseau de neurones à impulsions (SNN) en utilisant un modèle **Leaky Integrate-and-Fire (LIF)**, en PyTorch. 
L'objectif est d'implémenter la méthode de rétropropagation de l'erreur dans un réseau SNN afin de réaliser de la classification sur MINST (banque de données d'image de chiffres manuscrits). 
Ce modèle est conçu pour étudier les performances des réseaux de neurones à impulsions dans des tâches de classification en utilisant des trains d'impulsions comme entrée.

## Structure du Code

Le code est structuré en quatre sections principales :

1. **Préparation de la configuration pour le réseau de neurones**  
   Cette section définit la configuration du réseau, y compris l'initialisation des poids, des paramètres temporels (tau de la membrane et des courants), et la préparation du train d'impulsions d'entrée.

2. **Entrainement et Validation**  
   Cette étape consiste à entraîner le modèle en ajustant les poids du réseau sur un ensemble de données d'apprentissage. Les performances sont validées sur un jeu de données de validation pour évaluer la capacité du réseau à généraliser.

3. **Test**  
   Le modèle entraîné est testé sur des données inconnues (jeu de test) pour mesurer sa précision et sa robustesse dans des conditions réelles d'utilisation.

4. **Graphiques**  
   Des graphiques sont générés pour visualiser l'évolution du potentiel membranaire des neurones, les courants d'entrée, ainsi que les impulsions générées au fil du temps. Cela permet de suivre l'activité neuronale et d'analyser les dynamiques du modèle.

## Études Menées

Ce modèle a été conçu pour explorer les dynamiques des neurones à impulsions avec une approche à base de courant. Les principales études comprennent :

- L'impact des poids sur la propagation des signaux et le déclenchement des impulsions.
- L'efficacité du modèle LIF dans le traitement des entrées en forme de trains d'impulsions.
- L'analyse du comportement du potentiel membranaire dans le temps et la gestion des seuils de déclenchement.
  
Le code peut être utilisé comme base pour explorer d'autres dynamiques neuronales ou pour tester des variantes du modèle LIF dans des réseaux à impulsions plus complexes.


## Auteurs :
Clémence Lamballe
Behrouz Nik-Nejad-Kazem-Pour
Jean-Sébastien Giroux
