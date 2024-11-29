# **GEI723-APP3**  
## **Réseau de Neurones à Impulsions - Modèle LIF en PyTorch**  

### **Objectif**  
Ce projet implémente un réseau de neurones à impulsions (SNN) basé sur le modèle **Leaky Integrate-and-Fire (LIF)** en PyTorch.  
L'objectif est d'appliquer la rétropropagation des erreurs dans un réseau SNN afin d'effectuer une classification sur MNIST (banque d'images de chiffres manuscrits).  
Ce modèle permet d'analyser les performances des réseaux de neurones à impulsions dans des tâches de classification, en traitant des trains d'impulsions comme entrées.  

---

### **Structure du Code**  

Le code est organisé en **quatre sections principales**.  

#### **1. Préparation de la Configuration pour le Réseau de Neurones**  
Cette section définit et initialise les éléments nécessaires pour construire et configurer le réseau.  
- **Sous-sections :**  
  - Configuration 
  - Préparation des données 
  - Conversion en décharges 
  - Division Entraînement/Test/Validation 
  - Création du réseau 
  - Différentes fonctions d'activation 
  - Choix de l'utilisateur 
  - Implémentation dynamique LIF 

#### **2. Entraînement et Validation**  
Dans cette section, le modèle est entraîné sur l'ensemble d'apprentissage :  
- **Entraînement :** Ajustement des poids en minimisant la perte.  
- **Validation :** Évaluation des performances sur un ensemble de validation pour contrôler la généralisation.  

#### **3. Test**  
Le modèle final est testé sur des données inconnues (jeu de test). 

#### **4. Visualisation et Graphiques**  
Cette section permet d'analyser les résultats et les dynamiques du réseau grâce à des graphiques.  
- **Sous-sections :**  
  - Évolution des poids  
  - Évolution de la perte 

---

### **Études Menées** 

Ce Notebook propose l'implémentation et l'analyse de plusieurs fonctions d'activation pour les réseaux de neurones à impulsions :  
- **ReLU Classique**  
- **Leaky ReLU**  
- **ReLU basée sur une fonction absolue (Abs ReLU)**  
- **Approximation sigmoïdale**  
- **Approximation triangulaire**  
- **Approximation gaussienne**  

Ce Notebook permet de tester les performances de classification en fonction des différents paramètres des fonctions listé ci-dessus.

Ce Notebook permet aussi de tester des configurations d'apprentissage extrêmes


## Auteurs :
Clémence Lamballe
Behrouz Nik-Nejad-Kazem-Pour
Jean-Sébastien Giroux
