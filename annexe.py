import numpy as np
import matplotlib.pyplot as plt

# Définition des fonctions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def custom_absolute_relu(x):
    return np.maximum(0, x) - np.minimum(0, x)

def sigmoid_approximation(x, alpha=1.0):
    return 1 / (1 + np.exp(-alpha * x))

def linear_approximation(x, epsilon=1.0):
    return np.where(np.abs(x) <= epsilon, 1, 0)

def tanh_approximation(x, alpha=1.0):
    return np.tanh(alpha * x)


def triangular_approximation(x, theta=0.0, delta=1.0):
    """Approximation triangulaire autour de theta avec largeur delta."""
    return np.maximum(0, 1 - np.abs(x - theta) / delta)

def spiking_activation(U_j, theta=1.0):
    """Simuler un neurone qui génère un spike (1) si son potentiel dépasse un seuil."""
    return np.where(U_j >= theta, 1, 0)

def gaussian_approximation(x, alpha=1.0, theta=0.0):
    """Fonction gaussienne lissée."""
    return np.exp(-alpha * (x - theta) ** 2)

# Dérivées des fonctions
def sigmoid_derivative(x, alpha=1.0):
    sig = sigmoid_approximation(x, alpha)
    return alpha * sig * (1 - sig)

def triangular_derivative(x, theta=0.0, delta=1.0):
    return np.where(np.abs(x - theta) <= delta, 
                    np.sign(x - theta) / delta, 
                    0)

def linear_derivative(x, epsilon=1.0):
    return np.where(np.abs(x) <= epsilon, 1, 0)  # Gradient constant ou nul

def gaussian_derivative(x, alpha=1.0, theta=0.0):
    """Dérivée de l'approximation gaussienne."""
    return -2 * alpha * (x - theta) * np.exp(-alpha * (x - theta) ** 2)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def custom_absolute_relu_derivative(x):
    return np.where(x > 0, 1, -1)


# Génération des données
x = np.linspace(-5, 5, 500)

# Calcul des valeurs des fonctions
y_relu = relu(x)
y_leaky_relu = leaky_relu(x, alpha=0.01)
y_custom_absolute_relu = custom_absolute_relu(x)
y_sigmoid = sigmoid_approximation(x, alpha=2.0)
y_linear = linear_approximation(x, epsilon=1.0)
y_tanh = tanh_approximation(x, alpha=2.0)
y_triangular = triangular_approximation(x, theta=0.0, delta=1.0)
y_gaussian = gaussian_approximation(x, alpha=1.0, theta=0.0)

# Simuler le potentiel membranaire U_j(t) avec une fonction sinusoïdale (juste pour l'exemple)
U_j = np.sin(x) * 2  # Potentiel qui monte et descend

# Appliquer l'activation de spike
y_spiking = spiking_activation(U_j, theta=0.5)  # Choix d'un seuil à 0.5



# Calcul des dérivées
y_sigmoid_derivative = sigmoid_derivative(x, alpha=2.0)
y_linear_derivative = linear_derivative(x, epsilon=1.0)
y_triangular_derivative = triangular_derivative(x, theta=0.0, delta=1.0)
y_gaussian_derivative = gaussian_derivative(x, alpha=1.0, theta=0.0)
y_relu_derivative = relu_derivative(x)
y_leaky_relu_derivative = leaky_relu_derivative(x, alpha=0.01)
y_custom_absolute_relu_derivative = custom_absolute_relu_derivative(x)


# Création des graphiques
plt.figure(figsize=(5, 3))

plt.plot(x, y_sigmoid, label="Sigmoïde", color="blue", linewidth=2.5)
#plt.plot(x, y_linear, label="Linéaire", color="green", linestyle="--", linewidth=2.5)
#plt.plot(x, y_tanh, label="Tangente Hyperbolique", color="purple")
plt.plot(x, y_gaussian, label="Gaussienne", color="green", linestyle="--", linewidth=2.5)
plt.plot(x, y_triangular, label="Triangulaire", color="red", linestyle=":", linewidth=2.5)

plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("Comparaison des approximations de gradient")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.grid()
plt.legend()
plt.tight_layout()




# Dérivées des fonctions d'activation

plt.figure(figsize=(5, 3))
plt.plot(x, y_sigmoid_derivative, label="Dérivée Sigmoïde", color="blue", linewidth=2.5)
#plt.plot(x, y_linear_derivative, label="Dérivée Linéaire", color="green", linestyle="--", linewidth=2.5)
plt.plot(x, y_gaussian_derivative, label="Dérivée Gaussienne", color="green", linestyle="--", linewidth=2.5)
plt.plot(x, y_triangular_derivative, label="Dérivée Triangulaire", color="red", linestyle=":", linewidth=2.5)

plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("Dérivées des fonctions d'activation")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.grid()
plt.legend()
plt.tight_layout()

# Création des graphiques RELU

plt.figure(figsize=(4, 3))

plt.plot(x, y_relu, label="ReLU simplifiée", color="blue", linestyle="-", linewidth=2.5)
plt.plot(x, y_leaky_relu, label="ReLU avec fuite", color="green", linestyle="--", linewidth=2.5)
plt.plot(x, y_custom_absolute_relu, label="ReLU absolue ", color="red", linestyle=":", linewidth=2.5)

plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("Comparaison des variantes de ReLU")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()
plt.tight_layout()

# Création des graphiques des dérivées des fonctions ReLU
plt.figure(figsize=(4, 3))

plt.plot(x, y_relu_derivative, label="Dérivée de ReLU", color="blue", linewidth=2.5)
plt.plot(x, y_leaky_relu_derivative, label="Dérivée de Leaky ReLU", color="green", linestyle="--", linewidth=2.5)
plt.plot(x, y_custom_absolute_relu_derivative, label="Dérivée de ReLU Absolue", color="red", linestyle=":", linewidth=2.5)

plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("Dérivées des fonctions d'activation ReLU")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.grid()
plt.legend()
plt.tight_layout()




# Affichage de l'activation de spike (fonction d'activation non différentiable)
plt.figure(figsize=(5, 3))

plt.plot(x, y_spiking, label="Spiking Activation $S_j(t)$", color="purple", linestyle=":", linewidth=2.5)

plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("Activation de Spike (fonction non différentiable)")
plt.xlabel("x")
plt.ylabel("$S_j(t)$")
plt.grid()
plt.legend()


plt.tight_layout()
plt.show()


