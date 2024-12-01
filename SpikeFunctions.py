import torch
from torch.autograd import Function

'''dans les fonctions backward on doit multiplier grad_input (le gradient des couches apres) 
par le gradient de la couche actuelle => derivation en chaine'''
class SpikeFunction_Classical_RELU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) # Sauvegarde l'entrée pour l'utiliser dans la passe arrière.
        #return (input > 0).float() # Retourne 1 si l'entrée est positive, sinon 0.
        return max(0, input).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors # Récupère l'entrée sauvegardée depuis la passe avant.
        grad_input = grad_output.clone() 
        grad_input[input <= 0] = 0 
        return grad_input


class SpikeFunction_Leaky_RELU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) # Sauvegarde l'entrée pour la passe arrière.
        return torch.where(input > 0, input, 0.01 * input) # Applique Leaky ReLU. (if condition, action, else action)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] *= 0.01 # Multiplie par 0.01 pour les entrées négatives. 
        return grad_input


class SpikeFunction_Abs_RELU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.abs(input)# Retourne la valeur absolue de l'entrée.

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()# Copie le gradient de sortie.
        grad_input[input < 0] *= -1# Multiplie par -1 pour les entrées négatives.
        return grad_input


class SpikeFunction_Sigmoid(Function):
    
    @staticmethod
    def forward(ctx, input, alpha=1.0):
        sigmoid = torch.sigmoid(alpha*input) # Calcule la sigmoïde de l'entrée.
        ctx.save_for_backward(sigmoid, torch.tensor(alpha)) # Sauvegarde la sortie pour la passe arrière.
        return sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid, alpha = ctx.saved_tensors # Récupère la sortie sauvegardée (sigmoïde).
        grad_input = grad_output * sigmoid * (1 - sigmoid)* alpha # Applique la dérivée de la sigmoïde.
        return grad_input


class SpikeFunction_Triangular(Function):
    @staticmethod
    def forward(ctx, input, theta=0, delta=1):
        ctx.save_for_backward(input) # Sauvegarde l'entrée pour la passe arrière.
        ctx.theta = theta
        ctx.delta = delta
        
        # Calcul de la fonction triangulaire en utilisant les paramètres theta et delta
        # 1. input - theta : Décalage de l'entrée par rapport au paramètre theta
        # 2. abs(input - theta) : Calcul de la valeur absolue de l'entrée par rapport au paramètre theta
        # 3. 1 - abs(input - theta) / delta : Applique une forme triangulaire centrée autour de theta avec une largeur de delta
        # 4. torch.maximum : Si la valeur est négative, on remplace par 0
        grad_input = torch.maximum(1 - torch.abs(input - theta) / delta, torch.tensor(0.0, device=input.device))
        return grad_input


    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors  # Récupère l'entrée sauvegardée
        grad_input = grad_output.clone()  # Copie le gradient de sortie
        grad_input[torch.abs(input - ctx.theta) >= ctx.delta] = 0  # Pas de gradient si |input - theta| >= delta
        grad_input[torch.abs(input - ctx.theta) < ctx.delta] = -1 / ctx.delta  # Gradient linéaire
        return grad_input , None, None


class SpikeFunction_Gaussian(Function):
    @staticmethod
    def forward(ctx, input, alpha=1.0, theta=0.0):
        
        # Calcul de la fonction gaussienne avec les paramètres alpha et theta
        #gaussian = torch.exp(-((input - theta) ** 2) / (2 * alpha ** 2))  # Gaussienne normalisee
        #gaussian = torch.exp(-input ** 2)# Calcule la fonction gaussienne. de base
 
        gaussian = torch.exp(-alpha * (input - theta) ** 2)

        ctx.save_for_backward(input, gaussian)  # Sauvegarde de l'entrée et de la sortie pour la passe arrière
        ctx.alpha = alpha
        ctx.theta = theta
        return gaussian

    @staticmethod
    def backward(ctx, grad_output):
        input, gaussian = ctx.saved_tensors# Récupère l'entrée et la sortie sauvegardées.
        alpha = ctx.alpha
        theta = ctx.theta
        #grad_input = grad_output * (-2 * input * gaussian)# Applique la dérivée de la gaussienne. parametres de base
        #grad_input = grad_output * ((input - theta) / (alpha ** 2)) * gaussian # Gaussienne normalisee
        grad_input = grad_output * (-2 * alpha * (input - theta)) * gaussian

        return grad_input,None,None