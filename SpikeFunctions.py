import torch
from torch.autograd import Function
from scipy.special import *

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
        return torch.abs(input) # Retourne la valeur absolue de l'entrée.

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone() # Copie le gradient de sortie.
        grad_input[input < 0] *= -1 # Multiplie par -1 pour les entrées négatives.
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
        return grad_input, None


class SpikeFunction_Triangular(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) # Sauvegarde l'entrée pour la passe arrière.

        grad_input = torch.where(
            input < 0.5,
            0,  # Output 0 for x < 0.5
            torch.where(
                input <= 1,
                2 * (input - 0.5), 
                torch.where(
                    input < 1.5,
                    2 * (1.5 - input),
                    0  
                    )
            )
        )
        
        return grad_input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors  
        grad_input = grad_output.clone()  

        grad_input[(input>=1) & (input<=1.5)] *= -1 
        grad_input[(input<0.5) | (input>1.5)] = 0
        return grad_input


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
    
class SpikeFunction_Voigt(Function):
    @staticmethod
    def forward(ctx, input, sigma=1.0, gamma = 1.0):
        voigt = voigt_profile(input.detach().numpy(), sigma, gamma)
        voigt = torch.tensor(voigt, dtype=input.dtype, device=input.device)  
        ctx.save_for_backward(input, torch.tensor(sigma), torch.tensor(gamma))
        return voigt

    @staticmethod
    def backward(ctx, grad_output):
        input, sigma, gamma = ctx.saved_tensors
        #TODO THIS DERIVATIVE IS SO ASS
        grad_input = grad_output * sigmoid * (1 - sigmoid)* alpha # Applique la dérivée de la sigmoïde.
        return grad_input, None
