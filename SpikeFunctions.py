from enum import Enum
import torch
from torch.autograd import Function
from scipy.special import *

'''dans les fonctions backward on doit multiplier grad_input (le gradient des couches apres) 
par le gradient de la couche actuelle => derivation en chaine'''

"""
Cette class permet de calculer la sortie d'une fonction lors de la propagation avant et de personaliser la derivée lors de la retropropagation de l'erreur.
Voir cet exemple pour plus de détails : https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
"""
class SpikeFunction_Default(torch.autograd.Function):
    """
    Dans la passe avant, nous recevons un tenseur contenant l'entrée (potential-threshold).
    Nous appliquons la fonction Heaviside et renvoyons un tenseur contenant la sortie.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0 # On génère une décharge quand (potential-threshold) > 0
        return out

    """
    Dans la passe arrière, nous recevons un tenseur contenant le gradient de l'erreur par rapport à la sortie.
    Nous calculons le gradient de l'erreur par rapport à l'entrée en utilisant la dérivée de la fonction ReLu.
    """
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_relu = torch.ones_like(input) 
        grad_relu[input < 0] = 0          # La dérivée de la fonction ReLU
        return grad_output.clone()*grad_relu
    
    
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
class SpikeFunction_Triangular2(Function):
    @staticmethod
    def forward(ctx, input, theta=0, delta=1):
        ctx.save_for_backward(input)# Sauvegarde l'entrée pour la passe arrière.
        ctx.theta = theta
        ctx.delta = delta
        
        # Calcul de la fonction triangulaire en utilisant les paramètres theta et delta
        # 1. input - theta : Décalage de l'entrée par rapport au paramètre theta
        # 2. abs(input - theta) : Calcul de la valeur absolue de l'entrée par rapport au paramètre theta
        # 3. 1 - abs(input - theta) / delta : Applique une forme triangulaire centrée autour de theta avec une largeur de delta
        # 4. torch.maximum : Si la valeur est négative, on remplace par 0
        grad_input = torch.maximum(1 - torch.abs(input - theta) / delta, torch.tensor(0.0, device=input.device))
        return grad_input


class SpikeFunctionEnum(Enum):
    classical_relu = 1
    leaky_relu = 2
    abs_relu = 3
    heaviside = 4
    sigmoid = 5
    triangular = 6
    guassian = 7
    

spike_func_dict = {
    SpikeFunctionEnum.classical_relu:     (SpikeFunction_Classical_RELU,  None), 
    SpikeFunctionEnum.leaky_relu:         (SpikeFunction_Leaky_RELU,      None),
    SpikeFunctionEnum.abs_relu:           (SpikeFunction_Abs_RELU,        None),
    SpikeFunctionEnum.heaviside:            (SpikeFunction_Default,         None),
    SpikeFunctionEnum.triangular:         (SpikeFunction_Triangular,      None),
    SpikeFunctionEnum.sigmoid:            (SpikeFunction_Sigmoid,         {"alpha": 2.0}),
    SpikeFunctionEnum.guassian:           (SpikeFunction_Gaussian,        {"alpha": 1.0, "theta": 0.0})
}

class CustomActivation(torch.nn.Module):
    def __init__(self, activation_func: SpikeFunctionEnum):
        super(CustomActivation, self).__init__()
        if activation_func not in SpikeFunctionEnum:
            raise ValueError(f"Unsupported activation_func: {activation_func}")
        
        self.params = None
        self.activation_class, self.params = spike_func_dict[activation_func]

    def forward(self, x):
        if not self.params:
            return self.activation_class.apply(x)
        return self.activation_class.apply(x, **self.params)

