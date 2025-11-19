import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# Factor 2m/hbar^2 para la masa del electrón libre
# Valor aprox: 26.2468 eV^-1 nm^-2
K_FACTOR = 26.2468 

def coeficiente_transmision(E_arr, V0, d):
    """
    Calcula T(E) para barrera triángulo rectángulo (subida vertical, bajada lineal).
    d: ancho total de la base (nm).
    """
    
    a = 0.0     #nm
    b = 1e-6  
    c = d     
    
    q2 = K_FACTOR * V0
    
    alpha1 = (q2 / (b - a))**(1/3)
    alpha2 = -1.0 * (q2 / (c - b))**(1/3) 
    geom_factor = (np.pi**2) / (alpha1 * alpha2)

    T_list = []
    
    for E in E_arr:
        if E <= 0: E = 1e-10 # Evitar error en k=0
        
        k = np.sqrt(K_FACTOR * E)
        k2 = k**2
        
        beta1 = - (1/alpha1**2) * ( (q2*a)/(b-a) + k2 )
        beta2 =   (1/alpha2**2) * ( (q2*c)/(c-b) - k2 )
        
        z1_a, z1_b = alpha1*a + beta1, alpha1*b + beta1
        z2_b, z2_c = alpha2*b + beta2, alpha2*c + beta2

        ai_1a, aip_1a, bi_1a, bip_1a = airy(z1_a)
        ai_1b, aip_1b, bi_1b, bip_1b = airy(z1_b)
        ai_2b, aip_2b, bi_2b, bip_2b = airy(z2_b)
        ai_2c, aip_2c, bi_2c, bip_2c = airy(z2_c)
        

        h1 = alpha1*alpha2 * bip_1b*bip_2c - alpha1*(1j*k) * bip_1b*bi_2c
        h2 = (1j*k)*alpha2 * bi_1b*bi_2c   - alpha2**2     * bi_1b*bip_2c
        h3 = alpha1*(1j*k) * bip_1b*bi_2b  - alpha2*(1j*k) * bi_1b*bip_2b
        h4 = alpha2**2     * bi_1b*bip_2b  - alpha1*alpha2 * bip_1b*bi_2b
        

        l1 = alpha1*alpha2 * aip_1b*aip_2c - alpha1*(1j*k) * aip_1b*ai_2c
        l2 = (1j*k)*alpha2 * ai_1b*ai_2c   - alpha2**2     * ai_1b*aip_2c
        l3 = alpha1*(1j*k) * ai_2b*aip_1b  - alpha2*(1j*k) * ai_1b*aip_2b
        l4 = alpha2**2     * ai_1b*aip_2b  - alpha1*alpha2 * ai_2b*aip_1b
        

        term_h = (h1*(1j*k*ai_1a*ai_2b + alpha1*aip_1a*ai_2b) +
                  h2*(1j*k*ai_1a*aip_2b + alpha1*aip_1a*aip_2b) +
                  h3*(1j*k*ai_1a*ai_2c + alpha1*aip_1a*ai_2c) +
                  h4*(1j*k*ai_1a*aip_2c + alpha1*aip_1a*aip_2c))
                  
        term_l = (l4*(1j*k*bi_1a*bip_2c + alpha1*bip_1a*bip_2c) +
                  l3*(1j*k*bi_1a*bi_2c + alpha1*bip_1a*bi_2c) +
                  l1*(1j*k*bi_1a*bi_2b + alpha1*bip_1a*bi_2b) +
                  l2*(1j*k*bi_1a*bip_2b + alpha1*bip_1a*bip_2b))
        

        gamma11_bracket = - (term_h + term_l)
 
        # T = | 1 / (Prefactor * Gamma) |^2
        # Prefactor = (e^ika / -2ik) * (pi^2 / a1*a2) * e^ik(c-2a)
        # Simplificamos exponentes: e^ika * e^ik(c-2a) = e^ik(c-a)

        fase_total = np.exp(1j * k * (c - a))
        
        denominador = (fase_total / (-2j * k)) * geom_factor * gamma11_bracket
        
        T = 1.0 / (np.abs(denominador)**2)
        T_list.append(T if T <= 1.0 else 1.0) # Clipping por seguridad numérica
        
    return np.array(T_list)

# Parámetros de la barrera y energía
V0 = 0.23         # eV
anchos = [2, 3, 4, 5, 6] # nm 
E_vals = np.linspace(0, 0.4, 400) # Rango de Energía 0 a 0.4 eV

plt.figure(figsize=(8, 6))

for d in anchos:
    T = coeficiente_transmision(E_vals, V0, d)
    plt.plot(E_vals, T, linewidth=2, label=f'd = {d} nm')

plt.axvline(x=V0, color='k', linestyle='--', alpha=0.5, label='$V_0$')
plt.title(f'Transmisión Barrera Triángulo Rectángulo ($V_0={V0}$ eV)')
plt.xlabel('Energía (eV)')
plt.ylabel('Coeficiente de Transmisión T')
plt.xlim(0, 0.35)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()