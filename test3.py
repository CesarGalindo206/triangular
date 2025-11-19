import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# --- CONSTANTES FÍSICAS ---
# Factor 2m/hbar^2 para la masa del electrón libre (Unidades: eV^-1 nm^-2)
K_FACTOR = 26.2468 

def transmission_general(E_arr, V0, a, b, c):
    """
    Calcula T(E) para una barrera triangular definida por a, b, c.
    """
    q2 = K_FACTOR * V0
    
    # Protección para pendientes infinitas (si b=a o c=b)
    if b == a: b += 1e-9
    if c == b: c += 1e-9
    
    # Alphas (Pendientes)
    alpha1 = (q2 / (b - a))**(1/3)
    alpha2 = -1.0 * (q2 / (c - b))**(1/3) # Forzamos raíz real negativa
    
    T_list = []
    
    for E in E_arr:
        if E <= 0: E = 1e-10 # Evitar división por cero en k
        
        k = np.sqrt(K_FACTOR * E)
        k2 = k**2
        
        # Betas y Argumentos z
        beta1 = - (1/alpha1**2) * ( (q2*a)/(b-a) + k2 )
        beta2 =   (1/alpha2**2) * ( (q2*c)/(c-b) - k2 )
        
        z1_a, z1_b = alpha1*a + beta1, alpha1*b + beta1
        z2_b, z2_c = alpha2*b + beta2, alpha2*c + beta2
        
        # Funciones de Airy
        ai_1a, aip_1a, bi_1a, bip_1a = airy(z1_a)
        ai_1b, aip_1b, bi_1b, bip_1b = airy(z1_b)
        ai_2b, aip_2b, bi_2b, bip_2b = airy(z2_b)
        ai_2c, aip_2c, bi_2c, bip_2c = airy(z2_c)
        
        # Coeficientes h (v=Bi)
        h1 = alpha1*alpha2 * bip_1b*bip_2c - alpha1*(1j*k) * bip_1b*bi_2c
        h2 = (1j*k)*alpha2 * bi_1b*bi_2c   - alpha2**2     * bi_1b*bip_2c
        h3 = alpha1*(1j*k) * bip_1b*bi_2b  - alpha2*(1j*k) * bi_1b*bip_2b
        h4 = alpha2**2     * bi_1b*bip_2b  - alpha1*alpha2 * bip_1b*bi_2b
        
        # Coeficientes l (u=Ai)
        l1 = alpha1*alpha2 * aip_1b*aip_2c - alpha1*(1j*k) * aip_1b*ai_2c
        l2 = (1j*k)*alpha2 * ai_1b*ai_2c   - alpha2**2     * ai_1b*aip_2c
        l3 = alpha1*(1j*k) * ai_2b*aip_1b  - alpha2*(1j*k) * ai_1b*aip_2b
        l4 = alpha2**2     * ai_1b*aip_2b  - alpha1*alpha2 * ai_2b*aip_1b
        
        # Suma de términos para Gamma_11
        term_h = (h1*(1j*k*ai_1a*ai_2b + alpha1*aip_1a*ai_2b) +
                  h2*(1j*k*ai_1a*aip_2b + alpha1*aip_1a*aip_2b) +
                  h3*(1j*k*ai_1a*ai_2c + alpha1*aip_1a*ai_2c) +
                  h4*(1j*k*ai_1a*aip_2c + alpha1*aip_1a*aip_2c))
                  
        term_l = (l4*(1j*k*bi_1a*bip_2c + alpha1*bip_1a*bip_2c) +
                  l3*(1j*k*bi_1a*bi_2c + alpha1*bip_1a*bi_2c) +
                  l1*(1j*k*bi_1a*bi_2b + alpha1*bip_1a*bi_2b) +
                  l2*(1j*k*bi_1a*bip_2b + alpha1*bip_1a*bip_2b))
        
        # Gamma_11
        bracket = - (term_h + term_l)
        gamma11 = np.exp(1j * k * (c - 2*a)) * bracket
        
        # --- FÓRMULA FINAL T ---
        # T = | (2ik * a1 * a2) / (pi^2 * e^{ika} * Gamma11) |^2
        
        numerador = 2j * k * alpha1 * alpha2
        denominador = (np.pi**2) * np.exp(1j * k * a) * gamma11
        
        T = np.abs(numerador / denominador)**2
        
        # Clipping por seguridad numérica (T <= 1)
        T_list.append(T if T <= 1.0 else 1.0)
        
    return np.array(T_list)

# --- PARÁMETROS AJUSTABLES ---
V0 = 0.23       # Altura de la barrera en eV

# Geometría en nm (a=1, b=2, c=3)
a = 1.0
b = 4.0
c = 9.0

# Rango de energía
E_vals = np.linspace(0, 0.4, 500) # De 0 a 0.4 eV

# --- CÁLCULO Y GRÁFICA ---
T = transmission_general(E_vals, V0, a, b, c)

plt.figure(figsize=(8, 6))
plt.plot(E_vals, T, linewidth=2, color='blue', label=f'a={a}, b={b}, c={c} nm')

plt.axvline(x=V0, color='k', linestyle='--', alpha=0.6, label='$V_0$ (Cima)')
plt.title(f'Coeficiente de Transmisión - Barrera Triangular ($V_0={V0}$ eV)')
plt.xlabel('Energía (eV)')
plt.ylabel('Transmisión $T$')
plt.xlim(0, 0.35)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()