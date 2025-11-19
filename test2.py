import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# --- CONSTANTES FÍSICAS ---
# Factor 2m/hbar^2 para la masa del electrón libre
# Unidades: eV^-1 nm^-2
K_FACTOR = 26.2468 

def transmission_general(E_arr, V0, a, b, c):
    """
    Calcula T(E) para una barrera triangular GENÉRICA.
    a, b, c: posiciones en nanómetros (nm).
    V0: Altura en eV.
    """
    
    q2 = K_FACTOR * V0
    
    # Protección por si el usuario pone b=a (pendiente infinita)
    if b == a: b += 1e-9
    if c == b: c += 1e-9
    
    # Alphas (Pendientes)
    alpha1 = (q2 / (b - a))**(1/3)
    alpha2 = -1.0 * (q2 / (c - b))**(1/3) # Raíz real negativa
    
    T_list = []
    
    for E in E_arr:
        if E <= 0: E = 1e-10 # Evitar k=0
        
        k = np.sqrt(K_FACTOR * E)
        k2 = k**2
        
        # Betas y Argumentos z = alpha*x + beta
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
        
        # Suma de términos (Bracket gigante)
        term_h = (h1*(1j*k*ai_1a*ai_2b + alpha1*aip_1a*ai_2b) +
                  h2*(1j*k*ai_1a*aip_2b + alpha1*aip_1a*aip_2b) +
                  h3*(1j*k*ai_1a*ai_2c + alpha1*aip_1a*ai_2c) +
                  h4*(1j*k*ai_1a*aip_2c + alpha1*aip_1a*aip_2c))
                  
        term_l = (l4*(1j*k*bi_1a*bip_2c + alpha1*bip_1a*bip_2c) +
                  l3*(1j*k*bi_1a*bi_2c + alpha1*bip_1a*bi_2c) +
                  l1*(1j*k*bi_1a*bi_2b + alpha1*bip_1a*bi_2b) +
                  l2*(1j*k*bi_1a*bip_2b + alpha1*bip_1a*bip_2b))
        
        # Gamma_11 = e^{ik(c-2a)} * (- Suma)
        bracket = - (term_h + term_l)
        gamma11 = np.exp(1j * k * (c - 2*a)) * bracket
        
        # --- FÓRMULA FINAL PARA T ---
        # T = | (2ik * a1 * a2) / (pi^2 * e^{ika} * Gamma11) |^2
        
        numerador = 2j * k * alpha1 * alpha2
        denominador = (np.pi**2) * np.exp(1j * k * a) * gamma11
        
        T = np.abs(numerador / denominador)**2
        
        # Clipping (opcional) para evitar valores > 1 por errores flotantes
        T_list.append(T if T <= 1.0 else 1.0)
        
    return np.array(T_list)

# --- SECCIÓN DE EJECUCIÓN (Modifica aquí tu geometría) ---

# 1. Definir parámetros
V0 = 0.23   # eV
E_vals = np.linspace(0, 0.4, 400) # eV

plt.figure(figsize=(9, 6))

# EJEMPLO: Comparar diferentes geometrías manteniendo el ancho total fijo en 5 nm
# a siempre en 0

# Caso 1: Triángulo Isósceles (Pico en el centro)
a1, b1, c1 = 0.0, 2.5, 5.0
T1 = transmission_general(E_vals, V0, a1, b1, c1)
plt.plot(E_vals, T1, linewidth=2, color='blue', label=f'Isósceles (b={b1}nm)')

# Caso 2: Triángulo Asimétrico (Pico hacia la izquierda)
a2, b2, c2 = 0.0, 1.0, 5.0
T2 = transmission_general(E_vals, V0, a2, b2, c2)
plt.plot(E_vals, T2, linewidth=2, color='green', linestyle='--', label=f'Asimétrico (b={b2}nm)')

# Caso 3: Triángulo Asimétrico (Pico hacia la derecha)
a3, b3, c3 = 0.0, 4.0, 5.0
T3 = transmission_general(E_vals, V0, a3, b3, c3)
plt.plot(E_vals, T3, linewidth=2, color='red', linestyle='-.', label=f'Asimétrico (b={b3}nm)')


# Detalles de la gráfica
plt.axvline(x=V0, color='k', linestyle=':', alpha=0.6, label='$V_0$')
plt.title(f'Transmisión para distintas geometrías ($V_0={V0}$ eV, Ancho=5nm)')
plt.xlabel('Energía (eV)')
plt.ylabel('Coeficiente de Transmisión T')
plt.xlim(0, 0.35)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()