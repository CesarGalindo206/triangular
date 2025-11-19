import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# Factor 2m/hbar^2 para la masa del electrón libre (Unidades: eV^-1 nm^-2)

K_FACTOR = 26.2468 

def calculate_transport(E_arr, V0, a, b, c):

    q2 = K_FACTOR * V0
    
    # Protección de geometría
    if b == a: b += 1e-9
    if c == b: c += 1e-9
    
    alpha1 = (q2 / (b - a))**(1/3)
    alpha2 = -1.0 * (q2 / (c - b))**(1/3)
    
    T_list = []
    R_list = []
    Sum_list = []
    
    for E in E_arr:
        if E <= 0: E = 1e-10
        
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
        
        # --- 4. Construcción de Gamma_11 ---
        # Estructura: (ik*u + alpha*u')
        term_h_11 = (h1*(1j*k*ai_1a*ai_2b + alpha1*aip_1a*ai_2b) +
                     h2*(1j*k*ai_1a*aip_2b + alpha1*aip_1a*aip_2b) +
                     h3*(1j*k*ai_1a*ai_2c + alpha1*aip_1a*ai_2c) +
                     h4*(1j*k*ai_1a*aip_2c + alpha1*aip_1a*aip_2c))
                     
        term_l_11 = (l4*(1j*k*bi_1a*bip_2c + alpha1*bip_1a*bip_2c) +
                     l3*(1j*k*bi_1a*bi_2c + alpha1*bip_1a*bi_2c) +
                     l1*(1j*k*bi_1a*bi_2b + alpha1*bip_1a*bi_2b) +
                     l2*(1j*k*bi_1a*bip_2b + alpha1*bip_1a*bip_2b))
        
        bracket_11 = - (term_h_11 + term_l_11)
        gamma11 = np.exp(1j * k * (c - 2*a)) * bracket_11
        
        # --- 5. Construcción de Gamma_21 ---
        # Estructura: (alpha*u' - ik*u)

        term_h_21 = (h1*(alpha1*aip_1a*ai_2b - 1j*k*ai_1a*ai_2b) +
                     h2*(alpha1*aip_1a*aip_2b - 1j*k*ai_1a*aip_2b) +
                     h3*(alpha1*aip_1a*ai_2c - 1j*k*ai_1a*ai_2c) +
                     h4*(alpha1*aip_1a*aip_2c - 1j*k*ai_1a*aip_2c))
        
        term_l_21 = (l4*(alpha1*bip_1a*bip_2c - 1j*k*bi_1a*bip_2c) +
                     l3*(alpha1*bip_1a*bi_2c - 1j*k*bi_1a*bi_2c) +
                     l1*(alpha1*bip_1a*bi_2b - 1j*k*bi_1a*bi_2b) +
                     l2*(alpha1*bip_1a*bip_2b - 1j*k*bi_1a*bip_2b))
                     
        bracket_21 = (term_h_21 + term_l_21) 
        gamma21 = np.exp(1j * k * c) * bracket_21
        
        
        # T
        num_T = 2j * k * alpha1 * alpha2
        den_T = (np.pi**2) * np.exp(1j * k * a) * gamma11
        T = np.abs(num_T / den_T)**2
        
        # R
        R = np.abs(gamma21 / gamma11)**2
        
        T_list.append(T)
        R_list.append(R)
        Sum_list.append(T + R)
        
    return np.array(T_list), np.array(R_list), np.array(Sum_list)

V0 = 0.23
a, b, c = 1.0, 4.0, 15.0 
E_vals = np.linspace(0, 0.4, 400)

T, R, Suma = calculate_transport(E_vals, V0, a, b, c)

plt.figure(figsize=(9, 6))

plt.plot(E_vals, T, linewidth=2, color='aquamarine', label='Transmisión $T$')
plt.plot(E_vals, R, linewidth=2, color='darkgoldenrod', linestyle='--', label='Reflexión $R$')
plt.plot(E_vals, Suma, linewidth=3, color='black', linestyle=':', label='Suma $T+R$')

plt.axvline(x=V0, color='k', linestyle='-', alpha=0.5, label='$V_0$')
plt.title(f'Verificación de Unitariedad ($V_0={V0}$ eV, a={a}, b={b}, c={c})')
plt.xlabel('Energía (eV)')
plt.ylabel('Probabilidad')
plt.xlim(0, 0.35)
plt.ylim(-0.05, 1.05)
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Valor medio de T+R: {np.mean(Suma):.10f}")