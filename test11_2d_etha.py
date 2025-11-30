import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

K_FACTOR = 26.2468 

def calculate_transport(E_arr, V0, a, b, c):
    q2 = K_FACTOR * V0
    
    if abs(b - a) < 1e-9: b = a + 1e-9
    if abs(c - b) < 1e-9: c = b + 1e-9
    
    alpha1 = (q2 / (b - a))**(1/3)
    alpha2 = -1.0 * (q2 / (c - b))**(1/3)
    
    T_list = []
    
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
        
        term_h = (h1*(1j*k*ai_1a*ai_2b + alpha1*aip_1a*ai_2b) +
                  h2*(1j*k*ai_1a*aip_2b + alpha1*aip_1a*aip_2b) +
                  h3*(1j*k*ai_1a*ai_2c + alpha1*aip_1a*ai_2c) +
                  h4*(1j*k*ai_1a*aip_2c + alpha1*aip_1a*aip_2c))
                  
        term_l = (l4*(1j*k*bi_1a*bip_2c + alpha1*bip_1a*bip_2c) +
                  l3*(1j*k*bi_1a*bi_2c + alpha1*bip_1a*bi_2c) +
                  l1*(1j*k*bi_1a*bi_2b + alpha1*bip_1a*bi_2b) +
                  l2*(1j*k*bi_1a*bip_2b + alpha1*bip_1a*bip_2b))
        
        bracket = - (term_h + term_l)
        gamma11 = np.exp(1j * k * (c - 2*a)) * bracket
        
        num_T = 2j * k * alpha1 * alpha2
        den_T = (np.pi**2) * np.exp(1j * k * a) * gamma11
        T = np.abs(num_T / den_T)**2
        
        T_list.append(T)
        
    return np.array(T_list)


V0 = 0.23      # eV
W = 5.0        # Ancho total (nm)
a = 0.0
c = W
E_vals = np.linspace(0.0, 0.4, 500) # Rango de energía

# Casos de Asimetría (eta)
etas = [0.0, 0.5, 1.0]
labels = [
    'Rectángulo ($\\eta=0$, $b \\to a$)', 
    'Asimétrico ($\\eta=0.5$, $b=W/3$)', 
    'Isósceles ($\\eta=1$, $b=W/2$)'
]
colors = ['darkred', 'darkgoldenrod', 'Aquamarine']
linestyles = ['-', '--', '-.']

plt.figure(figsize=(10, 7))

for i, eta in enumerate(etas):
    # Calcular la posición del pico b basada en eta
    # b = a + (eta / (1+eta)) * W
    if eta == 0:
        b = a 
    else:
        fraction = eta / (1.0 + eta)
        b = a + fraction * W
        
    T = calculate_transport(E_vals, V0, a, b, c)
    
    plt.plot(E_vals, T, label=labels[i], color=colors[i], linestyle=linestyles[i], linewidth=2)

# Detalles de la gráfica
plt.axvline(x=V0, color='gray', linestyle=':', label='Altura Barrera $V_0$')
plt.title(f'Comparación de Asimetría: Barrera Triangular\n(Ancho constante $W={W}$ nm, Altura $V_0={V0}$ eV)', fontsize=14)
plt.xlabel('Energía (eV)', fontsize=12)
plt.ylabel('Coeficiente de Transmisión $T$', fontsize=12)
plt.xlim(0, 0.4)
plt.ylim(0, 1.05)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()