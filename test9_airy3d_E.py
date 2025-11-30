import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import airy


K_FACTOR = 26.2468 


def calculate_transport(E_arr, V0, a, b, c):
    q2 = K_FACTOR * V0
    
    if b == a: b += 1e-9
    if c == b: c += 1e-9
    
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
        
        num_T = 2j * k * alpha1 * alpha2
        den_T = (np.pi**2) * np.exp(1j * k * a) * gamma11
        T = np.abs(num_T / den_T)**2
        
        T_list.append(T)
        
    return np.array(T_list)



a, b, c = 3.0, 4.0, 5.0 


E_vals = np.linspace(0.01, 0.6, 100)  
V0_vals = np.linspace(0.1, 0.5, 80)   


E_grid, V0_grid = np.meshgrid(E_vals, V0_vals)
T_grid = np.zeros_like(E_grid)        


for i, v0 in enumerate(V0_vals):
    T_grid[i, :] = calculate_transport(E_vals, v0, a, b, c)


fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')



surf = ax.plot_surface(V0_grid, T_grid, E_grid, 
                       cmap='inferno', 
                       linewidth=0, 
                       antialiased=True, 
                       alpha=0.85)


ax.set_xlabel('\nAltura Barrera $V_0$ (eV)', fontsize=11)
ax.set_ylabel('\nTransmisión $T$', fontsize=11)
ax.set_zlabel('\nEnergía $E$ (eV)', fontsize=12)

ax.set_title(f'Transmitancia 3D - $V_0$,T,E\n(a={a}, b={b}, c={c} nm)', fontsize=14)

# Línea de Resonancia (E = V0)
line_V0 = [] 
line_T = []  
line_E = []  
for v in V0_vals:
    t_val = calculate_transport([v], v, a, b, c)[0]
    line_V0.append(v) 
    line_T.append(t_val) 
    line_E.append(v)  

ax.plot(line_V0, line_T, line_E, color='k', linewidth=3, linestyle='--', label='Límite $E=V_0$')

ax.view_init(elev=30, azim=-135)

fig.colorbar(surf, shrink=0.5, aspect=10, label='Nivel de Energía $E$')
plt.legend()
plt.tight_layout()
plt.show()