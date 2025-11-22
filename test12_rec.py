import numpy as np
import matplotlib.pyplot as plt

Tlist = []
Rlist = []
Suma = []
V_0 = 0.23
b = 6

for E in np.linspace(0.01, 1, 500):
    if E < V_0:
        
        q = np.sqrt(26.2468*(V_0 - E))    
        k = np.sqrt(26.2468*E)      
        T = 1 / (np.abs((np.cosh(q*b)+ (1j*(k**2-q**2)/(2*k*q))*np.sinh(q*b)))**2)
    
    else:
        
        q = np.sqrt(26.2468*(E - V_0))
        k = np.sqrt(26.2468*E)      
        T = 1 / (np.abs(np.cos(q*b)+ (1j*(k**2 + q**2)/(2*k*q))*np.sin(q*b))**2)
                
    Tlist.append(T)

for E in np.linspace(0.01, 1, 500):
    
    if E < V_0:
        
        q = np.sqrt(26.2468*(V_0 - E))  
        k = np.sqrt(26.2468*E)      
        R = ((- (k**2 + q**2)/(2*k*q) * np.sinh(q*b))**2)  * (1 / (np.abs((np.cosh(q*b)+ (1j*(k**2-q**2)/(2*k*q))*np.sinh(q*b)))**2))
    
    else:
        
        q = np.sqrt(26.2468*(E - V_0))
        k = np.sqrt(26.2468*E)      
        R = (( (k**2 - q**2)/(2*k*q) * np.sin(q*b))**2)  * (1 / (np.abs(np.cos(q*b)+ (1j*(k**2 + q**2)/(2*k*q))*np.sin(q*b))**2))
       
                
    Rlist.append(R)
    
for E in np.linspace(0.01, 1, 500):
    
    if E < V_0:
        
        q = np.sqrt(26.2468*(V_0 - E))  
        k = np.sqrt(26.2468*E)      
        T = 1 / (np.abs((np.cosh(q*b)+ (1j*(k**2-q**2)/(2*k*q))*np.sinh(q*b)))**2)
        R = ((- (k**2 + q**2)/(2*k*q) * np.sinh(q*b))**2)  * (1 / (np.abs((np.cosh(q*b)+ (1j*(k**2-q**2)/(2*k*q))*np.sinh(q*b)))**2))
    
    else:
        
        q = np.sqrt(26.2468*(E - V_0))
        k = np.sqrt(26.2468*E)      
        T = 1 / (np.abs(np.cos(q*b)+ (1j*(k**2 + q**2)/(2*k*q))*np.sin(q*b))**2)
        R = (( (k**2 - q**2)/(2*k*q) * np.sin(q*b))**2)  * (1 / (np.abs(np.cos(q*b)+ (1j*(k**2 + q**2)/(2*k*q))*np.sin(q*b))**2))
                
    Suma.append(T + R)

plt.plot(np.linspace(0.01, 1, 500), Suma, color='Aquamarine', linewidth=2)
plt.plot(np.linspace(0.01, 1, 500), Tlist, color='gray', linewidth=2)
plt.plot(np.linspace(0.01, 1, 500), Rlist, color='darkred', linewidth=2)
plt.axvline(x=V_0, color='k', linestyle='--', alpha=0.5, label='$V_0$')
plt.xlabel('E (eV)')
plt.ylabel('Transmisión T(E)')
plt.title('Transmisión')
plt.grid()
plt.show()

    