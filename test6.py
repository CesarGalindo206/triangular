import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# 1. Definir el dominio de z
z = np.linspace(-15, 5, 1000)

# 2. Calcular las funciones
ai, aip, bi, bip = airy(z)

# 3. Configuración de la Gráfica
plt.figure(figsize=(10, 6))

# Graficar Ai(z) con color aguamarina
plt.plot(z, ai, label='$Ai(z)$ (Decae)', color='aquamarine', linewidth=2)

# Graficar Bi(z) con color oro oscuro
plt.plot(z, bi, label='$Bi(z)$ (Crece)', color='darkgoldenrod', linewidth=2, linestyle='--')

# 4. Elementos visuales didácticos
plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
plt.text(0.2, 1.5, 'Punto de Retorno\n(z=0)', fontsize=10)

# Anotaciones de zonas físicas
plt.text(-10, 1.0, 'Región Oscilatoria\n($z < 0 \\Rightarrow E > V$)', 
         ha='center', fontsize=12, color='green',
         bbox=dict(facecolor='white', alpha=0.8))

plt.text(3, 1.0, 'Región Prohibida\n($z > 0 \\Rightarrow E < V$)', 
         ha='center', fontsize=12, color='purple',
         bbox=dict(facecolor='white', alpha=0.8))

# Límites y etiquetas
plt.ylim(-1, 2)
plt.xlim(-15, 5)
plt.axhline(0, color='black', linewidth=0.5)

plt.title('Comportamiento de las Funciones de Airy', fontsize=14)
plt.xlabel('Argumento $z$', fontsize=12)
plt.ylabel('Amplitud', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()