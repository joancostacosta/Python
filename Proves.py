import numpy as np
import matplotlib.pyplot as pyplot

def ri(x):
    return 1 - np.e ** (-((x - 1.5)**2)/(2*0.5**2))

# Valores del eje X que toma el gráfico.
#x = range(0, 5, 0.001)
x = np.linspace(0, 5, 5000)

# Graficar funcion
pyplot.plot(x, [ri(i) for i in x])

# Establecer el color de los ejes.
pyplot.axhline(0, color="black")
pyplot.axvline(0, color="black")

# Limitar los valores de los ejes.
pyplot.xlim(0, 5)
pyplot.ylim(-1, 2)
# Guardar gráfico como imágen PNG.
#pyplot.savefig("output.png")
# Mostrarlo.
pyplot.show()