import numpy as np
import matplotlib.pyplot as pyplot

pc = 1      # punt central (mínim de la funció)
st = 0.25   # desv. std. = distancia del punt central on la funció val la meitat del domini

Ri = (lambda x: 1 - np.e ** (-((x - pc)**2)/(2*st**2)), 
    lambda x: ((x-pc)/st**2) * np.e ** (-((x - pc)**2)/(2*st**2)) )

# Valores del eje X que toma el gráfico.
#x = range(0, 5, 0.001)
x = np.linspace(0, pc+st*4, 5000)

# Graficar funcion
pyplot.plot(x, [Ri[0](i) for i in x])
pyplot.plot(x, [Ri[1](i) for i in x])

# Establecer el color de los ejes.
pyplot.axhline(0, color="black")
pyplot.axvline(0, color="black")

# Limitar los valores de los ejes.
pyplot.xlim(0, pc+st*4)
pyplot.ylim(-3, 3)
# Guardar gráfico como imágen PNG.
#pyplot.savefig("output.png")
# Mostrarlo.
pyplot.show()