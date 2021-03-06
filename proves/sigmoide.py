import numpy as np
import matplotlib.pyplot as pyplot

centre = 0.0
stdev = 1.0

Ri = (lambda x: 1 / (1 + np.e ** ((centre-x)/stdev**2)), 
        lambda x: (1 / (1 + np.e ** ((centre-x)/stdev**2))) * (1 - (1 / (1 + np.e ** ((centre-x)/stdev**2)))) )

# Valores del eje X que toma el gráfico.
#x = range(0, 5, 0.001)
x = np.linspace(-10, 10, 500)

# Graficar funcion
pyplot.plot(x, [Ri[0](i) for i in x])
pyplot.plot(x, [Ri[1](i) for i in x])

# Establecer el color de los ejes.
pyplot.axhline(0, color="black")
pyplot.axvline(0, color="black")

# Limitar los valores de los ejes.
pyplot.xlim(-10, 10)
pyplot.ylim(-0.5, 1.5)
# Guardar gráfico como imágen PNG.
#pyplot.savefig("output.png")
# Mostrarlo.
pyplot.show()