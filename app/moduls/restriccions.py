# classe restricció
class Restriccio:
    def __init__(self, ambit, entitat, tipo, cpte, cpteRef, maxim, minim):
        self.ambit = ambit
        self.entitat = entitat
        self.tipo = tipo
        self.cpte = cpte
        self.cpteRef = cpteRef
        self.maxim = maxim
        self.minim = minim

    def desv(self, D):
        # calcular la desviació de la restriccio per la dieta
        return 1
