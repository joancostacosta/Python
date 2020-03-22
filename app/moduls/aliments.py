# classe element
class Element:
    def __init__(self, nom, unitat):
        self.nom = nom
        self.unitat = unitat

# classe composició
class Composicio:
    def __init__(self, element, quantitat):
        self.element = element
        self.quantitat = quantitat

# classe aliment
class Aliment:
    def __init__(self, nom):
        self.nom = nom
        self.grup = ''
        self.unitat = ''
        self.maxapat = 0
        self.minapat = 0
        self.composicio = []

