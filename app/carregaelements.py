# CARREGA ELEMENTS DE JSON FILE A OBJECTES
import os
import json
from json import JSONEncoder

import moduls.aliments as al

# subclass JSONEncoder
class Encoder(JSONEncoder):
        def default(self, o): # pylint: disable=method-hidden
            return o.__dict__

# proves del path
script_path = os.path.abspath(__file__)
print("path: " + script_path)

# recuperem els elements del JSON file
with open('elements.json') as jsonfile:
    elemsJSONDataIn = json.load(jsonfile)
    jsonfile.close()
print(elemsJSONDataIn)

# creem la llista d'objectes Element
elems = []
try:
    decoded = json.loads(elemsJSONDataIn)
 
    # Access data
    for x in decoded:
        elems.append(al.Element(x['nom'], x['unitat']))
        #print(x['nom']) 
        #print(x['unitat'])

except (ValueError, KeyError, TypeError):
    print("JSON format error") 

print(elems[0].nom + " " + elems[0].unitat)
print(elems[len(decoded)-1].nom + " " + elems[len(decoded)-1].unitat)