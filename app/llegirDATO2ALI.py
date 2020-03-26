# CARREGA ALIMENTS I COMPOSICIO DE .DAT I EXPORTA A JSON
import json
from json import JSONEncoder

import moduls.aliments as al

# subclass JSONEncoder
class Encoder(JSONEncoder):
        def default(self, o): # pylint: disable=method-hidden
            return o.__dict__

# estructura de DATOALI.DAT
stru = (1, 24, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8)
elem = ('Calories', 'Proteïnes', 'Hidr.Carboni', 'Greixos', 'Gr.Saturats', 'Gr.Insaturats', 'Colesterol', 'Fibra', 'Sodi', 'Potasi', 'Calç', 'Fósfor', 'Magnesi', 'Ferro', 'Vitam. A', 'Vitam. B1', 'Riboflavina', 'Niacina', 'Vitam. C')
unit = ('cal', 'gr', 'gr', 'gr', 'gr', 'gr', 'mg', 'gr', 'gr', 'mg', 'mg', 'mg', 'mg', 'mg', 'mg', 'mg', 'mg', 'mg', 'mg')

ne = len(elem)

# muntem la taula d'elements
elems = []
for i in range(ne):
    elems.append(al.Element(elem[i], unit[i]))
#print(elems[0].nom)
#print(elems[ne-1].nom)

# muntem la taula d'aliments i composició
alims = []
i = 0
j = 0
with open('DATO2ALI.DAT', encoding="CP437", errors="surrogateescape") as f: 
    while True:
        buf = f.read(stru[i])
        if not buf:
            break
        if i == 0:
            # no tractem
            pass
        elif i == 1:
            # nou aliment
            alim = al.Aliment(buf)
            alim.unitat = 'gr'
            #print("nou aliment: " + alim.nom) 
            j = 0
        elif i == 2:
            # afegim grup
            alim.grup = buf
            #print("grup aliment: " + alim.grup) 
        else:
            # afegim components (quantitats origen x 100 grs)
            try:
                c = round(float(buf) / 100, 5)
            except:
                print("error buf a float: " + alim.nom + str(j) + buf)
            else:
                comp = al.Composicio(elems[j], c)
                alim.composicio.append(comp)
                #print("component: " + alim.nom + " " + elems[j].nom + " " + str(i) + " " + str(j) + " " + buf)
            j = j + 1
            # si hem processat tots els elements, afegim aliment
            if j == ne:
                alims.append(alim)
        i = (i + 1) % len(stru)
    f.close()
#print(alims)

'''# write to a file
with open("4forces.json","w") as f:
  json.dump(d, f)

# reads it back
with open("4forces.json","r") as f:
  d = json.load(f)'''

with open("elements.json","w") as f:
  json.dump(elems, f, indent=4, cls=Encoder, ensure_ascii=False)
  f.close()
'''with open('elements.json') as jsonfile:
    elemsJSONDataIn = json.load(jsonfile)
    jsonfile.close()
print(elemsJSONDataIn)'''

with open("aliments.json","w") as f:
  json.dump(alims, f, indent=4, cls=Encoder, ensure_ascii=False)
  f.close()
'''with open('aliments.json') as jsonfile:
    alimsJSONDataIn = json.load(jsonfile)
    jsonfile.close()
print(alimsJSONDataIn)'''

'''# convertim la list d'objectes Element a JSON
elemsJSONData = json.dumps(elems, indent=4, cls=Encoder, ensure_ascii=False)
print(elemsJSONData)

# guardem el JSON dels Element
with open('elements.json', 'w') as outfile:
    json.dump(elemsJSONData, outfile)
    outfile.close()

# recuperem els elements del JSON file
with open('elements.json') as jsonfile:
    elemsJSONDataIn = json.load(jsonfile)
    jsonfile.close()
print(elemsJSONDataIn)

# convertim la list d'objectes Aliment a JSON
alimsJSONData = json.dumps(alims, indent=4, cls=Encoder, ensure_ascii=False)
#print(alimsJSONData)

# guardem el JSON dels Aliment
with open('aliments.json', 'w') as outfile:
    json.dump(alimsJSONData, outfile)
    outfile.close()

# recuperem els aliments del JSON
with open('aliments.json') as jsonfile:
    alimsJSONDataIn = json.load(jsonfile)
    jsonfile.close()
print(alimsJSONDataIn)'''

'''print("Decode JSON formatted Data")
elemsJSON = json.loads(elemsJSONData)
print(elemsJSON)

jselems = json.dumps(elems)
print(jselems)'''

# Carreguem els objectes Element del fitxer JSON
