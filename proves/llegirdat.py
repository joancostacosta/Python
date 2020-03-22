'''import numpy as np

data = np.genfromtxt('F:/DATO2ALI.DAT', skip_header=1,                      skip_footer=1, names=True, dtype=None, delimiter=' ')

print(data)

datContent = [i.strip().split() for i in open('F:/DATO2ALI.DAT').readlines()]
print(datContent)

file = open('F:/DATO2ALI.DAT','r')

lines = file.readlines()

for line in lines[0:1]:
    print(line)
    print(line.split)'''


f = open("DATO2ALI.DAT")

print("123456789012345678901234567890")
#print(f.readline())

print(f.read(30))
f.close()
