import os

dir = os.fsencode("samples")

count=0
for file in os.listdir(dir):
    filename = os.fsdecode(file)
    print(filename)
    count+=1
    if count>10:
        break