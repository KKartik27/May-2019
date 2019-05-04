#Strings
name = "Sreeni Jilla"
namesq = 'xyz'

print(type(name))
print(type(namesq))

#access string content
print(name[0])
print(name[2:8])

#modify string content
name[0] = 'A' #Being string is immutable object, you can not edit

name2 = name + ' Hyd'
name = name + 'xyz'
print(name)
name = name.upper()
name = "mr"
name = name.capitalize()
print(name)
name = 'Sreeni Jilla'
name = concatenate(name, '10') #Use only + symbol for concatenate

name = name.replace('Mr','Miss')
print(name)
name = 10
name = 'Mumbaii'
isinstance(name, str) #True
isinstance(name, int) #False

          
          
          
          
          
          
          
          
          
