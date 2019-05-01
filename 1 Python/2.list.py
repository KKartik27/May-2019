#Basic List containers

#For lists use [] brackets
list1 = [10,20,30]
print(type(list1))

list2=['abc', 10, True]
print(type(list2))

#access list elements by slicing
print(list1[0])
print(list1[0:2])
print(list1[2:3])

list1[2] = 100
print(list1)

list1.append(1) 
print(list1)

#extend expects a value/collection of values
list1.extend([10, 20])
list1.extend(list2)

#pop is for removing by index
list1.pop(1)
print(list1)
list3 = [list1, 60, list2]
len(list3)
