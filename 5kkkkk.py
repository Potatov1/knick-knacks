import numpy as np

i1 = int(input("enter value of i cap for first vector " ))
j1 = int(input("enter value of j cap for first vector " ))
k1 = int(input("enter value of k cap for first vector " ))

theta = float(input("enter value of angle in pi radians: "))


i2 = int(input("enter value of i cap for 2nd vector " ))
j2 = int(input("enter value of j cap for 2nd vector " ))
k2 = int(input("enter value of k cap for 2nd vector " ))


v1 = [i1,j1,k1]
v2 = [i2,j2,k2]

mag_v1 = (i1**2+j1**2+k1**2)**0.5

mag_v2 = (i2**2+j2**2+k2**2)**0.5


sum_mag = (mag_v1**2 + mag_v2**2 + 2*mag_v1*mag_v2*np.cos(theta*np.pi))**0.5


print("sum of the vectors is", (i1+i2),"i",(j1+j2),"j",(k1+k2),"k")


print( "sum of magnitude vectors :",sum_mag)



