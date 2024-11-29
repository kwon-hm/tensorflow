def f(x):
    return x**2

def x에대한y변화량구하다(x):
    h=0.00000001
    return (f(x+h)-f(x))/(h)

print (x에대한y변화량구하다(x=1))
print (x에대한y변화량구하다(x=0.5))
print (x에대한y변화량구하다(x=0))
print (x에대한y변화량구하다(x=-1))