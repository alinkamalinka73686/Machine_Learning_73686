from engine import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b + c
e = (d ** 2).relu()

print("d:", d)
print("e:", e)
print("e._op:", e._op)
print("e parents:", [p._op for p in e._prev])
print("d parents:", [p._op for p in d._prev])