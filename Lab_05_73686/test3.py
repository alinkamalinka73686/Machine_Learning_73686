from engine import Value


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = ((a * b + c) ** 2).relu()
e.backward()
print("e:", e.data)
print("a.grad:", a.grad, "(beklenen -24.0)")
print("b.grad:", b.grad, "(beklenen 16.0)")
print("c.grad:", c.grad, "(beklenen 8.0)")


x = Value(3.0)
y = x * x
y.backward()
print("x.grad:", x.grad, "(beklenen 6.0)")