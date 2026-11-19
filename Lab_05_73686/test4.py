from engine import Value

x = Value(0.5)
y = x.sigmoid().log()
y.backward()
print("y:", y.data)
print("x.grad:", x.grad)