import sys
import importlib.metadata as md

print("Python", sys.version)
print()
for dist in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
    print(dist.metadata["Name"], dist.version)