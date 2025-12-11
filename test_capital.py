#!/usr/bin/env python
print("Testing import paths...")

try:
    from capital.capital import Client
    print("OK: from capital.capital import Client")
except Exception as e:
    print("FAIL capital.capital:", e)

try:
    from capital import Client
    print("OK: from capital import Client")
except Exception as e:
    print("FAIL capital:", e)

try:
    import capital
    print("capital module dir:", dir(capital))
    print("capital file:", capital.__file__)
except Exception as e:
    print("FAIL import capital:", e)
