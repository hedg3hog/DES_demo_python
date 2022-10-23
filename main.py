import des
import numpy as np
if __name__ == "__main__":
    k = des.ascii_to_key("password123")
    d = des.string_to_array("hallo dies ist ein test um zu sehen ob meine funktion funktioniert")
    print(d)
