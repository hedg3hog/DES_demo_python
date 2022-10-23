import des
import numpy as np
if __name__ == "__main__":
    k = des.ascii_to_key("password123")
    d = des.from_file("/Users/rouvenbissinger/Desktop/test.txt")
    print(d)
    print(des.to_file("/Users/rouvenbissinger/Desktop/test2.txt", d))
