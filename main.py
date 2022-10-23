import des
import numpy as np
if __name__ == "__main__":
    x = des.genPseudoRandomKey()
    print(des.gen_round_keys(x))