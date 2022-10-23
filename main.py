import des
import numpy as np
if __name__ == "__main__":
   print(des.DES.gen_round_keys(des.genPseudoRandomKey()))