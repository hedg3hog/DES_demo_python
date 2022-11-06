import des
if __name__ == "__main__":
    key = des.Key()     #Generates pseudo random Key
    key.fromstring("NowTheStringIsTheKey")
    data, padd = des.from_file("file.txt")
    enc = des.encrypt(data, key)
    dec = des.decypt(enc, key)
    des.to_file("enc.txt", enc)
    des.to_file("dec.txt", dec, padd)