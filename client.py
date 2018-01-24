import base64
import requests
import json
import numpy as np
import matplotlib.pyplot as plt

def clientSubmit (name : str, url : str, output : str):
  url = "http://52.183.96.146:5000/" + url
  with open(name,"rb") as f:
    x = f.read()
    x = base64.b64encode (x)
    x = x.decode ('utf-8')

  r = requests.post (url, {'image':x})
  j = r.json ()
  del (x)
  del (r)

  if url == "patch":
    ans = int (j['class'])
    del (j)
    return ans

  elif url == 'wsi':
    b = j['ratio'].encode ()
    b = base64.b64decode (b)
    X = np.frombuffer (b, dtype=j['dtype'])
    X = X.reshape ((j['d0'], j['d1'], j['d2']))
    np.save (output, X)
    ans = X.shape

    del (j)
    del (b)
    del (X)
    return ans

  else:
    del (j)
    return -1

def main ():
  parser = argparse.ArgumentParser()
  parser.add_argument ("--image", help="image file path .tif")
  parser.add_argument ("--size", help="wsi / patch")
  parser.add_argument ("--output", help="save file path .npy")
  args = parser.parse_args ()
  if args.image == None or args.size == None or args.output == None:
    print ("argument missing")
    return -1
  if args.size != "wsi" and args.size != "patch":
    print ("unknown size")
    return -1
  return clientSubmit (args.image, args.size, args.output)

if __name__ == '__main__':
  main ()
