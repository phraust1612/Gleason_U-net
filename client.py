import base64
import requests
import json
import numpy as np
import matplotlib.pyplot as plt

url = "http://52.183.96.146:5000/submit"
def clientTest (name : str):
  with open("xaal.tif","rb") as f:
    x = f.read()
    x = base64.b64encode (x)
    x = x.decode ('utf-8')

  r = requests.post (url, {'image':x})
  j = r.json ()
  b = j['ratio'].encode ()
  b = base64.b64decode (b)

  X = np.frombuffer (b, dtype=j['dtype'])
  X = X.reshape ((j['d0'], j['d1'], j['d2']))

  plt.imshow (X)
  plt.show()

def main ():
  parser = argparse.ArgumentParser()
  parser.add_argument ("--image", help="image file path .tif")
  rags = parser.parse_args ()
  clientTest (args.image)

if __name__ == '__main__':
  main ()
