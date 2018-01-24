from flask import Flask
from flask_restful import Resource, Api, reqparse
from applicate import *
import base64

app = Flask (__name__)
api = Api (app)

class DecodeB64Image (Resource):
  def post (self):
    try:
      parser = reqparse.RequestParser()
      parser.add_argument('image', type=str)
      args = parser.parse_args()
      with open ("tmp.tif", "wb") as f:
        x = str.encode (args['image'])
        x = base64.b64decode (x)
        f.write (x)

      del (x)
      del (args)
      del (parser)

    except Exception as e:
      return {'error': str(e)}

class TakeWSI (DecodeB64Image):
  def post (self):
    super().post()
    I = readTif ("tmp.tif")
    X = ratioWSI (I, 10)
    s = base64.b64encode (X)
    s = s.decode ('utf-8')
    del (I)

    return {'ratio' : s,
        'dtype' : str(X.dtype),
        'd0' : X.shape[0],
        'd1' : X.shape[1],
        'd2' : X.shape[2]}

class TakePatch (DecodeB64Image):
  def post (self):
    super().post()
    I = readTif ("tmp.tif")
    X = classifyPatch (I, 10)
    X = str (X)
    del (I)

    return {'class' : X}

api.add_resource (TakeWSI, '/wsi')
api.add_resource (TakePatch, '/patch')

if __name__ == '__main__':
  app.run(host='52.183.96.146',debug=True)
