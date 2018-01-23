from flask import Flask
from flask_restful import Resource, Api, reqparse
from applicate import *
import base64

app = Flask (__name__)
api = Api (app)

class SubmitTiff(Resource):
  def post(self):
    try:
      parser = reqparse.RequestParser()
      parser.add_argument('image', type=str)
      args = parser.parse_args()
      with open ("tmp.tif", "wb") as f:
        x = str.encode (args['image'])
        x = base64.b64decode (x)
        f.write (x)

      I = readTif ("tmp.tif")
      X = ratioWSI (I, 10)
      s = base64.b64encode (X)
      s = s.decode ('utf-8')

      del (I)
      del (X)

      return {'ratio' : s,
          'dtype' : str(I.dtype),
          'd0' : I.shape[0],
          'd1' : I.shape[1],
          'd2' : I.shape[2]}
    except Exception as e:
      return {'error': str(e)}

api.add_resource(SubmitTiff, '/submit')

if __name__ == '__main__':
  app.run(host='52.183.96.146',debug=True)
