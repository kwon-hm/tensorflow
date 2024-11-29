import pymongo
import traceback
from conf import config

host = config["host"]
port = config["port"]
database = config["database"]
collections_predict = config["collections"]

def getMongoData(param):
  try:
    client = pymongo.MongoClient(host, int(port))
    print("mongo connected+++", client)

    # get table name
    if(param['type'] == 1):
      db = client[database][collections_predict]
      data = db.find({'dynamicStat': 0, "deviceId": {'$exists': True}}).limit(1)

    # get train data
    if(param['type'] == 2):
      db = client[database][collections_predict + '_' + param['table']]
      data = db.find({'dynamicStat': 1}).sort('_id', 1)

    # get predict data count
    if(param['type'] == 3):
      db = client[database][collections_predict]
      data = db.find({'dynamicStat': 1}).count()

    # get predict data
    if(param['type'] == 4):
      db = client[database][collections_predict]
      count = db.find({'dynamicStat': 1}).count()
      skip = count - param['time_steps']
      data = db.find({'dynamicStat': 1}).sort('_id', 1).skip(skip).limit(param['time_steps'])

    return data
  except Exception as e:
    print('Mongo Exception: ' + traceback.format_exc())
  finally:
        client.close()
        print('MongoDB Closed.')