import pymongo
import traceback

host = "localhost"
port = "27017"
database = "test"
collections_train = "deviceStats"
collections2 = "deviceStats"

def getMongoData(param, TIME_STEPS):
  try:
    client = pymongo.MongoClient(host, int(port))
    print("mongo connected+++", client)
    if(param == 1):
      db = client[database][collections_train]
      data = db.find({'dynamicStat': 1}).sort('_id', 1)
    if(param == 2):
      db = client[database][collections2]
      count = db.find({'dynamicStat': 1}).count()
      skip = count - TIME_STEPS
      data = db.find({'dynamicStat': 1}).sort('_id', 1).skip(skip).limit(10)
    if(param == 3):
      db = client[database][collections2]
      data = db.find({'dynamicStat': 1}).count()

    return data
  except Exception as e:
    print('Mongo Exception: ' + traceback.format_exc())
  finally:
        client.close()
        print('MongoDB Closed.')