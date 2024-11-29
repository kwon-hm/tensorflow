import pymongo

def conn(param):
  host = "localhost"
  port = "27017"
  database = "test"
  client = pymongo.MongoClient(host, int(port))

  print("mongo connected+++", client)

  db = client[database]
  connection = db[param]

  return connection