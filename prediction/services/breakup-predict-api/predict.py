import json
import numpy as np
import util
import boto3
import os
import joblib


def handler(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }
    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }
    return response
    
    # dump=util.getModel()
    # # Preprocess
    # data=pre_process(event,dump)

    # # Predict the value and its probability
    # vendor,probability=predict(dump["model"],data)
    
    # return json.dumps({ "breakup": vendor,
    #                     "confidence": probability})


def pre_process(data,dump):
    pass

def predict(model,data):
    res=model.predict(data)[0]
    probability=sorted(model.predict_proba(data)[0], reverse=True)[0]
    probability=round(float(probability)*100,2)
    return res,probability


if __name__ == "__main__":
    handler('', '')