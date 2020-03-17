#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time  : 2020/3/15 10:04
# @Author: zhouxing
# PRODUCT_NAME: PyCharm 
# 纯色与花色的分类模型，百度API结果

# url = https://aip.baidubce.com/rpc/2.0/ai_custom/v1/classification/colorMatch
# 接口文档：https://ai.baidu.com/ai-doc/EASYDL/Sk38n3baq

import base64
import json
import os

import numpy as np
import cv2
import requests
import pprint


def image_to_base64(image_path):
    frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    retval, buffer = cv2.imencode('.jpg', frame)
    base64_pic = base64.b64encode(buffer)
    base64_pic = base64_pic.decode()
    return base64_pic


def get_access_token(client_id, client_secret):
    """
    获取access_token
    :param client_id:
    :param client_secret:
    :return:
    """
    host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}'
    response = requests.get(host)
    if response:
        response = response.json()
        # print(response)
        return response["access_token"]
    assert print("获取 token error")


def color_match(access_token, image_base64, top_num=5):
    # 接口地址
    request_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/classification/colorMatch"

    params = {
        "image": image_base64,
        "top_num": top_num
    }
    request_url = request_url + "?access_token=" + access_token

    res = requests.post(url=request_url, data=json.dumps(params))
    # request.add_header('Content-Type', 'application/json')
    # response = urllib2.urlopen(request)
    # print(res.text)
    return res.text


if __name__ == '__main__':
    count = 0
    ok = 0
    root = "D:\\color\\images"
    for label in os.listdir(root)[::1]:
        for file_name in os.listdir(os.path.join(root, label)):
            file_path = os.path.join(root, label, file_name)
            access_token = get_access_token(client_id="8TT2Bw4MBxGIBjSdx2Mfven7",
                                            client_secret="L06KGXrK4GD6fGzIM5R6zdsk00mEXybu")
            image_base64 = image_to_base64(image_path=file_path)
            result = color_match(access_token=access_token, image_base64=image_base64)
            data = json.loads(result)
            class_1 = data["results"][0]["name"]
            score1 = data["results"][0]["score"]
            class_2 = data["results"][1]["name"]
            score2 = data["results"][1]["score"]
            if score1>score2:
                class_ = class_1
            else:
                class_ = class_2
            if int(class_)==int(label):
                ok += 1
            count += 1
            print(f"{ok}/{count}", f"{ok/count}")
