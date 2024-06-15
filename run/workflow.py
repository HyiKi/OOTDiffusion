from flask import Flask, request
import sys
import json
import requests

app = Flask(__name__)

# {
#     "model_type": "hd", # "hd" or "dc"
#     "category": 0, # 0:upperbody; 1:lowerbody; 2:dress
#     "garment": "", # 服饰图片 base64
#     "mask": "", # 蒙板图片 base64 可为空
#     "scale": 2.0, # 可为空
#     "steps": 30, # 可为空
#     "seed": -1, # 可为空
#     "type": "idm" # "idm" or "ootd"
# }


# response[data][0][image]
def request_idm(image, step):
    url = "http://220.168.146.21:8627"
    model_type = step["model_type"]
    category = step["category"]
    garment = step["garment"]
    mask = step.get("mask", None)
    scale = step.get("scale", 2.0)
    steps = step.get("steps", 30)
    seed = step.get("seed", -1)
    payload = json.dumps(
        {
            "model_type": model_type,
            "category": category,
            "human_image": image,
            "garment_image": garment,
            "mask_image": mask,
            "scale": scale,
            "steps": steps,
            "seed": seed,
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    response_json = response.json()
    # 获取所需的值
    image_value = response_json["data"][0]["image"]
    return image_value

# response[data][0][image]
def request_ootd(image, step):
    url = "http://220.168.146.21:8099"
    model_type = step["model_type"]
    category = step["category"]
    cloth_img = step["garment"]
    mask_img = step.get("mask", None)
    scale = step.get("scale", 2.0)
    steps = step.get("steps", 20)
    seed = step.get("seed", -1)
    payload = json.dumps(
        {
            "model_type": model_type,
            "category": category,
            "model_img": image,
            "cloth_img": cloth_img,
            "mask_img": mask_img,
            "scale": scale,
            "steps": steps,
            "seed": seed,
            "sample": 1
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)
    response_json = response.json()
    # 获取所需的值
    image_value = response_json["data"][0]["image"]
    return image_value

def render(data) :
    result_dict = {}
    result_dict['code'] = 'SUCCESS'
    result_dict['msg'] = 'success'
    result_dict['data'] = data
    return json.dumps(result_dict)

@app.route("/", methods=["POST"])
def handler():
    for k, v in request.headers.items():
        if k.startswith("HTTP_"):
            # process custom request headers
            pass

    request_body = request.data
    request_method = request.method
    path_info = request.path
    content_type = request.content_type
    query_string = request.query_string.decode("utf-8")

    print("request_body: {}".format(request_body))
    print(
        "method: {} path: {} query_string: {}".format(
            request_method, path_info, query_string
        )
    )

    # do something here
    body = json.loads(request_body)
    # image base64
    image = body["image"]
    # workflow: model_type category garment mask scale steps seed type
    workflow = body["workflow"]

    for step in workflow:
        type = step["type"].lower()
        if type == "idm":
            image = request_idm(image, step)
        elif type == "ootd":
            image = request_ootd(image, step)

    sys.stdout.flush()
    return render({"image":image}), 200, {"Content-Type": "application/json"}


if __name__ == "__main__":
    app.run()
