import numpy as np
from PIL import Image
import io
import base64
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

def process_image(base64_image):
    # 将 base64 编码的图像字符串转换为图像
    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    
    # 如果图像不是灰度图像，转换为灰度图像
    if image.mode != 'L':
        image = image.convert('L')

    # 将图像转换为 numpy 数组
    inpaint_mask = np.array(image) / 255.0

    # 创建 mask
    mask = Image.fromarray((inpaint_mask * 255).astype(np.uint8))

    # 创建 mask_gray
    mask_gray = Image.fromarray((inpaint_mask * 127).astype(np.uint8))

    return mask, mask_gray

from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


# import argparse
# parser = argparse.ArgumentParser(description='run ootd')
# parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
# parser.add_argument('--model_path', type=str, default="", required=True)
# parser.add_argument('--cloth_path', type=str, default="", required=True)
# parser.add_argument('--model_type', type=str, default="hd", required=False)
# parser.add_argument('--category', '-c', type=int, default=0, required=False)
# parser.add_argument('--scale', type=float, default=2.0, required=False)
# parser.add_argument('--step', type=int, default=20, required=False)
# parser.add_argument('--sample', type=int, default=4, required=False)
# parser.add_argument('--seed', type=int, default=-1, required=False)

from flask import Flask, request
import sys
import json

app = Flask(__name__)

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
    body = json.loads(request_body)
    
    # args
    model_type = body.get('model_type', "hd") # "hd" or "dc"
    category = body.get('category', 0) # 0:upperbody; 1:lowerbody; 2:dress
    # cloth base64
    cloth_image = body['cloth_img']
    # model base64
    model_image = body['model_img']
    # mask base64
    mask_image = body.get('mask_img', None)

    image_scale = body.get('scale', 2.0)
    n_steps = body.get('step', 20)
    n_samples = body.get('sample', 4)
    seed = body.get('seed', -1)

    gpu_id = body.get('gpu_id', 0)

    # initialize
    # args = parser.parse_args()


    openpose_model = OpenPose(gpu_id)
    parsing_model = Parsing(gpu_id)


    category_dict = ['upperbody', 'lowerbody', 'dress']
    category_dict_utils = ['upper_body', 'lower_body', 'dresses']

    # model_type = args.model_type # "hd" or "dc"
    # category = args.category # 0:upperbody; 1:lowerbody; 2:dress
    # # base64
    # cloth_path = args.cloth_path
    # # base64
    # model_path = args.model_path

    # image_scale = args.scale
    # n_steps = args.step
    # n_samples = args.sample
    # seed = args.seed

    if model_type == "hd":
        model = OOTDiffusionHD(gpu_id)
    elif model_type == "dc":
        model = OOTDiffusionDC(gpu_id)
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    # do something here
    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

    # cloth_img = Image.open(cloth_path).resize((768, 1024))
    cloth_img = Image.open(io.BytesIO(base64.b64decode(cloth_image))).resize((768, 1024))
    model_img = Image.open(io.BytesIO(base64.b64decode(model_image))).resize((768, 1024))

    if mask_image:
        mask, mask_gray = process_image(mask_image)
    else:
        keypoints = openpose_model(model_img.resize((384, 512)))
        model_parse, _ = parsing_model(model_img.resize((384, 512)))
        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask.save('./images_output/mask.jpg')
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    mask_gray.save('./images_output/mask_gray.jpg')
    
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    masked_vton_img.save('./images_output/masked_vton_img.jpg')

    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )

    image_idx = 0
    data = []
    for image in images:
        image.save('./images_output/out_' + model_type + '_' + str(image_idx) + '.png')
        image_idx += 1
        # 将图片转换为 BytesIO 对象
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # 将 BytesIO 对象转换为 base64 编码
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        # 将 Base64 字符串包装在 {"image": <base64>} 的格式中
        image_data = {"image": base64_image}

        # 将包含图像数据的字典附加到 data 列表中
        data.append(image_data)

    sys.stdout.flush()

    return render(data), 200, {"Content-Type": "application/json"}


if __name__ == "__main__":
    app.run()

# if __name__ == '__main__':

#     if model_type == 'hd' and category != 0:
#         raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

#     cloth_img = Image.open(cloth_path).resize((768, 1024))
#     model_img = Image.open(model_path).resize((768, 1024))
#     # keypoints = openpose_model(model_img.resize((384, 512)))
#     # model_parse, _ = parsing_model(model_img.resize((384, 512)))

#     # mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
#     mask = mask.resize((768, 1024), Image.NEAREST)
#     mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
#     masked_vton_img = Image.composite(mask_gray, model_img, mask)
#     masked_vton_img.save('./images_output/mask.jpg')

#     images = model(
#         model_type=model_type,
#         category=category_dict[category],
#         image_garm=cloth_img,
#         image_vton=masked_vton_img,
#         mask=mask,
#         image_ori=model_img,
#         num_samples=n_samples,
#         num_steps=n_steps,
#         image_scale=image_scale,
#         seed=seed,
#     )

#     image_idx = 0
#     for image in images:
#         image.save('./images_output/out_' + model_type + '_' + str(image_idx) + '.png')
#         image_idx += 1
