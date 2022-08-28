import argparse
import base64, uuid
import os, shutil
from pathlib import Path
from io import BytesIO
import time

import random
from flask import Flask, request, jsonify, session
from flask_cors import CORS, cross_origin
from consts import DEFAULT_IMG_OUTPUT_DIR
from utils import parse_arg_boolean, parse_arg_dalle_version
from consts import ModelSize

import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = "xyzzkjciiweuriu"
CORS(app)
print("--> Starting DALL-E Server. This might take up to two minutes.")

#from dalle_model import DalleModel
dalle_model = None

import img2img as i2i

parser = argparse.ArgumentParser(description = "A DALL-E app to turn your textual prompts into visionary delights")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
parser.add_argument("--model_version", type = parse_arg_dalle_version, default = ModelSize.MINI, help = "Mini, Mega, or Mega_full")
parser.add_argument("--save_to_disk", type = parse_arg_boolean, default = False, help = "Should save generated images to disk")
parser.add_argument("--img_format", type = str.lower, default = "JPEG", help = "Generated images format", choices=['jpeg', 'png'])
parser.add_argument("--output_dir", type = str, default = DEFAULT_IMG_OUTPUT_DIR, help = "Customer directory for generated images")
parser.add_argument("--gpu", type = str, help = "gpu info")

args = parser.parse_args()

#from jina import Client

from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import torch



def load_model_from_config(config_path = "configs/stable-diffusion/v1-inference.yaml", ckpt = "models/ldm/stable-diffusion-v1/model.ckpt", verbose=False):

    from ldm.util import instantiate_from_config

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 10.21)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 10.21)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

@app.route("/upscale", methods=["POST"])
@cross_origin()
def esrgan_api():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_name = "RealESRGAN_x4plus"
    model_path = os.path.join('./dalle-playground/backend/pretrained_models', model_name + '.pth')
    netscale = 4
    from realesrgan import RealESRGANer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None)

    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            
            
    json_data = request.get_json(force=True)
    img_formay = "JPEG" # args.img_format
    buffered = BytesIO()
    img.save(buffered, format=img_format)
    returned_generated_images = []
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    returned_generated_images.append(img_str)
    response = {'upscaledImgs': returned_generated_images, 'generatedImgsFormat': img_format}
    return jsonify(response)


@app.route("/dalle", methods=["POST"])
@cross_origin()
def generate_images_api():
    json_data = request.get_json(force=True)
    text_prompt = json_data["text"]
    num_images = json_data["num_images"]
    generated_imgs = dalle_model.generate_images(text_prompt, num_images)

    returned_generated_images = []
    if args.save_to_disk: 
        dir_name = os.path.join(args.output_dir,f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{text_prompt}")
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    for idx, img in enumerate(generated_imgs):
        if args.save_to_disk: 
          img.save(os.path.join(dir_name, f'{idx}.{args.img_format}'), format=args.img_format)

        buffered = BytesIO()
        img.save(buffered, format=args.img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        returned_generated_images.append(img_str)

    print(f"Created {num_images} images from text prompt [{text_prompt}]")
    
    response = {'generatedImgs': returned_generated_images,
    'generatedImgsFormat': args.img_format}
    return jsonify(response)

@app.route("/img2img", methods=["POST"])
@cross_origin()
def img2img():
    prompt_64, data, sid = request.form['data'].split(',')
    prompt = base64.b64decode(prompt_64).decode("utf-8")
    sid = base64.b64decode(sid).decode("utf-8")
    print(prompt)
    print('Session:', sid)
    image_data = base64.b64decode(data)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)

    StartDir = 'Images/%s' % sid
    Path(StartDir).mkdir(parents=True, exist_ok=True)

    StartImage = '%s/image__1.png' % StartDir
    cv2.imwrite( StartImage, image  )

    maxsize = 512 #@param ["512", "640", "704"] {type:"raw"}

    v_img = cv2.imread(StartImage) # vertical image
    scaled_v_img = resizeAndPad(v_img, (maxsize,maxsize), 127)

    h_img = cv2.imread(StartImage) # horizontal image
    scaled_h_img = resizeAndPad(h_img, (maxsize,maxsize), 127)

    sq_img = cv2.imread(StartImage) # square image
    scaled_sq_img = resizeAndPad(sq_img, (maxsize,maxsize), 127)

    Horizontal_Picture = scaled_h_img
    Vertical_Picture = scaled_v_img
    Crop_Picture = scaled_sq_img

    cv2.imwrite( StartImage, Horizontal_Picture  )
    sampler = DDIMSampler(app.model__)
    Strength = request.form.get('strength')
    steps = request.form.get('steps')
    NumIters = request.form.get('recur')

    Strength = float(Strength)
    steps = int(steps)
    NumIters = int(NumIters)

    if Strength >= 1:
      Strength = 0.9
    if Strength < 0.05:
      Strength = 0.05

    if steps >= 150:
      steps = 150
    if steps < 20:
      steps = 20
    
    if NumIters >= 20:
      NumIters = 20
    if NumIters < 2:
      NumIters = 2


    print(Strength, steps, NumIters)
    for r in range (1,NumIters):
        #@markdown Don't use short and simple prompts, complex prompts will give better results
        #prompt = "anime tentacles monster, dark color scheme, artstation, green lighting, hyperdetailed, insanely detailed and intricate, octane render, vfx, postprocessing, cinematic, alluring" #@param {type:"string"}
        #StartImage = "/content/stable-diffusion/ImageC/image_1.png"
        #@markdown The higher it is, the more the image will be modified [0.1 to 1]
         #@param {type:"slider", min:0, max:1, step:0.001}

        #@markdown The more steps you use, the better image you get, but I don't recommend using more than 150 steps
         #@param {type:"slider", min:1, max:150, step:1}
        Height = 256
        Width = 256



        #@markdown Setting
        Samples = 1 #@param ["1", "2", "3", "4"] {type:"raw"}
        Iteration = 1 #@param ["1", "2", "3", "4"] {type:"raw"}
        Seed = random.randrange(9999999999)
        CFGScale = 7.5 #@param {type:"slider", min:-2, max:20, step:0.1}

        #@title <---- Start generator

        print('>>', StartImage)
        grid_count = i2i.image2image(prompt = prompt, model=app.model__, sampler=sampler, plms=False, init_img = StartImage, outdir=StartDir, strength = Strength, ddim_steps = steps, H = Height, W = Width, n_samples = Samples, n_iter = Iteration, seed = Seed, scale = CFGScale, precision="autocast", skip_grid=False)

        print('<<', grid_count)
        img = cv2.imread(f"%s/grid-{grid_count-1:04}.png" % StartDir)

        #cv2_imshow(img)    
        old_file_name = f"%s/grid-{grid_count-1:04}.png"% StartDir
        new_file_name = "%s/image_1.png" % StartDir

        os.rename(old_file_name, new_file_name)

        target = new_file_name.replace("_1", "__%d" % r)
        shutil.copy(new_file_name, "%s" % target)

        print("%d The result is ready to be recycled: " % r, target)
        StartImage = target

    '''
    c = Client(host='grpc://172.28.0.2', port=51000)
    from docarray import Document
    d1 = Document()
    d1.text = StartImage
    c.post('/', [d1])
    '''
    print('NewFile:', new_file_name)
    with open(new_file_name, "rb") as img_file:
        encoded =  base64.b64encode(img_file.read()).decode('utf-8')
    response = {'generatedImg':"data:image/png;base64,"+encoded}

    return jsonify(response)



@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    imgs = []
    print('Session:', session)
    print(request.args)
    if 'sesh' in request.args:
        session['uuid'] = request.args['sesh']
        _dir = 'Images/%s/samples' % session['uuid']
        if os.path.isdir(_dir):
          files = os.listdir(_dir)
          for f in files:
              fd = open("%s/%s" % (_dir, f), "rb")
              encoded = "data:image/png;base64,"+ base64.b64encode(fd.read()).decode('utf-8')
              imgs.append(encoded)

    elif not 'uuid' in session:# or 'new' in request.args:
        session['uuid'] = uuid.uuid4()

    ret = {'success':True, 's':session['uuid'], 'gpu':args.gpu}
    if len(imgs):
      ret['imgs'] = imgs
    
    return jsonify(ret)


with app.app_context():
#    dalle_model = DalleModel(args.model_version)
#    dalle_model.generate_images("warm-up", 1)
    torch.no_grad()
    app.model__ = load_model_from_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("--> DALL-E Server is up and running!")
#    print(f"--> Model selected - DALL-E {args.model_version}")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False)




