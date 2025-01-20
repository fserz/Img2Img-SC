import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from scripts.dataset import Flickr8kDataset,Only_images_Flickr8kDataset
from itertools import islice
from ldm.util import instantiate_from_config
from PIL import Image
import PIL
import torch
import numpy as np
import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm
import lpips as lp
from einops import rearrange, repeat
from torch import autocast
from tqdm import tqdm, trange
from transformers import pipeline
from scripts.qam import qam16ModulationTensor, qam16ModulationString
import time
from PIL import Image
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from SSIM_PIL import compare_ssim
from torchvision.transforms import Resize, ToTensor, Normalize, Compose


import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"



'''

INIT DATASET AND DATALOADER

'''
capt_file_path  =  "D:\code\Img2Img\Img2Img-SC\Flickr8kDataset\captions.txt"          #"G:/Giordano/Flickr8kDataset/captions.txt"
images_dir_path =  "D:/code/Img2Img/Img2Img-SC/Flickr8kDataset/Images/"                #"G:/Giordano/Flickr8kDataset/Images/"
batch_size      =  1    

dataset = Only_images_Flickr8kDataset(images_dir_path)

test_dataloader=DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True)


'''
MODEL CHECKPOINT

'''


model_ckpt_path = "D:\code\Img2Img\Img2Img-SC\stablediffusion\checkpoints\\v1-5-pruned.ckpt" #"G:/Giordano/stablediffusion/checkpoints/v1-5-pruned.ckpt"  #v2-1_512-ema-pruned.ckpt"
config_path     = "D:\code\Img2Img\Img2Img-SC\stablediffusion\checkpoints\\v1-inference.yaml"     #"G:/Giordano/stablediffusion/configs/stable-diffusion/v1-inference.yaml"


# 模型加载函数 用于从给定的配置和检查点路径加载模型。加载后的模型被移动到GPU，并设置为评估模式。
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
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


# 加载图像，将其转换为RGB模式，调整尺寸为512x512，归一化并转换为PyTorch张量。
def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = (512,512)#image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


    
# 进行一系列的测试：包括图像到文本的生成、文本噪声模拟、图像重建、以及SSIM和LPIPS值的计算。
# 将结果保存为图像文件和文本文件。
def test(dataloader,
         snr=10,
         num_images=100,
         sampling_steps = 50,
         outpath="outpath"
         ):

    # 使用Hugging Face的transformers库加载一个图像描述生成模型BLIP。
    blip = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    # 指定使用的Stable Diffusion模型ID。
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # 模型移到GPU运行
    pipe = pipe.to("cuda")
    # 定义一个图像转换管道，调整图像大小到512x512，并将其转换为PyTorch张量。
    transform = Compose([Resize((512,512), antialias=True), transforms.PILToTensor() ])
    # 使用LPIPS库计算感知相似度，加载AlexNet作为基准网络。
    lpips = lp.LPIPS(net='alex')


    sample_path = os.path.join(outpath,f'Test-TEXTONLY-sample-{snr}-{sampling_steps}')
    

    os.makedirs(sample_path, exist_ok=True)

    sample_orig_path = os.path.join(outpath,f'Test-TEXTONLY-sample-orig-{snr}-{sampling_steps}')

    os.makedirs(sample_orig_path, exist_ok=True)

    text_path = os.path.join(outpath,f'Test-TEXTONLY-text-{snr}-{sampling_steps}')

    os.makedirs(text_path, exist_ok=True)


    lpips_values = []
    time_values = []
    ssim_values = []

    i=0

    # 使用tqdm为dataloader添加进度条，迭代图片批次。
    for batch in tqdm(dataloader,total=num_images):

            # 获取批次中的第一个图片路径。
            img_file_path = batch[0]

            #Open Image
            init_image = Image.open(img_file_path)

            #Automatically extract caption using BLIP model
            # 使用BLIP模型生成图片描述。
            prompt_blip = blip(init_image)[0]["generated_text"]

            #Save Caption for Clip metric computation
            # 打开一个文件用于写入生成的描述。
            f = open(os.path.join(text_path, f"{i}.txt"),"a")        
            f.write(prompt_blip)
            f.close()


            #Introduce noise in the text (aka. simulate noisy channel)
            # 使用QAM16调制对文本引入噪声，模拟噪声通道。
            prompt_corrupted =  qam16ModulationString(prompt_blip,snr)

            #Compute time to reconstruct image
            time_start = time.time()
            #Reconstruct image using noisy text caption
            # 使用噪声文本生成图片
            image_generated = pipe(prompt_corrupted,num_inference_steps=sampling_steps).images[0]
            time_finish = time.time()

            time_elapsed = time_finish - time_start
            time_values.append(time_elapsed)

            #Save images for subsequent FID and CLIP Score computation
            # 保存生成的图片。
            image_generated.save(os.path.join(sample_path,f'{i}.png'))
            # 保存原始图片
            init_image.save(os.path.join(sample_orig_path,f'{i}.png'))

            #Compute SSIM
            # 调整原始图片的大小。
            init_image_copy = init_image.resize((512, 512), resample=PIL.Image.LANCZOS)
            # 计算并保存原始图片和生成图片之间的SSIM值。
            ssim_values.append(compare_ssim(init_image_copy, image_generated))

            #Compute LPIPS
            # 对生成的图片进行归一化和转换。
            image_generated = (transform(image_generated) / 255) *2 -1
            init_image = (transform(init_image) / 255 ) *2 - 1
            # 计算并保存LPIPS分数。
            lp_score=lpips(init_image.cpu(),image_generated.cpu()).item()
            lpips_values.append(lp_score)

            i+=1
            if i==num_images:
              break

    # Calculate mean scores
    # 计算平均LPIPS分数。
    mean_lpips_score = sum(lpips_values) / len(lpips_values)
    # 计算平均SSIM分数。
    mean_ssim_score = sum(ssim_values) / len(ssim_values)
    # 计算平均生成时间
    mean_time = sum(time_values) / len(time_values)

    print(f'mean lpips score at snr={snr} : {mean_lpips_score}')
    print(f'mean ssim score at snr={snr} : {mean_ssim_score}')
    print(f'mean time with sampling iterations {sampling_steps} : {mean_time}')

    # Write mean scores to a file
    results_file = os.path.join(outpath, f"results-t2i-snr-{snr}.txt")
    with open(results_file, "w") as f:
        f.write(f"Mean LPIPS score at SNR={snr}: {mean_lpips_score}\n")
        f.write(f"Mean SSIM score at SNR={snr}: {mean_ssim_score}\n")
        f.write(f"Mean time with sampling iterations {sampling_steps}: {mean_time}\n")

    return 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples/t2i"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )


    opt = parser.parse_args()
    seed_everything(opt.seed)


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    #START TESTING

    test(test_dataloader,snr=10,num_images=100,outpath=outpath)

    test(test_dataloader,snr=8.75,num_images=100,outpath=outpath)

    test(test_dataloader,snr=7.50,num_images=100,outpath=outpath)

    test(test_dataloader,snr=6.25,num_images=100,outpath=outpath)

    test(test_dataloader,snr=5,num_images=100,outpath=outpath)

    test(test_dataloader, snr=3, num_images=100, outpath=outpath)

    test(test_dataloader, snr=12.5, num_images=100, outpath=outpath)

    test(test_dataloader, snr=15, num_images=100, outpath=outpath)

    test(test_dataloader, snr=17.5, num_images=100, outpath=outpath)

    test(test_dataloader, snr=20, num_images=100, outpath=outpath)

    test(test_dataloader, snr=25, num_images=100, outpath=outpath)
    