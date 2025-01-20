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
import huggingface_hub
from SSIM_PIL import compare_ssim

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


model_ckpt_path = "D:\code\Img2Img\Img2Img-SC\stablediffusion\checkpoints\\v1-5-pruned.ckpt" #"G:/Giordano/stablediffusion/checkpoints/v1-5-pruned.ckpt"             #v2-1_512-ema-pruned.ckpt"
config_path     = "D:\code\Img2Img\Img2Img-SC\stablediffusion\checkpoints\\v1-inference.yaml"     #"G:/Giordano/stablediffusion/configs/stable-diffusion/v1-inference.yaml"




def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    # 加载检查点，加载检查点文件 ckpt，将模型加载到CPU上
    pl_sd = torch.load(ckpt, map_location="cpu")
    # 检查检查点中是否包含 global_step，如果有则打印该值，通常用于跟踪模型训练的进度
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    # 从加载的检查点中提取 state_dict，这是模型的权重和参数
    sd = pl_sd["state_dict"]
    # 实例化模型，函数根据提供的配置文件实例化模型。
    model = instantiate_from_config(config.model)
    # 加载模型权重
    m, u = model.load_state_dict(sd, strict=False)
    # 检查丢失和意外的键
    # 如果有丢失的键（模型中定义的参数未在检查点中找到）且 verbose 为真，则打印这些键。
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    # 如果有意外的键（检查点中有未在模型中定义的参数）且 verbose 为真，则打印这些键。
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # 将模型移至GPU并设置为评估模式
    # 将模型移动到GPU上以利用硬件加速。
    model.cuda()
    # 将模型设置为评估模式，以确保某些层（如 dropout 和 batch normalization）以推理模式运行。
    model.eval()
    return model

# 该函数的功能是从给定路径加载图像，调整其大小，归一化并转换为 PyTorch 张量格式，最后返回一个适合模型输入的图像张量。
def load_img(path):
    # 将图像打开的图像，转换为RGB格式，确保图像有三个通道（红、绿、蓝）。
    image = Image.open(path).convert("RGB")
    # 目标尺寸被固定为 (512, 512)
    w, h = (512,512)#image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    # 调整尺寸为64的倍数，使用 map 和 lambda 函数，将图像的宽度和高度调整为最接近的64的倍数。
    # 这种调整是为了确保后续处理（如卷积神经网络）可以顺利进行，因为某些网络架构要求输入尺寸为特定倍数。
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    # 转换为 NumPy 数组并归一化
    # 使用 PIL.Image.LANCZOS 插值方法将图像调整到新尺寸 (w, h)。LANCZOS 是一种高质量的抗锯齿重采样滤波器。
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # 将图像转换为 NumPy 数组，并将数据类型转换为 float32，除以255.0将像素值归一化到 [0, 1] 的范围。
    image = np.array(image).astype(np.float32) / 255.0

    # 改变数组维度顺序
    # image[None] 在第0维新增一个维度，代表批次大小。.transpose(0, 3, 1, 2) 将数组的维度
    # 从 (batch, height, width, channels)  转换为 (batch, channels, height, width)。
    image = image[None].transpose(0, 3, 1, 2)
    # 将 NumPy 数组转换为 PyTorch 张量，以便在模型中使用。
    image = torch.from_numpy(image)
    # 将像素值映射到 [-1, 1] 范围
    return 2. * image - 1.



    
# dataloader: 数据加载器，用于批量读取图像数据。 snr: 信噪比（Signal-to-Noise Ratio）。
# num_images: 要处理的图像总数。 batch_size: 每批次的图像数量。
# num_images_per_sample: 每个样本生成的图像数量。 outpath: 输出路径，用于保存生成的图像和文本。
# model: 用于生成图像的模型。 device: 设备（CPU或GPU）。
# sampler: 采样器，用于生成图像。 strength: 控制图像生成强度的参数。
# ddim_steps: DDIM采样步数。 scale: 调整生成图像质量的缩放参数。
def test(dataloader,
         snr=10,
         num_images=100,
         batch_size=1,
         num_images_per_sample=2,
         outpath='',
         model=None,
         device=None,
         sampler=None,
         strength=0.8,
         ddim_steps=50,
         scale=9.0):

    # 使用 transformers 库加载 BLIP 模型，该模型用于自动生成图像的描述。
    blip = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    i=0

    # 计算采样步骤，并根据指定步数和参数设置采样器的调度。
    sampling_steps = int(strength*50)
    print(sampling_steps)
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False) #attenzione ai parametri
    # 为保存生成图像的目录创建路径，如果不存在则创建目录。
    sample_path = os.path.join(outpath, f"Test-samples-{snr}-{sampling_steps}")
    os.makedirs(sample_path, exist_ok=True)

    text_path = os.path.join(outpath, f"Test-text-samples-{snr}-{sampling_steps}")
    os.makedirs(text_path, exist_ok=True)

    sample_orig_path = os.path.join(outpath, f"Test-samples-orig-{snr}-{sampling_steps}")
    os.makedirs(sample_orig_path, exist_ok=True)

    # 初始化 LPIPS（感知图像质量评估）和 SSIM（结构相似性指数）计算的工具和列表，用于存储评估结果。
    lpips = lp.LPIPS(net='alex')

    lpips_values = []

    ssim_values = []

    time_values = []

    # 循环处理图像数据
    # 使用 tqdm 进行进度跟踪，并遍历数据加载器中的每个批次。
    tq = tqdm(dataloader,total=num_images)
    for batch in tq:

        # 读取图像文件，使用 BLIP 模型生成图像描述。
        img_file_path = batch[0]

        #Open Image
        init_image = Image.open(img_file_path)

        #Automatically extract caption using BLIP model
        prompt = blip(init_image)[0]["generated_text"]  
        prompt_original = prompt 
        
        base_count = len(os.listdir(sample_path))

        # 预处理图像，将其加载到指定设备，并将其转换为模型的潜在表示。
        assert os.path.isfile(img_file_path)
        init_image = load_img(img_file_path).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
        
        #print(init_latent.shape,init_latent.type())
        
        '''
        CHANNEL SIMULATION
        '''

        # 使用 QAM16 调制模拟信道噪声。
        init_latent = qam16ModulationTensor(init_latent.cpu(),snr_db=snr).to(device)

        prompt = qam16ModulationString(prompt,snr_db=snr)  #NOISY BLIP PROMPT
       
        data = [batch_size * [prompt]]
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * ddim_steps)
        
        precision_scope = autocast 
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for n in range(1):
                        for prompts in data: 
                            start_time = time.time()
                            uc = None
                            if scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            # encode (scaled latent)
                            # 对潜在表示进行编码和解码，生成新的图像样本。
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                            # z_enc = init_latent
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc, )
                            x_samples = model.decode_first_stage(samples)

                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            end_time = time.time()
                            execution_time = end_time - start_time

                            time_values.append(execution_time)
    
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                #SAVE IMAGE
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                #SAVE TEXT
                                f = open(os.path.join(text_path, f"{base_count:05}.txt"),"a")        
                                f.write(prompt_original)
                                f.close()

                                #SAVE ORIGINAL IMAGE
                                init_image_copy = Image.open(img_file_path)
                                init_image_copy = init_image_copy.resize((512, 512), resample=PIL.Image.LANCZOS)
                                init_image_copy.save(os.path.join(sample_orig_path, f"{base_count:05}.png"))

                                # Compute SSIM
                                ssim_values.append(compare_ssim(init_image_copy, img))
                                base_count += 1
                            all_samples.append(x_samples)
                    
                    #Compute LPIPS
                    sample_out = (all_samples[0][0] * 2) - 1 
                    lp_score=lpips(init_image[0].cpu(),sample_out.cpu()).item()
                    
                    tq.set_postfix(lpips=lp_score)
                    
                    if not np.isnan(lp_score):
                        lpips_values.append(lp_score)


        i+=1
        if i== num_images:
            break

    # Calculate mean scores
    mean_lpips_score = sum(lpips_values) / len(lpips_values)
    mean_ssim_score = sum(ssim_values) / len(ssim_values)
    mean_time = sum(time_values) / len(time_values)

    print(f'mean lpips score at snr={snr} : {mean_lpips_score}')
    print(f'mean ssim score at snr={snr} : {mean_ssim_score}')
    print(f'mean time with sampling iterations {sampling_steps} : {mean_time}')

    # Write mean scores to a file
    results_file = os.path.join(outpath, f"results-i2i-snr-{snr}.txt")
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
        default="outputs/img2img-samples/i2i"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )


    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{config_path}")
    model = load_model_from_config(config, f"{model_ckpt_path}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    #INIZIO TEST

    #Strength is used to modulate the number of sampling steps. Steps=50*strength 
    test(test_dataloader,snr=10,num_images=100,batch_size=1,num_images_per_sample=1,outpath=outpath,
         model=model,device=device,sampler=sampler,strength=0.6,scale=9)

    test(test_dataloader,snr=8.75,num_images=100,batch_size=1,num_images_per_sample=1,outpath=outpath,
         model=model,device=device,sampler=sampler,strength=0.6,scale=9)

    test(test_dataloader,snr=7.50,num_images=100,batch_size=1,num_images_per_sample=1,outpath=outpath,
         model=model,device=device,sampler=sampler,strength=0.6,scale=9)

    test(test_dataloader,snr=6.25,num_images=100,batch_size=1,num_images_per_sample=1,outpath=outpath,
         model=model,device=device,sampler=sampler,strength=0.6,scale=9)

    test(test_dataloader,snr=5,num_images=100,batch_size=1,num_images_per_sample=1,outpath=outpath,
         model=model,device=device,sampler=sampler,strength=0.6,scale=9)

    test(test_dataloader,snr=3,num_images=100,batch_size=1,num_images_per_sample=1,outpath=outpath,
         model=model,device=device,sampler=sampler,strength=0.6,scale=9)

    test(test_dataloader, snr=12.5, num_images=100, batch_size=1, num_images_per_sample=1, outpath=outpath,
         model=model, device=device, sampler=sampler, strength=0.6, scale=9)

    test(test_dataloader, snr=15, num_images=100, batch_size=1, num_images_per_sample=1, outpath=outpath,
         model=model, device=device, sampler=sampler, strength=0.6, scale=9)

    test(test_dataloader, snr=17.5, num_images=100, batch_size=1, num_images_per_sample=1, outpath=outpath,
         model=model, device=device, sampler=sampler, strength=0.6, scale=9)

    test(test_dataloader, snr=20, num_images=100, batch_size=1, num_images_per_sample=1, outpath=outpath,
         model=model, device=device, sampler=sampler, strength=0.6, scale=9)

    test(test_dataloader, snr=25, num_images=100, batch_size=1, num_images_per_sample=1, outpath=outpath,
         model=model, device=device, sampler=sampler, strength=0.6, scale=9)