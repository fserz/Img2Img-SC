from scipy.special import erfc
import random
import torch
import math
import bitstring
import numpy as np


'''
    This python file is dedicated to the QAM Modulation and noisy channel estimation 
    
    we used K-ary discerete memoryless channel (DMC) where the crossover probability is given by 
    the bit error rate (BER) of the M-ary quadrature amplitude modulation (QAM). 
   
    For DMC, you can find the channel model from (9) in https://ieeexplore.ieee.org/abstract/document/10437659. 
    
    For the crossover probability, you assumed an AWGN channel where the BER is a Q-function 
    of the SNR and M: https://www.etti.unibw.de/labalive/experiment/qam/. 

'''

# Modulate Tensor in 16QAM transmission and noisy channel conditions
def qam16ModulationTensor(input_tensor,snr_db=10):

  message_shape = input_tensor.shape

  message = input_tensor

  #Convert tensor in bitstream
  bit_list = tensor2bin(message)

  #Introduce noise to the bitstream according to SNR
  # bit_list_noisy = introduce_gaussian_noise(bit_list,snr=snr_db)
  # bit_list_noisy = introduce_rayleigh_fading_noise(bit_list,snr=snr_db)
  bit_list_noisy = introduce_rician_noise(bit_list,snr=snr_db)

  #Convert bitstream back to tensor 
  back_to_tensor = bin2tensor(bit_list_noisy)

  return back_to_tensor.reshape(message_shape)


# Modulate String in 16QAM transmission and noisy channel conditions
def qam16ModulationString(input_tensor,snr_db=10):

  message = input_tensor

  #Convert string to bitstream
  bit_list = list2bin(message)

  #Introduce noise to the bitstream according to SNR
  # bit_list_noisy = introduce_gaussian_noise(bit_list,snr=snr_db)
  # bit_list_noisy = introduce_rayleigh_fading_noise(bit_list,snr=snr_db)
  bit_list_noisy = introduce_rician_noise(bit_list,snr=snr_db)

  #Convert bitstream back to list of char
  back_to_tensor = bin2list(bit_list_noisy)

  return "".join(back_to_tensor)




# bit_list：一个列表，包含多个比特串（字符串形式）。
# snr（可选，默认值10）：信噪比（Signal-to-Noise Ratio），用来衡量信号与噪声的比值。
# qam（可选，默认值16）：表示使用的正交幅度调制（QAM）等级，默认为16-QAM。
def introduce_gaussian_noise(bit_list,snr=10,qam=16):

    # Compute ebno according to SNR
    # 将信噪比（SNR）转换为 Eb/No
    ebno = 10 ** (snr/10)

    # 误码率估计

    # Estimate probability of bit error according to https://www.etti.unibw.de/labalive/experiment/qam/
    # 计算每个符号包含的比特数，这里 qam=16 时 K=4
    K = np.sqrt(qam)  # 4
    # 计算可能的符号数目，对于 16-QAM，M=16。
    M = 2 ** K
    # 估计符号错误概率（公式来自特定的 QAM 参考）
    Pm = (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 / 2 / (M - 1) * K * ebno))
    # 符号错误率的概率。
    Ps_qam = 1 - (1 - Pm) ** 2
    # 比特错误率的概率，计算方式是将符号错误率除以每个符号的比特数 K。
    Pb_qam = Ps_qam / K


    # 逐比特引入噪声

    # 统计翻转的比特数量。
    bit_flipped = 0
    # 统计处理的总比特数量。
    bit_tot = 0
    # 存储引入噪声后的比特串列表。
    new_list = []
    # 遍历比特串
    for num in bit_list:
      num_new = []
      for b in num:

        # 引入噪声：使用 random.random() 生成随机数，如果小于 Pb_qam，翻转比特（从 0 变 1 或从 1 变 0）
        if random.random() < Pb_qam:
          num_new.append(str(1 - int(b)))  # Flipping the bit
          bit_flipped+=1
        else:
          num_new.append(b)
        bit_tot+=1
      # 组装新的比特串：将翻转后的比特串加入 new_list。
      new_list.append(''.join(num_new))

    #print(bit_flipped/bit_tot)
    return new_list

# 瑞利衰落噪声引入比特翻转
def introduce_rayleigh_fading_noise(bit_list, snr=10, qam=16, fading_factor=1.0):
        # Compute ebno according to SNR
        ebno = 10 ** (snr / 10)

        # Estimate probability of bit error according to QAM (as before, for reference)
        K = np.sqrt(qam)  # 4 for 16-QAM
        M = 2 ** K
        Pm = (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 / 2 / (M - 1) * K * ebno))
        Ps_qam = 1 - (1 - Pm) ** 2
        Pb_qam = Ps_qam / K

        bit_flipped = 0
        bit_tot = 0
        new_list = []

        for num in bit_list:
            num_new = []
            for b in num:
                # Simulating Rayleigh Fading noise by generating a Rayleigh-distributed random value
                fading_amplitude = np.random.rayleigh(scale=fading_factor)

                # Compare fading amplitude with a threshold to decide bit flip
                if fading_amplitude < Pb_qam:  # If fading is severe, flip the bit
                    num_new.append(str(1 - int(b)))  # Flip the bit
                    bit_flipped += 1
                else:
                    num_new.append(b)
                bit_tot += 1

            new_list.append(''.join(num_new))

        # Optional: Print bit flip ratio
        # print(bit_flipped / bit_tot)

        return new_list

# 莱斯噪声（Rician Noise）
def rician_noise(A, sigma):
    """Generate Rician noise with amplitude A and scale parameter sigma."""
    x = np.random.normal(loc=A, scale=sigma)
    y = np.random.normal(loc=0, scale=sigma)
    return np.sqrt(x**2 + y**2)

def introduce_rician_noise(bit_list, snr=10, qam=16, A=1, sigma=1):
    # Compute Eb/No according to SNR
    ebno = 10 ** (snr / 10)

    # Estimate probability of bit error according to QAM and Rician noise
    K = np.sqrt(qam)  # Number of bits per symbol
    M = 2 ** K
    Pm = (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 / 2 / (M - 1) * K * ebno))
    Ps_qam = 1 - (1 - Pm) ** 2
    Pb_qam = Ps_qam / K

    bit_flipped = 0
    bit_tot = 0
    new_list = []

    for num in bit_list:
        num_new = []
        for b in num:
            # Generate Rician noise
            noise = rician_noise(A, sigma)
            noise_probability = 1 / (1 + np.exp(-noise))  # Convert noise to a probability for bit flip

            # Introduce noise by flipping the bit with probability Pb_qam
            if random.random() < Pb_qam * noise_probability:
                num_new.append(str(1 - int(b)))  # Flip the bit
                bit_flipped += 1
            else:
                num_new.append(b)
            bit_tot += 1

        new_list.append(''.join(num_new))

    return new_list


def bin2float(b):
    ''' Convert binary string to a float.

    Attributes:
        :b: Binary string to transform.
    '''

    num = bitstring.BitArray(bin=b).float

    #print(num)
    if math.isnan(num) or math.isinf(num):
        
        num = np.random.randn()
      

    if num > 10:
     
      num=np.random.randn()

    if num < -10:
      
      num=np.random.randn()

    if num < 1e-2 and num>-1e-2:
      
      num = np.random.randn()

    return num


def float2bin(f):
    ''' Convert float to 64-bit binary string.

    Attributes:
        :f: Float number to transform.
    '''

    f1 = bitstring.BitArray(float=f, length=64)
    return f1.bin


def tensor2bin(tensor):

  tensor_flattened = tensor.view(-1).numpy()

  bit_list = []
  for number in tensor_flattened:
    bit_list.append(float2bin(number))

  
  return bit_list


def bin2tensor(input_list):
  tensor_reconstructed = [bin2float(bin) for bin in input_list]
  return torch.FloatTensor(tensor_reconstructed)


def string2int(char):
  return ord(char)


def int2bin(int_num):
  return '{0:08b}'.format(int_num)

def int2string(int_num):
  return chr(int_num)

def bin2int(bin_num):
  return int(bin_num, 2)


def list2bin(input_list):

  bit_list = []
  for number in input_list:
    bit_list.append(int2bin(string2int(number)))

  return bit_list

def bin2list(input_list):
  list_reconstructed = [int2string(bin2int(bin)) for bin in input_list]
  return list_reconstructed






