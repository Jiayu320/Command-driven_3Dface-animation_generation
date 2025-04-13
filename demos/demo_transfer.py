import os, sys
import cv2
import numpy as np
from time import time
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

from audio import getInstruction as audioIns
from text import getInstrustion as textIns

from tqdm import tqdm
import glob
import re
from PIL import Image

def generatePose(ori_ten, tar_ten, frame_num):
    diff_ten = tar_ten - ori_ten
    timesteps = torch.linspace(0, 1, steps=frame_num + 1, device=ori_ten.device)
    timesteps = timesteps[:, None, None].expand(-1, *ori_ten.shape[:2]) 
    interpolated_seq = ori_ten[None, ...] + timesteps * diff_ten[None, ...]
    return interpolated_seq

def generateTarPose(change_pose, tar_level):
    if change_pose=='up':
        tar_pose = torch.tensor([[-0.33, 0., 0.]], dtype=torch.float32)
    elif change_pose=='down':
        tar_pose = torch.tensor([[0.33, 0., 0.]], dtype=torch.float32)
    elif change_pose=='left':
        tar_pose = torch.tensor([[0., -0.33, 0.]], dtype=torch.float32)
    elif change_pose=='right':
        tar_pose = torch.tensor([[0., 0.33, 0.]], dtype=torch.float32)
    elif change_pose=='Fixed':
        tar_pose = torch.tensor([[0., 0., 0.]], dtype=torch.float32)
    return tar_pose*tar_level

def generateTarLevel(change_level):
    if change_level=='Fixed':
        tar_level = 1
    elif change_level=='slight':
        tar_level = 0.8
    elif change_level=='large':
        tar_level = 1.2
    return tar_level

def sort_key(file_name):
    regex = re.compile(r'\d+')
    num = int(max(regex.findall(file_name)))
    return num

def images_to_gif(image_files, output_gif):
    frames = [Image.open(image) for image in image_files]
    frame_one = frames[0]
    frame_one.save(output_gif, format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)

def generate_gif_result(folder_path, output_gif_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    image_files.sort(key=sort_key)
    images_to_gif(image_files, output_gif_path)
    print(f"-- please check the results in  {output_gif_path}")

def concatenate_images(image1_path, image2_path, savecombinpath):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    
    width1, height1 = img1.size
    width2, height2 = img2.size

    if height1 != height2:
        target_height = min(height1, height2)
        target_width1 = int(width1 * (target_height / height1))
        target_width2 = int(width2 * (target_height / height2))
        img1 = img1.resize((target_width1, target_height))
        img2 = img2.resize((target_width2, target_height))
    
    combined_image = Image.new('RGB', (img1.width + img2.width, img1.height))
    combined_image.paste(img2, (0, 0))
    combined_image.paste(img1, (img2.width, 0))
    
    savepath = savecombinpath + str(sort_key(image1_path)) + '_result.jpg'
    # 保存结果
    if sort_key(image1_path) == sort_key(image2_path):
        combined_image.save(savepath)
    
def concatenate_single_emotion_images(image1_path, image2_path, savefolder):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    
    width1, height1 = img1.size
    width2, height2 = img2.size
    
    if height1 != height2:
        target_height = min(height1, height2)
        target_width1 = int(width1 * (target_height / height1))
        target_width2 = int(width2 * (target_height / height2))
        img1 = img1.resize((target_width1, target_height))
        img2 = img2.resize((target_width2, target_height))
    
    combined_image = Image.new('RGB', (img1.width + img2.width, img1.height))
    combined_image.paste(img2, (0, 0))
    combined_image.paste(img1, (img2.width, 0))
    
    savepath = os.path.join(savefolder, str(sort_key(image2_path)) + '_result.jpg')
    combined_image.save(savepath)
    
def generateInterpolatedNeckPose(change_pose, generate_frame, tar_level):
    if change_pose == 'shake':
        ori_neck_pose = generateTarPose('Fixed', tar_level)
        tar_neck_pose = generateTarPose('left', tar_level)
        interpolated_neck_pose_step1 = generatePose(ori_neck_pose, tar_neck_pose, int(generate_frame/4)) 
        ori_neck_pose = generateTarPose('left', tar_level)
        tar_neck_pose = generateTarPose('right', tar_level)
        interpolated_neck_pose_step2 = generatePose(ori_neck_pose, tar_neck_pose, int(generate_frame/2))
        ori_neck_pose = generateTarPose('right', tar_level)
        tar_neck_pose = generateTarPose('Fixed', tar_level)
        interpolated_neck_pose_step3 = generatePose(ori_neck_pose, tar_neck_pose, int(generate_frame/4))
        interpolated_neck_pose = torch.cat((interpolated_neck_pose_step1, interpolated_neck_pose_step2, interpolated_neck_pose_step3), 0)
    elif change_pose == 'nod':
        ori_neck_pose = generateTarPose('Fixed', tar_level)
        tar_neck_pose = generateTarPose('down', tar_level)
        interpolated_neck_pose_step1 = generatePose(ori_neck_pose, tar_neck_pose, int(generate_frame/2)) 
        ori_neck_pose = generateTarPose('down', tar_level)
        tar_neck_pose = generateTarPose('Fixed', tar_level)
        interpolated_neck_pose_step2 = generatePose(ori_neck_pose, tar_neck_pose, int(generate_frame/2)) 
        interpolated_neck_pose = torch.cat((interpolated_neck_pose_step1, interpolated_neck_pose_step2), 0)
    else:
        ori_neck_pose = generateTarPose('Fixed', tar_level)
        tar_neck_pose = generateTarPose(change_pose, tar_level)
        interpolated_neck_pose = generatePose(ori_neck_pose, tar_neck_pose, generate_frame) 
    return interpolated_neck_pose
    
def main(args):
    testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)
    #original_image = testdata[i]['original_image'][None, ...].to(device)
    emotion_driven_function = args.emotion_driven_function
    device = args.device
    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
    
    if emotion_driven_function == 'Text_driver' or emotion_driven_function == 'Audio_driver':
        # prepare Instruction dict
        generate_frame = args.generate_frame
        DATASET_PATH = 'Standard_expression'
        emotionComDict = textIns.prepareDict(DATASET_PATH)
        PoseDict = textIns.createPoseDict()
        LevelsDict = textIns.createLevelsDict()
        if emotion_driven_function == 'Text_driver':
            language = args.language
            OriIns = args.textInstrustion
            change_exp = textIns.getInstrustion(OriIns, emotionComDict, language)
            change_pose = textIns.getInstrustion(OriIns, PoseDict, language)
            change_level = textIns.getInstrustion(OriIns, LevelsDict, language)
            print('--------------------Text instrustion-------------------------')
            print('|The original instrustion is: '+ OriIns)
            print('|The target expression to modify is: ' + change_exp)
            print('|The target head pose to modify is: ' + change_pose)
            print('|The range of the target action is: ' + change_level)
            print('-------------------------------------------------------------')

        elif emotion_driven_function == 'Audio_driver':
            audio_path = args.audio_path
            language = args.language
            text = textIns.audio2Text(audio_path, language)

            change_exp = textIns.getInstrustion(text, emotionComDict, language)
            change_pose = textIns.getInstrustion(text, PoseDict, language)
            change_level = textIns.getInstrustion(text, LevelsDict, language)

            print('--------------------Audio instrustion------------------------')
            print('|The original instrustion is: '+ text)
            print('|The target expression to modify is: ' + change_exp)
            print('|The target head pose to modify is: ' + change_pose)
            print('|The range of the target action is: ' + change_level)
            print('-------------------------------------------------------------')

            if change_exp == 'Fixed' and change_pose == 'Fixed' and change_level == 'Fixed':
                print('---------------未检测出有效的语音指令请重新输入-----------------')
                exit(0)
        
        savefolder = args.savefolder + testdata[0]['imagename'] + '/'
        savefolder = os.path.join(savefolder, change_exp + change_pose + change_level)
        savefolder_text = os.path.join(savefolder, change_exp + change_pose + change_level + '_text')
        savefolder_comb = os.path.join(savefolder, change_exp + change_pose + change_level + '_comb')
        exp_path = args.exp_path   
        
        os.makedirs(savefolder, exist_ok=True)
        os.makedirs(savefolder_text, exist_ok=True)
        os.makedirs(savefolder_comb, exist_ok=True)
        tar_level = generateTarLevel(change_level)

        if change_exp != 'Fixed':

            interpolated_neck_pose = generateInterpolatedNeckPose(change_pose, generate_frame, tar_level)

            path = exp_path +'/'+ change_exp +'/'
            expdata = datasets.TestData(path, iscrop=args.iscrop, face_detector=args.detector)
            exp_image = expdata[0]['image'].to(device)[None,...]

            with torch.no_grad():
                exp_codedict = deca.encode(exp_image)
            tar_pose = exp_codedict['pose'][:,3:] * tar_level
            tar_exp = exp_codedict['exp'] * tar_level

            images = testdata[0]['image'].to(device)[None,...]
            with torch.no_grad():
                id_codedict = deca.encode(images)
            id_opdict, id_visdict = deca.decode(id_codedict)
            id_visdict = {x:id_visdict[x] for x in ['inputs', 'shape_detail_images']}
            ori_pose = id_codedict['pose'][:,3:]
            ort_exp = id_codedict['exp']

            interpolated_pose = generatePose(ori_pose, tar_pose, generate_frame)
            interpolated_exp = generatePose(ort_exp, tar_exp, generate_frame)

            for i in tqdm(range(generate_frame)):
                name = testdata[0]['imagename'] + '_' + change_exp + '_' + str(i)
                savepath = '{}/{}.jpg'.format(savefolder, name)

                id_codedict['pose'][:,3:] = interpolated_pose[i]
                id_codedict['exp'] = interpolated_exp[i]
                id_codedict['neck_pose']=interpolated_neck_pose[i]

                transfer_opdict, transfer_visdict = deca.decode(id_codedict)
                transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']
                id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']
                cv2.imwrite(os.path.join(savefolder, name + '_animation.jpg'), deca.visualize(id_visdict))

                image_name = name
                visdict = transfer_visdict; opdict = transfer_opdict

                vis_name = 'rendered_images'
                image  = util.tensor2image(visdict[vis_name][0])

                cv2.imwrite(os.path.join(savefolder_text, name + '_rendered.jpg'), image)

            savecombinpath = savefolder_comb + '/'
            image1_files = glob.glob(os.path.join(savefolder, '*.jpg'))
            image2_files = glob.glob(os.path.join(savefolder_text, '*.jpg'))

            image1_files.sort(key=sort_key)
            image2_files.sort(key=sort_key)
            for i in range(min(len(image1_files), len(image2_files))):
                image1 = image1_files[i]
                image2 = image2_files[i]
                concatenate_images(image2, image1, savecombinpath) 

            output_gif_path = './results/' + testdata[0]['imagename'] + '_edit_' + change_exp + '_' + change_pose + '_' + change_level +'.gif'
            generate_gif_result(savecombinpath, output_gif_path)

        else:
            interpolated_pose = generateInterpolatedNeckPose(change_pose, generate_frame, tar_level)
            for i in tqdm(range(generate_frame)):
                name = testdata[0]['imagename'] + '_pose_' + str(i)
                savepath = '{}/{}.jpg'.format(savefolder, name)
                images = testdata[0]['image'].to(device)[None,...]
                with torch.no_grad():
                    id_codedict = deca.encode(images)
                id_opdict, id_visdict = deca.decode(id_codedict)
                id_visdict = {x:id_visdict[x] for x in ['inputs', 'shape_detail_images']}   
                id_codedict['neck_pose']=interpolated_pose[i]

                transfer_opdict, transfer_visdict = deca.decode(id_codedict)
                transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']
                id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']
                cv2.imwrite(os.path.join(savefolder, name + '_animation.jpg'), deca.visualize(id_visdict))

                image_name = name
                visdict = transfer_visdict; opdict = transfer_opdict

                vis_name = 'rendered_images'
                image  = util.tensor2image(visdict[vis_name][0])

                cv2.imwrite(os.path.join(savefolder_text, name + '_rendered.jpg'), image)

            savecombinpath = savefolder_comb + '/'
            image1_files = glob.glob(os.path.join(savefolder, '*.jpg'))
            image2_files = glob.glob(os.path.join(savefolder_text, '*.jpg'))

            image1_files.sort(key=sort_key)
            image2_files.sort(key=sort_key)
            for i in range(min(len(image1_files), len(image2_files))):
                image1 = image1_files[i]
                image2 = image2_files[i]
                concatenate_images(image2, image1, savecombinpath) 


            output_gif_path = './results/' + testdata[0]['imagename'] + '_edit_' + change_exp + '_' + change_pose + '_' + change_level +'.gif'
            generate_gif_result(savecombinpath, output_gif_path)
    
    elif emotion_driven_function == 'Single_image_driver' or emotion_driven_function == 'Video_sequence_driver':
        # load test images 
        testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)
         
        device = args.device
        
        if emotion_driven_function == 'Single_image_driver':
            expdata = datasets.TestData(args.activate_exp_path, iscrop=args.iscrop, face_detector=args.detector)
       
            for n in range(len(expdata)):
                if expdata[n]['imagename'] == args.activate_exp_name:
                    num = n
                    break

            savefolder = os.path.join(args.savefolder, testdata[0]['imagename'])
            savefolder = os.path.join(savefolder, expdata[num]['imagename'])

            savefolder_text = os.path.join(savefolder, testdata[0]['imagename'] + '_text')
            if not os.path.exists(savefolder_text):
                os.makedirs(savefolder_text)

            savefolder_comb = os.path.join(savefolder, testdata[0]['imagename'] + '_comb')
            if not os.path.exists(savefolder_comb):
                os.makedirs(savefolder_comb)

            savefolder_last = os.path.join(savefolder, testdata[0]['imagename'] + '_last')
            if not os.path.exists(savefolder_last):
                os.makedirs(savefolder_last)
        
            generate_frame = args.generate_frame
            exp_image = expdata[num]['image'].to(device)[None,...]
            
            with torch.no_grad():
                exp_codedict = deca.encode(exp_image)
                
            tar_pose = exp_codedict['pose'][:,3:]
            tar_exp = exp_codedict['exp']
            
            images = testdata[0]['image'].to(device)[None,...]
            
            with torch.no_grad():
                id_codedict = deca.encode(images)
            id_opdict, id_visdict = deca.decode(id_codedict)
            id_visdict = {x:id_visdict[x] for x in ['inputs', 'shape_detail_images']}
        
            ori_pose = id_codedict['pose'][:,3:]
            ort_exp = id_codedict['exp']
        
            interpolated_pose = generatePose(ori_pose, tar_pose, generate_frame)
            interpolated_exp = generatePose(ort_exp, tar_exp, generate_frame)

            for i in tqdm(range(generate_frame)):
                name = testdata[0]['imagename'] + '_' + expdata[num]['imagename'] + '_' + str(i)
                savepath = '{}/{}.jpg'.format(savefolder, name)

                id_codedict['pose'][:,3:] = interpolated_pose[i]
                id_codedict['exp'] = interpolated_exp[i]

                transfer_opdict, transfer_visdict = deca.decode(id_codedict)
                transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']
                id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']
                cv2.imwrite(os.path.join(savefolder, name + '_animation.jpg'), deca.visualize(id_visdict))

                image_name = name
                visdict = transfer_visdict; 
                opdict = transfer_opdict

                vis_name = 'rendered_images'
                image  = util.tensor2image(visdict[vis_name][0])

                cv2.imwrite(os.path.join(savefolder_text, name + '_rendered.jpg'), image)

            savecombinpath = savefolder_comb + '/'
            image1_files = glob.glob(os.path.join(savefolder, '*.jpg'))
            image2_files = glob.glob(os.path.join(savefolder_text, '*.jpg'))

            image1_files.sort(key=sort_key)
            image2_files.sort(key=sort_key)
            for i in range(min(len(image1_files), len(image2_files))):
                image1 = image1_files[i]
                image2 = image2_files[i]
                concatenate_images(image2, image1, savecombinpath) 
            
            savecombinpathfin = savefolder_last + '/'
            image1_files = glob.glob(os.path.join(savecombinpath, '*.jpg'))
            # image2_files = glob.glob(os.path.join(args.activate_exp_path, '*.jpg'))
            image1 = args.activate_exp_path + '/' + args.activate_exp_name + '.jpg'
            for i in range(len(image1_files)):
                image2 = image1_files[i]
                concatenate_single_emotion_images(image1, image2, savecombinpathfin) 

            output_gif_path = './results/' + testdata[0]['imagename'] + '_edit_' + expdata[num]['imagename'] +'.gif'
            generate_gif_result(savecombinpathfin, output_gif_path)
        
        elif emotion_driven_function == 'Video_sequence_driver':
            activate_exp_path = args.activate_exp_path + '/' + args.activate_exp_name
            expdata = datasets.TestData(activate_exp_path, iscrop=args.iscrop, face_detector=args.detector)
       
            savefolder = os.path.join(args.savefolder, args.activate_exp_path)
            savefolder = os.path.join(savefolder, args.activate_exp_name)
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
                
            savefolder_text = os.path.join(savefolder, testdata[0]['imagename'] + '_text')
            if not os.path.exists(savefolder_text):
                os.makedirs(savefolder_text)

            savefolder_comb = os.path.join(savefolder, testdata[0]['imagename'] + '_comb')
            if not os.path.exists(savefolder_comb):
                os.makedirs(savefolder_comb)

            savefolder_last = os.path.join(savefolder, testdata[0]['imagename'] + '_last')
            if not os.path.exists(savefolder_last):
                os.makedirs(savefolder_last)
        
            generate_frame = len(expdata)
            
            images = testdata[0]['image'].to(device)[None,...]
            
            with torch.no_grad():
                id_codedict = deca.encode(images)
            id_opdict, id_visdict = deca.decode(id_codedict)
            id_visdict = {x:id_visdict[x] for x in ['inputs', 'shape_detail_images']}
        
            ori_pose = id_codedict['pose'][:,3:]
            ort_exp = id_codedict['exp']

            for i in tqdm(range(generate_frame)):
                name = testdata[0]['imagename'] + '_' + expdata[i]['imagename']
                savepath = '{}/{}.jpg'.format(savefolder, name)

                exp_image = expdata[i]['image'].to(device)[None,...]
            
                with torch.no_grad():
                    exp_codedict = deca.encode(exp_image)

                tar_pose = exp_codedict['pose'][:,3:]
                tar_exp = exp_codedict['exp']
                
                id_codedict['pose'][:,3:] = tar_pose
                id_codedict['exp'] = tar_exp

                transfer_opdict, transfer_visdict = deca.decode(id_codedict)
                transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']
                id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']
                cv2.imwrite(os.path.join(savefolder, name + '_animation.jpg'), deca.visualize(id_visdict))

                image_name = name
                visdict = transfer_visdict; 
                opdict = transfer_opdict

                vis_name = 'rendered_images'
                image  = util.tensor2image(visdict[vis_name][0])

                cv2.imwrite(os.path.join(savefolder_text, name + '_rendered.jpg'), image)

            savecombinpath = savefolder_comb + '/'
            image1_files = glob.glob(os.path.join(savefolder, '*.jpg'))
            image2_files = glob.glob(os.path.join(savefolder_text, '*.jpg'))

            image1_files.sort(key=sort_key)
            image2_files.sort(key=sort_key)
            for i in range(min(len(image1_files), len(image2_files))):
                image1 = image1_files[i]
                image2 = image2_files[i]
                concatenate_images(image2, image1, savecombinpath) 
            
            savecombinpathfin = savefolder_last + '/'
            
            image1_files = glob.glob(os.path.join(activate_exp_path, '*.jpg'))
            image2_files = glob.glob(os.path.join(savecombinpath, '*.jpg'))
            
            image1_files.sort(key=sort_key)
            image2_files.sort(key=sort_key)
            for i in range(min(len(image1_files), len(image2_files))):
                image1 = image1_files[i]
                image2 = image2_files[i]
                concatenate_images(image1, image2, savecombinpathfin) 

            output_gif_path = './results/' + testdata[0]['imagename'] + '_edit_' + args.activate_exp_name +'.gif'
            generate_gif_result(savecombinpathfin, output_gif_path)
    
    elif emotion_driven_function == 'Tar_AU_Parameter_driver':
        savefolder = args.savefolder + testdata[0]['imagename'] + '/'
        savefolder = os.path.join(savefolder, 'AUs')
        savefolder_text = os.path.join(savefolder, 'AUs' + '_text')
        savefolder_comb = os.path.join(savefolder, 'AUs' + '_comb')
        
        os.makedirs(savefolder, exist_ok=True)
        os.makedirs(savefolder_text, exist_ok=True)
        os.makedirs(savefolder_comb, exist_ok=True)

        Tar_AU_Parameter = args.Tar_AU_Parameter
        generate_frame = args.generate_frame
        
        Tar_AU_Parameter = torch.tensor([[Tar_AU_Parameter],
        [[0.0000, 0.1020, 0.0000, 0.0880, 0.0680, 0.0200, 0.0520, 0.0000,
          0.2120, 0.0000, 0.0000, 0.0000, 0.1200, 0.0000, 0.1500, 0.2820,
          0.0000]]]).to(device)
        
        generator_Pretrained_Model = args.generator_Pretrained_Model
        generator = torch.load(generator_Pretrained_Model)
        z = torch.randn(2, 50).to(device)
        
        parameter = generator(z, Tar_AU_Parameter).unsqueeze(1).data.cpu()
        result_parameter = parameter[0]
        result_parameter = result_parameter[0]
        
        tar_exp = result_parameter.flatten()[:50]
        tar_pose = result_parameter[-3:]
        
        print('--------------------Tar_AU_Parameter_driver------------------------')
        print('|The target AUs is: '+ str(args.Tar_AU_Parameter))
        print('|The target expression Parameter is: ')
        print(str(tar_exp))
        print('|The target jaw pose Parameter is: ')
        print(str(tar_pose))
        print('-------------------------------------------------------------')
        
        tar_exp = tar_exp.unsqueeze(0).to(device)
        tar_pose = tar_pose.unsqueeze(0).to(device)
        tar_level = 1
        change_pose = args.change_pose
        interpolated_neck_pose = generateInterpolatedNeckPose(change_pose, generate_frame, tar_level)

        images = testdata[0]['image'].to(device)[None,...]
        with torch.no_grad():
            id_codedict = deca.encode(images)
        id_opdict, id_visdict = deca.decode(id_codedict)
        id_visdict = {x:id_visdict[x] for x in ['inputs', 'shape_detail_images']}
        ori_pose = id_codedict['pose'][:,3:]
        ort_exp = id_codedict['exp']
        interpolated_pose = generatePose(ori_pose, tar_pose, generate_frame)
        interpolated_exp = generatePose(ort_exp, tar_exp, generate_frame)

        for i in tqdm(range(generate_frame)):
            name = testdata[0]['imagename'] + '_AUs' + '_' + str(i)
            savepath = '{}/{}.jpg'.format(savefolder, name)

            id_codedict['pose'][:,3:] = interpolated_pose[i]
            id_codedict['exp'] = interpolated_exp[i]
            id_codedict['neck_pose']=interpolated_neck_pose[i]

            transfer_opdict, transfer_visdict = deca.decode(id_codedict)
            transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']
            id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']
            cv2.imwrite(os.path.join(savefolder, name + '_animation.jpg'), deca.visualize(id_visdict))

            image_name = name
            visdict = transfer_visdict; opdict = transfer_opdict

            vis_name = 'rendered_images'
            image  = util.tensor2image(visdict[vis_name][0])

            cv2.imwrite(os.path.join(savefolder_text, name + '_rendered.jpg'), image)

        savecombinpath = savefolder_comb + '/'
        image1_files = glob.glob(os.path.join(savefolder, '*.jpg'))
        image2_files = glob.glob(os.path.join(savefolder_text, '*.jpg'))

        image1_files.sort(key=sort_key)
        image2_files.sort(key=sort_key)
        for i in range(min(len(image1_files), len(image2_files))):
            image1 = image1_files[i]
            image2 = image2_files[i]
            concatenate_images(image2, image1, savecombinpath) 

        output_gif_path = './results/' + testdata[0]['imagename'] + '_edit_AU' + '.gif'
        generate_gif_result(savecombinpath, output_gif_path)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    
    parser.add_argument('--emotion_driven_function', default='Text_driver', type=str, help='change the driving mode into Single_image_driver|Audio_driver|Text_driver|Video_sequence_driver|Tar_AU_Parameter_driver')
    
    parser.add_argument('--Tar_AU_Parameter', type=float, nargs='+', help='Please enter a list of Tar_AU_Parameter by spaces')
    parser.add_argument('--language', default='Chinese', type=str, help='change the language into Chinese|English')
    
    parser.add_argument('--textInstrustion', type=str, help='Please enter a text action command')
    
    parser.add_argument('--generate_frame', default=20, type=int, help='The number of frames that generate action changes')
    
    parser.add_argument('--generator_Pretrained_Model', default='30_generator.pt', type=str, help='path to Pretrained Model')
    
    parser.add_argument('--change_exp', default='Fixed', type=str, help='change expression into happy|sad|angry|disgusted|fearful|surprised')
    
    parser.add_argument('--change_pose', default='Fixed', type=str, help='change pose into left|right|up|down')
    
    parser.add_argument('--change_level', default='Fixed', type=str, help='change range of the target action into slight|great')
    
    parser.add_argument('-i', '--image_path', default='TestSamples/re_experiments/images_to_animate/monalisa.jpg', type=str,
                        help='path to input image')
    
    parser.add_argument('--audio_path', default='audio/test_data/up.wav', type=str,
                        help='path to input image')
    
    parser.add_argument('-e', '--exp_path', default='Standard_expression', type=str, 
                        help='path to expression')
    
    parser.add_argument('--activate_exp_path', default='Aimation_source', type=str, help='path to expression')
    parser.add_argument('--activate_exp_name', default='emotionme', type=str, help='path to expression')
    
    parser.add_argument('-s', '--savefolder', default='TestSamples/re_experiments/', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details' )
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'], 
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())

    main(parser.parse_args())

