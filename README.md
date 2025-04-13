# Command-driven 3D face animation generation from in-the-wild image

The code is implemented mainly in Python language. 

## Requirements
To build the environment, you can follow the ipynb file **./setupImport.ipynb**, but first you need to follow [DECA: Detailed Expression Capture and Animation (SIGGRAPH 2021)](https://github.com/yfeng95/DECA) to build the baseline environment, then you can use the following instruction or the method written in **setupImport.ipynb** to build the rest environment. 

To build the environment you can run 
```bash
pip install -r requirements.txt
```

## Getting Started
There are five way to generate the 3D animation result. By using audio instruction (EN/CH), text instruction (EN/CH), single image, video sequence and action unites (AUs).

### Audio instruction
The running demo shows in **./Audio-Example.ipynb**, you can follow the ipynb settings to run the code.

#### English Audio instruction
<p align="center"> 
    <img src="./image_for_readme/EN-audio-runing.jpg">
</p>  
<p align="center">Running screenshot for audio command in English<p align="center">
<p align="center"> 
    <img src="./image_for_readme/me_edit_sad_shake_large.gif">
</p>  
<p align="center">Generation result for audio command in English<p align="center">

#### Chinese Audio instruction
<p align="center"> 
    <img src="./image_for_readme/CH-audio-runing.jpg">
</p>  
<p align="center">Running screenshot for audio command in Chinese<p align="center">
<p align="center"> 
    <img src="./image_for_readme/me_edit_Fixed_up_large.gif">
</p>  
<p align="center">Generation result for audio command in Chinese<p align="center">

### Text instruction
The running demo shows in **./Text-Example.ipynb**, you can follow the ipynb settings to run the code.
#### English Text instruction
<p align="center"> 
    <img src="./image_for_readme/EN-text-runing.jpg">
</p>  
<p align="center">Running screenshot for text command in English<p align="center">
<p align="center"> 
    <img src="./image_for_readme/me_edit_surprised_left_large.gif">
</p>  
<p align="center">Generation result for text command in English<p align="center">

#### Chinese Text instruction
<p align="center"> 
    <img src="./image_for_readme/CH-text-runing.jpg">
</p>  

<p align="center">Running screenshot for text command in Chinese<p align="center">
<p align="center"> 
    <img src="./image_for_readme/me_edit_disgusted_shake_slight.gif">
</p>  
<p align="center">Generation result for text command in Chinese<p align="center">

### Image instruction
The running demo shows in **./Single_image_driver-Example.ipynb**, you can follow the ipynb settings to run the code.

<p align="center"> 
    <img src="./image_for_readme/me_edit_emotionmy.gif">
</p>  
<p align="center">Generation result for image driven<p align="center">

### Video instruction
The running demo shows in **./Video_sequence_driver-Example.ipynb**, you can follow the ipynb settings to run the code.

<p align="center"> 
    <img src="./image_for_readme/me_edit_videoSequence.gif">
</p>  
<p align="center">Generation result for video driven<p align="center">

### AUs instruction
The running demo shows in **./AUs-Example.ipynb**, you can follow the ipynb settings to run the code.
<p align="center"> 
    <img src="./image_for_readme/AU-running.png">
</p>  
<p align="center">Running screenshot for AUs command<p align="center">
<p align="center"> 
    <img src="./image_for_readme/me_edit_AU.gif">
</p>  
<p align="center">Generation result for AUs command<p align="center">

## Acknowledgements
For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.
Here are some great resources benefit:

- [DECA: Detailed Expression Capture and Animation](https://github.com/yfeng95/DECA) for 3D reconstruction from wild image
- [GANimation](https://github.com/albertpumarola/GANimation) for inspire me the generation method
- [speech_recognition](https://github.com/Uberi/speech_recognition/tree/master) for audio-to-text processing
- [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) and [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME) for the FLAME model  
- [Pytorch3D](https://pytorch3d.org/), [neural_renderer](https://github.com/daniilidis-group/neural_renderer), [SoftRas](https://github.com/ShichenLiu/SoftRas) for rendering  
- [kornia](https://github.com/kornia/kornia) for image/rotation processing  
- [face-alignment](https://github.com/1adrianb/face-alignment) for cropping   
- [FAN](https://github.com/1adrianb/2D-and-3D-face-alignment) for landmark detection
- [face_segmentation](https://github.com/YuvalNirkin/face_segmentation) for skin mask
- [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) for identity loss  

We would also like to thank other recent public 3D face reconstruction works that allow us to easily perform quantitative and qualitative comparisons :)  
[RingNet](https://github.com/soubhiksanyal/RingNet), 
[Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/renderer/rasterize_triangles.py), 
[Nonlinear_Face_3DMM](https://github.com/tranluan/Nonlinear_Face_3DMM),
[3DDFA-v2](https://github.com/cleardusk/3DDFA_V2),
[extreme_3d_faces](https://github.com/anhttran/extreme_3d_faces),
[facescape](https://github.com/zhuhao-nju/facescape)
<!-- 3DMMasSTN, DenseReg, 3dmm_cnn, vrn, pix2vertex -->