import numpy as np

class EnergyVAD:
    '''
    该类实现了一个基于信号能量的简单 VAD 算法
    '''

    def __init__(
        self,
        sample_rate: int = 16000,  #采样率：音频信号每秒钟采样的次数
        frame_length: int = 25,  #帧长度(毫秒)
        frame_shift: int = 20,  #帧移：连续两帧之间的偏移量，每处理完一帧后，
                                #向前移动20毫秒开始处理下一帧
        energy_threshold: float = 0.05,  #能量阈值：确定一帧语音是否有显著的语音活动，先计算每一帧信号的
                                         #能量（即所有样本点的平方和）然后将这个能量值与给定的阈值比较。
        pre_emphasis: float = 0.95,  #预加重系数：在语音信号预处理阶段，预加重是为了增强高频成分的一种
                                     #线性滤波操作。
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.energy_threshold = energy_threshold
        self.pre_emphasis = pre_emphasis

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        '''
        参数： 
            waveform (np.ndarray)：输入的波形数据，其形状为 (num_samples,)
        返回值： 
            np.ndarray：VAD（语音活动检测）输出结果，形状为 (num_frames,)
        '''

        new_waveform, is_stereo = self._convert_to_mono_if_needed(waveform) 

        # Pre-emphasis 音频预加重
        #计算方法是对语音信号进行卷积处理，每个样本点的值等于当前样本点减去前一个样本点乘以预加重系数的结果。
        new_waveform = np.append(new_waveform[0], new_waveform[1:] - self.pre_emphasis * new_waveform[:-1])
        
        # Compute energy 
        energy = self.compute_energy(new_waveform)

        # Compute VAD
        vad = self.compute_vad(energy)

        return vad

    def compute_energy(self, waveform: np.ndarray) -> np.ndarray:
        '''
        参数： 
            waveform (np.ndarray)：输入的波形数据，其形状为 (num_samples,)

        返回值： 
            np.ndarray：能量特征向量，形状为 (num_frames,)
        '''
        # Compute frame length and frame shift in number of samples (not milliseconds) 
        # 计算帧长度和帧移的样本数量（s为单位）
        frame_length = self.frame_length * self.sample_rate // 1000
        
        frame_shift = self.frame_shift * self.sample_rate // 1000

        # Compute energy 计算能量 
        # waveform.shape[0]= sample_rate
        energy = np.zeros((waveform.shape[0] - frame_length + frame_shift) // frame_shift)
        for i in range(energy.shape[0]):
            energy[i] = np.sum(waveform[i * frame_shift : i * frame_shift + frame_length] ** 2)

        return energy

    def compute_vad(self, energy: np.ndarray) -> np.ndarray:
        '''
        参数： 
            energy (np.ndarray): 能量特征数组，形状为 (num_frames,)
        返回值： 
            np.ndarray: 语音活动检测（VAD）的输出结果，形状为 (num_frames,)
        '''
        # Compute VAD
        vad = np.zeros(energy.shape)
        vad[energy > self.energy_threshold] = 1

        return vad

    def apply_vad(self, waveform: np.ndarray) -> np.ndarray:
        '''
        参数： 
            waveform (np.ndarray): 输入的波形数据，其形状为 (num_samples,)
        返回值：
            np.ndarray: 应用了VAD处理后的波形数据，其形状为 (num_samples,)
        '''
        processed_waveform, is_stereo = self._convert_to_mono_if_needed(waveform)

        vad = self(processed_waveform)

        shift = self.frame_shift * self.sample_rate // 1000
        new_waveform = []
        for channel in range(waveform.shape[0]):
            channel_waveform = []
            for i in range(len(vad)):
                if vad[i] == 1:
                    channel_waveform.extend(waveform[channel, i * shift : i * shift + shift])
            new_waveform.append(channel_waveform)

        new_waveform = np.array(new_waveform)

        return new_waveform
        
    def _convert_to_mono_if_needed(self, waveform: np.ndarray) -> np.ndarray:
        '''判断是否为单通道音频'''
        if waveform.ndim == 2 and waveform.shape[0] == 2:
            is_stereo = True
            print("Warning: stereo audio detected, using only the first channel")
            waveform = waveform[0]
            waveform = waveform[np.newaxis, :]
        else:
            is_stereo = False

        return waveform, is_stereo
        '''
        参数： 
            waveform (np.ndarray): 输入的波形数据，其形状为 (num_samples,)

        返回值： 
            np.ndarray: 调整维度后的波形，形状为 (1, num_samples) 
            bool: 如果信号是立体声则返回True，否则返回False
        '''