# PFM-JONet
## This paper has already published in TGRS.
This repo is the implementation of "Unsupervised Domain Adaptation for VHR Urban Scene Segmentation via Prompted Foundation Model Based Hybrid Training Joint-Optimized Network". We refer to  [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [mmagic](https://github.com/open-mmlab/mmagic). Many thanks to SenseTime and their two excellent repos.

<table>
    <tr>
    <td><img src="PaperFigs\Fig2.png" width = "100%" alt="PFM-JONet"/></td>
    </tr>
</table>

## Dataset Preparation

We select ISPRS (Postsdam/Vaihingen) and CITY-OSM (Paris/Chicago) as benchmark datasets.

**We follow [ST-DASegNet](https://github.com/cv516Buaa/ST-DASegNet) for detailed dataset preparation.**

<table>
<tr>
    <td><img src="PaperFigs\tree_data.png" width = "100%" alt="tree-data"/></td>
</tr>
</table>

## PFM-JONet

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.11
        
    cuda >= 11.7

   **This version depends on mmengine and mmcv (2.0.1)**
    
3. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

     ```
     cd PFM-JONet
     
     pip install -e .
     
     chmod 777 ./tools/dist_train.sh
     
     chmod 777 ./tools/dist_test.sh
     ```

### Training
1. ISPRS UDA-RSSeg task:

     ```
     cd PFM-JONet
     
     ./tools/dist_train.sh ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet.py 2
     ```

     ```
     ## We add LoRA training in 2025/04/09
     cd PFM-JONet
     
     ./tools/dist_train.sh ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet_Lora.py 2
     ```
     
2. CITY-OSM UDA_RSSeg task:

     ```
     cd PFM-JONet
     
    ./tools/dist_train.sh ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet_P2C.py 2
     ```

### Testing
  
Trained with the above commands, you can get your trained model to test the performance of your model.   

1. ISPRS UDA-RSSeg task:

     ```
     cd PFM-JONet
     
     ./tools/dist_test.sh ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet.py ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet_results/iter_11000_P2V_66.86.pth
     ```
     
2. CITY-OSM UDA_RSSeg task:

     ```
     cd PFM-JONet
     
    CUDA_VISIBLE_DEVICES=1 python ./tools/test.py ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet_P2C.py ./experiments/iter_35000_P2C_56.96.pth --show-dir ./P2C_results
     ```

[ArXiv version of this paper] (https://arxiv.org/abs/2411.05878).

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.

# References
Many thanks to their excellent works
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
* [mmagic](https://github.com/open-mmlab/mmagic)

# Please Cite
```
@ARTICLE{10976421,
  author={Lyu, Shuchang and Zhao, Qi and Sun, Yaxuan and Cheng, Guangliang and He, Yiwei and Wang, Guangbiao and Ren, Jinchang and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Unsupervised Domain Adaptation for VHR Urban Scene Segmentation via Prompted Foundation Model-Based Hybrid Training Joint-Optimized Network}, 
  year={2025},
  volume={63},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2025.3564216}}
```
