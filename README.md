# SAM-HTJONet

This repo is the implementation of "Joint-Optimized Unsupervised Adversarial Domain Adaptation in Remote Sensing Segmentation with Prompted Foundation Model". We refer to  [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [mmagic](https://github.com/open-mmlab/mmagic). Many thanks to SenseTime and their two excellent repos.

<table>
    <tr>
    <td><img src="PaperFigs\Fig1.png" width = "100%" alt="SAM-JOANet"/></td>
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

## SAM-HTJONet

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.11
        
    cuda >= 11.7

   **This version depends on mmengine and mmcv (2.0.1)**
    
3. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

     ```
     cd SAM-JOANet
     
     pip install -e .
     
     chmod 777 ./tools/dist_train.sh
     
     chmod 777 ./tools/dist_test.sh
     ```

### Training
1. ISPRS UDA-RSSeg task:

     ```
     cd SAM-JOANet
     
     ./tools/dist_train.sh ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet.py 2
     ```
     
2. CITY-OSM UDA_RSSeg task:

     ```
     cd SAM-JOANet
     
    ./tools/dist_train.sh ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet_P2C.py 2
     ```

### Testing
  
Trained with the above commands, you can get your trained model to test the performance of your model.   

1. ISPRS UDA-RSSeg task:

     ```
     cd SAM-JOANet
     
     ./tools/dist_test.sh ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet.py ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet_results/iter_11000_P2V_66.86.pth
     ```
     
2. CITY-OSM UDA_RSSeg task:

     ```
     cd SAM-JOANet
     
    CUDA_VISIBLE_DEVICES=1 python ./tools/test.py ./experiments/SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet_P2C.py ./experiments/iter_35000_P2C_56.96.pth --show-dir ./P2C_results
     ```

The ArXiv version of this paper will be release soon.

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.

# References
Many thanks to their excellent works
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
* [mmagic](https://github.com/open-mmlab/mmagic)
