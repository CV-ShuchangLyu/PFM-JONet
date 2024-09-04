# SAM-JOANet

## CODE for AAAI 2025 Anonymous submission

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

## SAM-JOANet

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
  
Trained with the above commands, you can get a trained model to test the performance of your model.   

1. Testing commands

    ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mIoU   
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mFscore 
     ```

2. Testing cases: P2V_IRRG_64.33.pth and V2P_IRRG_59.65.pth : [google drive](https://drive.google.com/drive/folders/1qVTxY0nf4Rm4-ht0fKzIgGeLu4tAMCr-?usp=sharing)

    ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mIoU   
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mFscore 
     ```
     
     ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2 ./experiments/segformerb5/ST-DASegNet_results/V2P_IRRG_59.65.pth --eval mIoU   
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2 ./experiments/segformerb5/ST-DASegNet_results/V2P_IRRG_59.65.pth --eval mFscore 
     ```

The ArXiv version of this paper is release. [ST-DASegNet_arxiv](https://arxiv.org/pdf/2301.05526.pdf). This paper has been published on JAG, please refer to [Self-Training Guided Disentangled Adaptation for Cross-Domain Remote Sensing Image Semantic Segmentation](https://doi.org/10.1016/j.jag.2023.103646).

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.

# References
Many thanks to their excellent works
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [MMGeneration](https://github.com/open-mmlab/mmgeneration)
* [DAFormer](https://github.com/lhoyer/DAFormer)

