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

## ST-DASegNet

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.4
        
    cuda >= 10.0
    
2. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

     ```
     cd ST-DASegNet
     
     pip install -e .
     
     chmod 777 ./tools/dist_train.sh
     
     chmod 777 ./tools/dist_test.sh
     ```

### Training

**mit_b5.pth** : [google drive](https://drive.google.com/drive/folders/1cmKZgU8Ktg-v-jiwldEc6IghxVSNcFqk?usp=sharing) For SegFormerb5 based ST-DASegNet training, we provide ImageNet-pretrained backbone here.

We select deeplabv3 and Segformerb5 as baselines. Actually, we use deeplabv3+, which is a more advanced version of deeplabv3. After evaluating, we find that deeplabv3+ has little modification compared to deeplabv3 and has little advantage than deeplabv3.

For LoveDA results, we evaluate on test datasets and submit to online server (https://github.com/Junjue-Wang/LoveDA) (https://codalab.lisn.upsaclay.fr/competitions/424). We also provide the evaluation results on validation dataset.

<table>
    <tr>
    <td><img src="PaperFigs\LoveDA_Leaderboard_Urban.jpg" width = "100%" alt="LoveDA UDA Urban"/></td>
    <td><img src="PaperFigs\LoveDA_Leaderboard_Rural.jpg" width = "100%" alt="LoveDA UDA Rural"/></td>
    </tr>
</table>

1. Potsdam IRRG to Vaihingen IRRG:

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2
     ```

2. Vaihingen IRRG to Potsdam IRRG:

    ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2
     ```

3. Potsdam RGB to Vaihingen IRRG:

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_PotsdamRGB2Vaihingen.py 2
     ```
     
4. Vaihingen RGB to Potsdam IRRG:

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2PotsdamRGB.py 2
     ```

5. LoveDA Rural to Urban

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config_LoveDA/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config_LoveDA/ST-DASegNet_segformerb5_769x769_40k_R2U.py 2
     ```

6. LoveDA Urban to Rural

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/deeplabv3/config_LoveDA/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_U2R.py 2
     ./tools/dist_train.sh ./experiments/segformerb5/config_LoveDA/ST-DASegNet_segformerb5_769x769_40k_U2R.py 2
     ```

7. LoveDA R-G-B Rural to LandCoverNet Sentinel-2

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/segformerb5/config_S2LoveDA/ST-DASegNet_segformerb5_769x769_40k_R2S.py 2
     ```

8. LoveDA R-G-B Rural to GID

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/segformerb5/config_GF2LoveDA/ST-DASegNet_segformerb5_769x769_40k_R2G.py 2
     ```

9. Paris to Chicago

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/segformerb5/config_Paris2Chicago/ST-DASegNet_segformerb5_769x769_40k_P2C.py 2
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

