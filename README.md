❗ **This is a modified code from** [SAVN]( https://github.com/allenai/savn) **to reproduce the experiment results of <u>Visual Semantic Navigation using Scene Priors</u>** ❗

# Visual Semantic Navigation using Scene Priors
By Wei Yang, Xiaolong Wang, Ali Farhadi, Abhinav Gupta, Roozbeh Mottaghi

This code is modified to experiment the performance of the agent with the knowledge graph. Original code was implemented to use pre-extracted feature, but I modified it to use raw image to see and understand how the agent moves.


## Citing
Original code was the implementation of **"Learning to Learn How to Learn: Self-Adaptive Visual Navigation Using Meta-Learning"**
```
@InProceedings{Wortsman_2019_CVPR,
  author={Mitchell Wortsman and Kiana Ehsani and Mohammad Rastegari and Ali Farhadi and Roozbeh Mottaghi},
  title={Learning to Learn How to Learn: Self-Adaptive Visual Navigation Using Meta-Learning},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```
This code is about **"Visual Semantic Navigation using Scene Priors"**

```
@conference{Yang-2019-113270,
author = {Wei Yang and Xiaolong Wang and Ali Farhadi and Abhinav Gupta and Roozbeh Mottaghi},
title = {Visual semantic navigation using scene priors},
booktitle = {Proceedings of Seventh International Conference on Learning Representations (ICLR 2019)},
year = {2019},
month = {May},
}
```

## Setup

**Follow the original code setup**. Note that the data file sizes are huge.

- Clone the repository with `git clone https://github.com/obin-hero/savn.git && cd savn`.
- Install the necessary packages. If you are using pip then simply run `pip install -r requirements.txt`.
- Download the [pretrained models](https://prior-datasets.s3.us-east-2.amazonaws.com/savn/pretrained_models.tar.gz) and
[data](https://prior-datasets.s3.us-east-2.amazonaws.com/savn/data.tar.gz) to the `savn` directory. Untar with
```bash
tar -xzf pretrained_models.tar.gz
tar -xzf data.tar.gz
```

The `data` folder contains:

- `thor_offline_data` which is organized into sub-folders, each of which corresponds to a scene in [AI2-THOR](https://ai2thor.allenai.org/). For each room we have scraped the [ResNet](https://arxiv.org/abs/1512.03385) features of all possible locations in addition to a metadata and [NetworkX](https://networkx.github.io/) graph of possible navigations in the scene.
- `thor_glove` which contains the [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings for the navigation targets.
- `gcn` which contains the necessary data for the [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907) in [Scene Priors](https://arxiv.org/abs/1810.06543), including the adjacency matrix.

Note that the starting positions and scenes for the test and validation set may be found in `test_val_split`.

☝ You have to download [thor_offlline_data_with_images](https://prior-datasets.s3.us-east-2.amazonaws.com/savn/offline_data_with_images.tar.gz). As this code is implemented to use image data.



## Evaluation using Pretrained Models

Use the following code to run the pretrained models on the test set. Add the argument `--gpu-ids 0 1` to speed up the evaluation by using GPUs.

#### Scene Priors
```bash
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/gcn_pretrained.dat \
    --model GCN \
    --glove_dir ./data/gcn \
    --results_json scene_priors_test.json
    --images_file_name images.hdf5

cat scene_priors_test.json 
```
The video of each episode will be saved in *outputs*/ folder

The result may vary depending on system and set-up though we obtain:

|                      Model                       | SPL  &geq; 1 | Success  &geq; 1 | SPL   &geq; 5 | Success  &geq; 5 |
| :----------------------------------------------: | :----------: | :--------------: | :-----------: | :--------------: |
| [Scene Priors](https://arxiv.org/abs/1810.06543) |    14.86     |      36.90       |     11.49     |      24.70       |

