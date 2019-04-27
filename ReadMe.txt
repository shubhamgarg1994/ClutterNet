This package provides testing code for DeepContext. It requries SUNRGBD dataset (http://rgbd.cs.princeton.edu/) to run. Please cite the following paper if you feel any part of the code is useful. Thanks!

@article{zhang2016deepcontext,
  title={DeepContext: context-encoding neural pathways for 3D holistic scene understanding},
  author={Zhang, Yinda and Bai, Mingru and Kohli, Pushmeet and Izadi, Shahram and Xiao, Jianxiong},
  journal={International Conference on Computer Vision},
  year={2017}
}

If you have any problem about the code or paper, please contact yindaz AT cs DOT princeton DOT edu.

Follow these step to run the code:
- run compile.sh in shell to compile marvin.
- run demo.m in MATLAB to get the testing result. PR curve will be plotted in the end.

Noted that most of the intermediate results are already provided. To get the PR curve and final result quickly, run
cd ./detection
script_evaluation

Several useful materials:
- We use a different train/test split than the original SUN-RGBD dataset. To find ours, check ./code/scene_data_new/train(test)_ids_mix.txt
- The pretrained models can be found in folders scene_template_classification, transformation, and detection
- We retrain deep sliding shape on our training split, and the result on our testing split can be found in ./detection/baseline_full
- Our result can be found in ./detection/experiment

