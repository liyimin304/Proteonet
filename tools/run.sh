arr1=("MS_ResWeightedPartNet50_FL3" "MS_ResWeightedPartNet50_FL7")
for subtest in {0..4}
do
  for method in "${arr1[@]}"
  do
    python tools/train.py models/resnet/${method}.py --kflod-validation ${subtest}
  done
done

#
##python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/train/HC/ models/resnet/MS_resnet50.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResNet/fold4/
##python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/test/HC/ models/resnet/MS_resnet50.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResNet/fold4/
##python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/train/igAN/ models/resnet/MS_resnet50.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResNet/fold4/
##python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/test/igAN/ models/resnet/MS_resnet50.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResNet/fold4/
##python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/train/MN/ models/resnet/MS_resnet50.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResNet/fold4/
##python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/test/MN/ models/resnet/MS_resnet50.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResNet/fold4/
##python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/train/DNK/ models/resnet/MS_resnet50.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResNet/fold4/
##python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/test/DNK/ models/resnet/MS_resnet50.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResNet/fold4/
#python tools/vis_pipeline.py datasets/train/MN models/resnet/MS_ResWeightedPartNet50_FL.py --show --number 10 --sleep 0.5 --output-dir aug_results
#
#python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/train/HC/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
#python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/test/HC/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
#python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/train/igAN/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
#python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/test/igAN/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
#python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/train/MN/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
#python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/test/MN/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
#python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/train/DNK/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
#python tools/vis_cam.py /home/huangjinze/code/MassSpectrumCls/datasets/test/DNK/ models/resnet/MS_ResWeightedPartNet50_FL.py --save-path /home/huangjinze/code/MassSpectrumCls/logs/ResPartNet/WeightedPartFLfold4/
#
##python tools/train.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 0
##python tools/train.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 1
##CUDA_VISIBLE_DEVICES=1 python tools/train.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 2
##CUDA_VISIBLE_DEVICES=1 python tools/train.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 3
##CUDA_VISIBLE_DEVICES=1 python tools/train.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 4
#
#python tools/evaluation.py models/resnet/MS_ResWeightedPartNet50_FL.py --kflod-validation 0
