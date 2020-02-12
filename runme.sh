
# wget "https://zenodo.org/record/3576403/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1" "Cnn14_DecisionLevelMax_mAP=0.385.pth"

cd /mnt/cephfs_new_wj/speechsv/kongqiuqiang/my_code_2019.12-/python/cvssp/pub_sound_event_detection_audioset
# python3 pytorch/sound_event_detection.py 2 --checkpoint_path="Cnn14_DecisionLevelMax_mAP=0.385.pth" --audio_path="examples/R9_ZSCveAHg_7s.wav" --cuda

git clone https://github.com/qiuqiangkong/audioset_tagging_cnn