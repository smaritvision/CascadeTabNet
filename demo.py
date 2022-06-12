from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
# Load model
config_file = 'Config/cascade_mask_rcnn_hrnetv2p_w32_20e_v2.py'
checkpoint_file = 'Weights/epoch_36_v2.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Test a single image
img = r'Z:\Document\tables\test\Most recent bank statements (2 months worth) - Ally Savings April_Page3.png'

# Run Inference
result = inference_detector(model, img)

# Visualization results
#show_result_pyplot(model, img, result,('Bordered', 'cell', 'Borderless'), score_thr=0.85)
show_result_pyplot(model, img, result, score_thr=0.5)