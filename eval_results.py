
## evaluate the results using COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# test LMO
# cocoGt = COCO("datasets/lmo/test/000002/scene_gt_coco.json")
# cocoDt = cocoGt.loadRes("datasets/lmo/test/000002/weight_samH_coco_instances_results.json")

# test YCBV
# cocoGt = COCO("datasets/ycbv/test/scene_gt_coco_all_v2.json")
# cocoDt = cocoGt.loadRes("datasets/ycbv/test/weight_adapted_samH_coco_instances_results_prediction_all.json")

# test RoboTools
cocoGt = COCO("datasets/RoboTools/test/scene_gt_coco_all_v2.json")
cocoDt = cocoGt.loadRes("datasets/RoboTools/test/weight_adapter_80epoch_samH_coco_instances_results_prediction_all.json")
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
# get IoU 0.95
# cocoEval.params.iouThrs = [0.95]
# Run the evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print(cocoEval.stats)
