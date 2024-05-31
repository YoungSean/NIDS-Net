import json
import os, sys
from matplotlib import pyplot as plt
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
from pathlib import Path


# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from src.utils.bbox_utils import CropResizePad, xywh_to_xyxy
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
from src.model.utils import Detections, convert_npz_to_json
from src.model.loss import Similarity
from src.utils.inout import save_json_bop23
import cv2
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

inv_rgb_transform = T.Compose(
    [
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
    ]
)


CURR_DIR = Path(__file__).parent.resolve()
PROJ_ROOT = CURR_DIR.parent.parent
TEMPLATE_ROOT = PROJ_ROOT / "HOT_Dataset/model_features"
DATA_ROOT = PROJ_ROOT / "HOT_Dataset"
SAVE_ROOT = DATA_ROOT / "CNOS_results"

OBJ_CLASS_COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


def make_clean_folder(folder_path):
    folder = Path(folder_path)
    if folder.exists():
        shutil.rmtree(str(folder))
    folder.mkdir(parents=True)


class BenchmarkRunner:
    def __init__(
        self,
        num_max_dets=1,
        conf_threshold=0.1,
        stability_score_thresh=0.5,
        debug=False,
    ) -> None:
        self._debug = debug
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_max_dets = num_max_dets
        self._conf_threshold = conf_threshold
        self._stability_score_thresh = stability_score_thresh
        self._metric = Similarity()

        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        self._proposal_processor = CropResizePad(processing_config.image_size)

        self._init_model()

        self._init_all_templates()

    def _init_model(self):
        logging.info("Initializing model")
        ts = time.time()
        with initialize(version_base=None, config_path="../../configs"):
            cfg = compose(config_name="run_inference.yaml")
        cfg_segmentor = cfg.model.segmentor_model
        if "fast_sam" in cfg_segmentor._target_:
            logging.info("Using FastSAM, ignore stability_score_thresh!")
        else:
            cfg.model.segmentor_model.stability_score_thresh = (
                self._stability_score_thresh
            )

        # start initializing model
        self._model = instantiate(cfg.model)
        self._model.descriptor_model.model = self._model.descriptor_model.model.to(
            self._device
        )
        self._model.descriptor_model.model.device = self._device
        # if there is predictor in the model, move it to device
        if hasattr(self._model.segmentor_model, "predictor"):
            self._model.segmentor_model.predictor.model = (
                self._model.segmentor_model.predictor.model.to(self._device)
            )
        else:
            self._model.segmentor_model.model.setup_model(
                device=self._device, verbose=True
            )
        logging.info(
            f"Moving models to {self._device} done! Time: {time.time() - ts:.2f}s"
        )

    def _init_templates(self):
        logging.info("Initializing template")

        self._ref_feats = {}

        for obj_id in self._object_ids:
            template_paths = (TEMPLATE_ROOT / obj_id).glob("*.png")
            boxes, templates = [], []
            for path in template_paths:
                image = Image.open(str(path))
                boxes.append(image.getbbox())

                image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
                templates.append(image)

            templates = torch.stack(templates).permute(0, 3, 1, 2)
            boxes = torch.tensor(np.array(boxes))

            templates = self._proposal_processor(images=templates, boxes=boxes).cuda()
            self._ref_feats[obj_id] = self._model.descriptor_model.compute_features(
                templates, token_name="x_norm_clstoken"
            )

    def _init_all_templates(self):
        logging.info("Initializing template")

        ts = time.time()

        self._ref_feats = {}

        self._object_ids = sorted(x.name for x in TEMPLATE_ROOT.iterdir() if x.is_dir())
        for obj_id in self._object_ids:
            template_paths = (TEMPLATE_ROOT / obj_id).glob("*.png")
            boxes, templates = [], []
            for path in template_paths:
                image = Image.open(str(path))
                boxes.append(image.getbbox())

                image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
                templates.append(image)

            templates = torch.stack(templates).permute(0, 3, 1, 2)
            boxes = torch.tensor(np.array(boxes))

            templates = self._proposal_processor(images=templates, boxes=boxes).cuda()
            self._ref_feats[obj_id] = self._model.descriptor_model.compute_features(
                templates, token_name="x_norm_clstoken"
            )

        logging.info(f"Templates initialized! Time: {time.time() - ts:.2f}s")

    def load_sequence(self, sequence_folder):
        self._sequence_folder = Path(sequence_folder).resolve()
        self._load_meta_data()

        self._save_root = SAVE_ROOT / self._subject_id / self._sequence_folder.name
        make_clean_folder(self._save_root)

    def _load_meta_data(self):
        meta_file = self._sequence_folder / "meta.json"
        data = json.loads(meta_file.read_text())
        self._subject_id = data["calibration"]["mano"]
        # self._object_ids = data["object_ids"]
        self._cam_serials = sorted(
            [x.name for x in self._sequence_folder.iterdir() if x.is_dir()]
        )

    def _get_frame_ids(self, camera_serial):
        cam_folder = self._sequence_folder / camera_serial
        frame_ids = sorted(
            [int(x.stem.split("_")[-1]) for x in cam_folder.glob("color_*.jpg")]
        )
        return frame_ids

    def _run_inference(self, serial, frame_id):
        # read rgb image
        rgb_path = self._sequence_folder / serial / f"color_{frame_id:06d}.jpg"
        rgb = Image.open(str(rgb_path)).convert("RGB")

        save_folder = self._save_root / serial / f"{frame_id:06d}"
        make_clean_folder(save_folder)

        # get mask detections
        seg_detections = self._model.segmentor_model.generate_masks(np.array(rgb))
        # run descriptor
        decriptors = self._model.descriptor_model.forward(
            np.array(rgb), Detections(seg_detections)
        )
        for idx, obj_id in enumerate(self._object_ids):
            ref_feats = self._ref_feats[obj_id]
            detections = Detections(seg_detections)

            # get scores per proposal
            scores = self._metric(decriptors[:, None, :], ref_feats[None, :, :])
            score_per_detection = torch.topk(scores, k=5, dim=-1)[0]
            score_per_detection = torch.mean(score_per_detection, dim=-1)

            # get top-k detections
            scores, index = torch.topk(
                score_per_detection, k=self._num_max_dets, dim=-1
            )
            detections.filter(index)
            # keep only detections with score > conf_threshold
            detections.filter(scores > self._conf_threshold)
            detections.add_attribute("scores", scores)
            # detections.add_attribute("object_ids", torch.zeros_like(scores))
            detections.add_attribute("object_ids", torch.ones_like(scores) * idx)

            detections.to_numpy()

            detections.save_to_file(
                scene_id=serial,
                frame_id=frame_id,
                runtime=0,
                file_path=str(save_folder / obj_id),
                dataset_name="custom",
                return_results=False,
            )
        detections = [
            convert_npz_to_json(
                idx=0,
                list_npz_paths=[str(save_folder / f"{obj_id}.npz")],
            )
            for obj_id in self._object_ids
        ]
        save_json_bop23(str(save_folder / "detections.json"), detections)
        if self._debug:
            visualize_bbox(
                rgb,
                detections,
                save_path=str(save_folder.parent / f"vis_{frame_id:06d}.png"),
            )

    def run(self):
        tqdm.write(
            f">>>>>>>>>> Processing {self._subject_id}/{self._sequence_folder.name} <<<<<<<<<<"
        )
        # self._init_templates()

        for serial in self._cam_serials:
            tqdm.write(f"- {serial}")
            frame_ids = self._get_frame_ids(serial)
            if len(frame_ids) == 0:
                continue
            for frame_id in tqdm(frame_ids, total=len(frame_ids), ncols=60):
                self._run_inference(serial, frame_id)


def visualize_bbox(rgb, detections, save_path="./tmp/tmp.png"):
    img = np.array(rgb).copy()
    overlay = np.zeros_like(img)
    alpha = 0.5

    scores = []
    for dets in detections:
        if len(dets) == 0:
            scores.append(0)
            continue
        scores.append(dets[0]["score"])

    top_k_indices = np.argsort(scores)[-4:][::-1]

    for idx, i in enumerate(top_k_indices):
        det = detections[i]
        if len(det) == 0:
            continue
        mask = rle_to_mask(det[0]["segmentation"])

        overlay[mask] = OBJ_CLASS_COLORS[idx + 1]
        bbox = np.array(det[0]["bbox"])
        bbox = xywh_to_xyxy(bbox)
        cv2.rectangle(
            overlay,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            OBJ_CLASS_COLORS[idx + 1],
            2,
        )

    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="Path to root directory of the template",
    )

    parser.add_argument(
        "--end_id",
        type=int,
        default=1,
        help="Path to root directory of the template",
    )

    args = parser.parse_args()

    # sequnece_folders = [
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_165502",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_165807",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_170105",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_170231",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_170532",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_170650",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_170959",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_171117",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_171314",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_1/20231025_171417",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231022_200657",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231022_201316",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231022_201449",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231022_201556",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231022_201942",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231022_202115",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231022_202617",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231022_203100",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231023_163929",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231023_164242",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231023_164741",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_2/20231023_170018",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_154531",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_154810",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_155008",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_161209",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_161306",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_161937",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_162028",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_162327",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_162409",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_162756",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_3/20231024_162842",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_4/20231026_162155",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_4/20231026_162248",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_4/20231026_163223",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_4/20231026_164131",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_4/20231026_164812",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_4/20231026_164909",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_4/20231026_164958",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_5/20231027_112303",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_5/20231027_113202",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_5/20231027_113535",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_6/20231025_110646",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_6/20231025_110808",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_6/20231025_111118",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_6/20231025_111357",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_6/20231025_112229",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_6/20231025_112332",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_6/20231025_112546",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_7/20231022_190534",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_7/20231022_192832",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_7/20231022_193506",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_7/20231022_193630",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_7/20231022_193809",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_7/20231023_162803",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_7/20231023_163653",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_8/20231024_180111",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_8/20231024_180651",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_8/20231024_180733",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_8/20231024_181413",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_9/20231027_123403",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_9/20231027_123725",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_9/20231027_123814",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_9/20231027_124057",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_9/20231027_124926",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_9/20231027_125019",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_9/20231027_125315",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_9/20231027_125407",
    #     "/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/sequences/subject_9/20231027_125457",
    # ]

    sequnece_folders = [
        DATA_ROOT / "sequences/subject_1/20231025_165502",
        DATA_ROOT / "sequences/subject_1/20231025_165807",
        DATA_ROOT / "sequences/subject_1/20231025_170105",
        DATA_ROOT / "sequences/subject_1/20231025_170231",
        DATA_ROOT / "sequences/subject_1/20231025_170532",
        DATA_ROOT / "sequences/subject_1/20231025_170650",
        DATA_ROOT / "sequences/subject_1/20231025_170959",
        DATA_ROOT / "sequences/subject_1/20231025_171117",
        DATA_ROOT / "sequences/subject_1/20231025_171314",
        DATA_ROOT / "sequences/subject_1/20231025_171417",
        DATA_ROOT / "sequences/subject_2/20231022_200657",
        DATA_ROOT / "sequences/subject_2/20231022_201316",
        DATA_ROOT / "sequences/subject_2/20231022_201449",
        DATA_ROOT / "sequences/subject_2/20231022_201556",
        DATA_ROOT / "sequences/subject_2/20231022_201942",
        DATA_ROOT / "sequences/subject_2/20231022_202115",
        DATA_ROOT / "sequences/subject_2/20231022_202617",
        DATA_ROOT / "sequences/subject_2/20231022_203100",
        DATA_ROOT / "sequences/subject_2/20231023_163929",
        DATA_ROOT / "sequences/subject_2/20231023_164242",
        DATA_ROOT / "sequences/subject_2/20231023_164741",
        DATA_ROOT / "sequences/subject_2/20231023_170018",
        DATA_ROOT / "sequences/subject_3/20231024_154531",
        DATA_ROOT / "sequences/subject_3/20231024_154810",
        DATA_ROOT / "sequences/subject_3/20231024_155008",
        DATA_ROOT / "sequences/subject_3/20231024_161209",
        DATA_ROOT / "sequences/subject_3/20231024_161306",
        DATA_ROOT / "sequences/subject_3/20231024_161937",
        DATA_ROOT / "sequences/subject_3/20231024_162028",
        DATA_ROOT / "sequences/subject_3/20231024_162327",
        DATA_ROOT / "sequences/subject_3/20231024_162409",
        DATA_ROOT / "sequences/subject_3/20231024_162756",
        DATA_ROOT / "sequences/subject_3/20231024_162842",
        DATA_ROOT / "sequences/subject_4/20231026_162155",
        DATA_ROOT / "sequences/subject_4/20231026_162248",
        DATA_ROOT / "sequences/subject_4/20231026_163223",
        DATA_ROOT / "sequences/subject_4/20231026_164131",
        DATA_ROOT / "sequences/subject_4/20231026_164812",
        DATA_ROOT / "sequences/subject_4/20231026_164909",
        DATA_ROOT / "sequences/subject_4/20231026_164958",
        DATA_ROOT / "sequences/subject_5/20231027_112303",
        DATA_ROOT / "sequences/subject_5/20231027_113202",
        DATA_ROOT / "sequences/subject_5/20231027_113535",
        DATA_ROOT / "sequences/subject_6/20231025_110646",
        DATA_ROOT / "sequences/subject_6/20231025_110808",
        DATA_ROOT / "sequences/subject_6/20231025_111118",
        DATA_ROOT / "sequences/subject_6/20231025_111357",
        DATA_ROOT / "sequences/subject_6/20231025_112229",
        DATA_ROOT / "sequences/subject_6/20231025_112332",
        DATA_ROOT / "sequences/subject_6/20231025_112546",
        DATA_ROOT / "sequences/subject_7/20231022_190534",
        DATA_ROOT / "sequences/subject_7/20231022_192832",
        DATA_ROOT / "sequences/subject_7/20231022_193506",
        DATA_ROOT / "sequences/subject_7/20231022_193630",
        DATA_ROOT / "sequences/subject_7/20231022_193809",
        DATA_ROOT / "sequences/subject_7/20231023_162803",
        DATA_ROOT / "sequences/subject_7/20231023_163653",
        DATA_ROOT / "sequences/subject_8/20231024_180111",
        DATA_ROOT / "sequences/subject_8/20231024_180651",
        DATA_ROOT / "sequences/subject_8/20231024_180733",
        DATA_ROOT / "sequences/subject_8/20231024_181413",
        DATA_ROOT / "sequences/subject_9/20231027_123403",
        DATA_ROOT / "sequences/subject_9/20231027_123725",
        DATA_ROOT / "sequences/subject_9/20231027_123814",
        DATA_ROOT / "sequences/subject_9/20231027_124057",
        DATA_ROOT / "sequences/subject_9/20231027_124926",
        DATA_ROOT / "sequences/subject_9/20231027_125019",
        DATA_ROOT / "sequences/subject_9/20231027_125315",
        DATA_ROOT / "sequences/subject_9/20231027_125407",
        DATA_ROOT / "sequences/subject_9/20231027_125457",
    ]
    bmark_runner = BenchmarkRunner(num_max_dets=1, conf_threshold=0.0, debug=True)

    for sequence_folder in sequnece_folders[args.start_id : args.end_id]:
        bmark_runner.load_sequence(sequence_folder)
        bmark_runner.run()
