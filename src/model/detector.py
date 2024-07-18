import torch
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
import os
import os.path as osp
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from src.utils.inout import save_json, load_json, save_json_bop23
from src.model.utils import BatchedData, Detections, convert_npz_to_json
from hydra.utils import instantiate
import time
import glob
from functools import partial
import multiprocessing
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import json
from torch import nn
import torch.nn.functional as F
import sys

class MaskedPatch_MatrixSimilarity(nn.Module):
    """Get from SAM6D."""
    def __init__(self, metric="cosine", chunk_size=64):
        super(MaskedPatch_MatrixSimilarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def compute_straight(self, query, reference):
        (N_query, N_patch, N_features) = query.shape
        sim_matrix = torch.matmul(query, reference.permute(0, 2, 1))  # N_query x N_query_mask x N_refer_mask

        # N2_ref score max
        max_ref_patch_score = torch.max(sim_matrix, dim=-1).values
        # N1_query score average
        factor = torch.count_nonzero(query.sum(dim=-1), dim=-1) + 1e-6
        scores = torch.sum(max_ref_patch_score, dim=-1) / factor  # N_query x N_objects x N_templates

        return scores.clamp(min=0.0, max=1.0)

    def compute_visible_ratio(self, query, reference, thred=0.5):

        sim_matrix = torch.matmul(query, reference.permute(0, 2, 1))  # N_query x N_query_mask x N_refer_mask
        sim_matrix = sim_matrix.max(1)[0]  # N_query x N_refer_mask
        valid_patches = torch.count_nonzero(sim_matrix, dim=(1,)) + 1e-6

        # fliter correspendence by thred
        flitered_matrix = sim_matrix * (sim_matrix > thred)
        sim_patches = torch.count_nonzero(flitered_matrix, dim=(1,))

        visible_ratio = sim_patches / valid_patches

        return visible_ratio

    def compute_similarity(self, query, reference):
        # all template computation
        N_query = query.shape[0]
        N_objects, N_templates = reference.shape[0], reference.shape[1]
        references = reference.unsqueeze(0).repeat(N_query, 1, 1, 1, 1)
        queries = query.unsqueeze(1).repeat(1, N_templates, 1, 1)

        similarity = BatchedData(batch_size=None)
        for idx_obj in range(N_objects):
            sim_matrix = torch.matmul(queries, references[:, idx_obj].permute(0, 1, 3,
                                                                              2))  # N_query x N_templates x N_query_mask x N_refer_mask
            similarity.append(sim_matrix)
        similarity.stack()
        similarity = similarity.data
        similarity = similarity.permute(1, 0, 2, 3, 4)  # N_query x N_objects x N_templates x N1_query x N2_ref

        # N2_ref score max
        max_ref_patch_score = torch.max(similarity, dim=-1).values
        # N1_query score average
        factor = torch.count_nonzero(query.sum(dim=-1), dim=-1)[:, None, None]
        scores = torch.sum(max_ref_patch_score, dim=-1) / factor  # N_query x N_objects x N_templates

        return scores.clamp(min=0.0, max=1.0)

    def forward_by_chunk(self, query, reference):
        # divide by N_query
        batch_query = BatchedData(batch_size=self.chunk_size, data=query)
        del query
        scores = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_query)):
            score = self.compute_similarity(batch_query[idx_batch], reference)
            scores.cat(score)
        return scores.data

    def forward(self, qurey, reference):
        if qurey.shape[0] > self.chunk_size:
            scores = self.forward_by_chunk(qurey, reference)
        else:
            scores = self.compute_similarity(qurey, reference)
        return scores

class ModifiedClipAdapter(nn.Module):
    """
    Modified version of the CLIP adapter for better performance.
    Add Dropout layer.
    """
    def __init__(self, c_in, reduction=4, ratio=0.6):
        super(ModifiedClipAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ratio = ratio

    def forward(self, inputs):
        inputs = F.normalize(inputs, dim=-1, p=2)
        x = self.fc(inputs)
        x = self.ratio * x + (1 - self.ratio) * inputs
        return x

class WeightAdapter(nn.Module):
    """
    Predict weights for each feature vector.
    """
    def __init__(self, c_in, reduction=4, scalar=10):
        """

        @param c_in: The channel size of the input feature vector
        @param reduction: the reduction factor for the hidden layer
        @param scalar: A scalar to scale the input feature vector
        """
        super(WeightAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        # self.ratio = ratio
        self.scalar = scalar

    def forward(self, inputs):
        inputs = self.scalar * inputs
        x = self.fc(inputs)
        x = x.sigmoid()
        x = x * inputs

        return x


class NIDSNET(pl.LightningModule):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.segmentor_model = None #segmentor_model
        self.descriptor_model = descriptor_model

        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.log_interval = log_interval
        self.log_dir = log_dir

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        logging.info(f"Init CNOS done!")
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor(vit_model="vit_h")
        logging.info("Initialize GDINO and SAM done!")
        self.use_adapter = False
        if self.use_adapter:
            self.adapter_type = 'weight'
            if self.adapter_type == 'clip':
                weight_name = f"bop_obj_shuffle_0507_clip_temp_0.05_epoch_500_lr_0.0001_bs_32_weights.pth"
                model_path = os.path.join("./adapter_weights/bop23", weight_name)
                self.adapter = ModifiedClipAdapter(1024, reduction=4, ratio=0.6).to('cuda')
            else:
                weight_name = f"bop_obj_shuffle_weight_0430_temp_0.05_epoch_500_lr_0.001_bs_32_weights.pth"
                model_path = os.path.join("./adapter_weights/bop23", weight_name)
                self.adapter = WeightAdapter(1024, reduction=4).to('cuda')
            self.adapter.load_state_dict(torch.load(model_path))
            self.adapter.eval()

    def set_reference_objects(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data = {"descriptors": BatchedData(None), "cls_descriptors": BatchedData(None), "appe_descriptors": BatchedData(None)}
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")
        # cls_descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors_cls.pth")  # for cls token
        if self.onboarding_config.rendering_type == "pbr":
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(self.device) # if you want to use the original template embedding, uncomment here.
            # uncomment the following lines if you want to use the adapter and the adapted template embedding.
            # adapter_descriptors_path = osp.join(self.ref_dataset.template_dir,
            #                                     f'{self.adapter_type}_obj_shuffle2_0501_bs32_epoch_500_adapter_descriptors_pbr.json')
            # with open(os.path.join(adapter_descriptors_path), 'r') as f:
            #     feat_dict = json.load(f)
            #
            # object_features = torch.Tensor(feat_dict['features']).cuda()
            # self.ref_data["descriptors"] = object_features.view(-1, 42, 1024)
            # print("using adapted object features")
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                FFA_feats, _, _ = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken", masks=ref_masks
                )
                self.ref_data["descriptors"].append(FFA_feats)

            self.ref_data["descriptors"].stack()  # N_objects x descriptor_size
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["descriptors"], descriptors_path)

        # # Loading cls token
        # if self.onboarding_config.rendering_type == "pbr":
        #     cls_descriptors_path = cls_descriptors_path.replace(".pth", "_pbr.pth")
        # if (
        #     os.path.exists(cls_descriptors_path)
        #     and not self.onboarding_config.reset_descriptors
        # ):
        #     self.ref_data["cls_descriptors"] = torch.load(cls_descriptors_path).to(self.device)
        #
        #     # adapter_descriptors_path = osp.join(self.ref_dataset.template_dir,
        #     #                                         f'{self.adapter_type}_cls_obj_shuffle_0510_bs32_epoch_500_adapter_descriptors_pbr.json')
        #     # with open(os.path.join(adapter_descriptors_path), 'r') as f:
        #     #     feat_dict = json.load(f)
        #     #
        #     # object_features = torch.Tensor(feat_dict['features']).cuda()
        #     # self.ref_data["cls_descriptors"] = object_features.view(-1, 42, 1024)
        #     # print("using adapted lmo object features")
        # else:
        #     for idx in tqdm(
        #         range(len(self.ref_dataset)),
        #         desc="Computing class token descriptors ...",
        #     ):
        #         ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
        #         ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
        #         _, _, cls_feats = self.descriptor_model.compute_features(
        #             ref_imgs, token_name="x_norm_clstoken", masks=ref_masks
        #         )
        #         self.ref_data["cls_descriptors"].append(cls_feats)
        #
        #     self.ref_data["cls_descriptors"].stack()
        #     self.ref_data["cls_descriptors"] = self.ref_data["cls_descriptors"].data
        #
        #     # save the precomputed features for future use
        #     torch.save(self.ref_data["cls_descriptors"], cls_descriptors_path)

        # Loading appearance descriptors
        appe_descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors_appe.pth")
        if self.onboarding_config.rendering_type == "pbr":
            appe_descriptors_path = appe_descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(appe_descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["appe_descriptors"] = torch.load(appe_descriptors_path).to(self.device)
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing appearance descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                ref_feats = self.descriptor_model.compute_masked_patch_feature(
                    ref_imgs, ref_masks
                )
                self.ref_data["appe_descriptors"].append(ref_feats)

            self.ref_data["appe_descriptors"].stack()
            self.ref_data["appe_descriptors"] = self.ref_data["appe_descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["appe_descriptors"], appe_descriptors_path)
        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}"
        )


    def move_to_device(self):
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device

        # if there is predictor in the model, move it to device
        # if hasattr(self.segmentor_model, "predictor"):
        #     self.segmentor_model.predictor.model = (
        #         self.segmentor_model.predictor.model.to(self.device)
        #     )
        # else:
        #     self.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")

    def best_template_pose(self, scores, pred_idx_objects):
        _, best_template_idxes = torch.max(scores, dim=-1)
        N_query, N_object = best_template_idxes.shape[0], best_template_idxes.shape[1]
        pred_idx_objects = pred_idx_objects[:, None].repeat(1, N_object)

        assert N_query == pred_idx_objects.shape[0], "Prediction num != Query num"

        best_template_idx = torch.gather(best_template_idxes, dim=1, index=pred_idx_objects)[:, 0]

        return best_template_idx
    def find_matched_proposals(self, proposal_decriptors):
        # compute matching scores for each proposals
        scores = self.matching_config.metric(
            proposal_decriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates
        # scores = self.matching_config.metric(
        #     proposal_decriptors, self.ref_data["cls_descriptors"]
        # )  # N_proposals x N_objects x N_templates
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_query
        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        pred_scores = score_per_proposal[idx_selected_proposals]

        # compute the best view of template
        flitered_scores = scores[idx_selected_proposals, ...]
        best_template = self.best_template_pose(flitered_scores, pred_idx_objects)

        return idx_selected_proposals, pred_idx_objects, pred_scores, best_template

    def compute_appearance_score(self, best_pose, pred_objects_idx, qurey_appe_descriptors):
        """
        Based on the best template, calculate appearance similarity indicated by appearance score
        """
        con_idx = torch.concatenate((pred_objects_idx[None, :], best_pose[None, :]), dim=0)
        ref_appe_descriptors = self.ref_data["appe_descriptors"][con_idx[0, ...], con_idx[1, ...], ...] # N_query x N_patch x N_feature

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(qurey_appe_descriptors, ref_appe_descriptors)

        return appe_scores, ref_appe_descriptors

    def test_step(self, batch, idx):
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}",
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0])
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # run propoals
        proposal_stage_start_time = time.time()

        image_pil = Image.fromarray(image_np).convert("RGB")
        bboxes, phrases, gdino_conf = self.gdino.predict(image_pil, "objects")
        w, h = image_pil.size  # Get image width and height
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
        image_pil_bboxes, masks = self.SAM.predict(image_pil, image_pil_bboxes)
        proposals = dict()
        proposals["masks"] = masks.squeeze(1).to(torch.float32)  # to N x H x W, torch.float32 type as the output of fastSAM
        proposals["boxes"] = image_pil_bboxes
        # else:
        # proposals = self.segmentor_model.generate_masks(image_np)

        # init detections with masks and boxes
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )
        # compute descriptors
        query_FFA_decriptors, query_appe_descriptors, query_cls_descriptors = self.descriptor_model(image_np, detections)
        query_decriptors = query_FFA_decriptors
        if self.use_adapter:
            with torch.no_grad():
                query_decriptors = self.adapter(query_decriptors)
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
            best_template,
        ) = self.find_matched_proposals(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor = self.compute_appearance_score(best_template, pred_idx_objects,
                                                                        query_appe_descriptors)

        # final score
        final_score = (pred_scores + appe_scores) / 2.0
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def test_epoch_end(self, outputs):
        if self.global_rank == 0:  # only rank 0 process
            # can use self.all_gather to gather results from all processes
            # but it is simpler just load the results from files so no file is missing
            result_paths = sorted(
                glob.glob(
                    osp.join(
                        self.log_dir,
                        f"predictions/{self.dataset_name}/{self.name_prediction_file}/*.npz",
                    )
                )
            )
            result_paths = sorted(
                [path for path in result_paths if "runtime" not in path]
            )
            num_workers = 10
            logging.info(f"Converting npz to json requires {num_workers} workers ...")
            pool = multiprocessing.Pool(processes=num_workers)
            convert_npz_to_json_with_idx = partial(
                convert_npz_to_json,
                list_npz_paths=result_paths,
            )
            detections = list(
                tqdm(
                    pool.imap_unordered(
                        convert_npz_to_json_with_idx, range(len(result_paths))
                    ),
                    total=len(result_paths),
                    desc="Converting npz to json",
                )
            )
            formatted_detections = []
            for detection in tqdm(detections, desc="Loading results ..."):
                formatted_detections.extend(detection)

            detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
            save_json_bop23(detections_path, formatted_detections)
            logging.info(f"Saved predictions to {detections_path}")
