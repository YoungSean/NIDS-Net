#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
parse pascal_voc XML file to COCO json
"""
import torch
import glob
import os
import random
import re
import shutil
import json
import xml.etree.ElementTree as ET
# from sklearn.model_selection import train_test_split
from data_utils import minify

CATEGORIES = ["000_aveda_shampoo", "001_binder_clips_median", "002_binder_clips_small", "003_bombik_bucket",
              "004_bonne_maman_blueberry", "005_bonne_maman_raspberry", "006_bonne_maman_strawberry",
              "007_costa_caramel", "008_essential_oil_bergamot", "009_garlic_toast_spread", "010_handcream_avocado",
              "011_hb_calcium", "012_hb_grapeseed", "013_hb_marine_collagen", "014_hellmanns_mayonnaise",
              "015_illy_blend", "016_japanese_finger_cookies", "017_john_west_canned_tuna", "018_kerastase_shampoo",
              "019_kiehls_facial_cream", "020_kiihne_balsamic", "021_kiihne_honey_mustard", "022_lindor_matcha",
              "023_lindor_salted_caramel", "024_lush_mask", "025_pasta_sauce_black_pepper", "026_pasta_sauce_tomato",
              "027_pepsi", "028_portable_yogurt_machine", "029_selfile_stick", "030_sour_lemon_drops",
              "031_sticky_notes", "032_stridex_green", "033_thermos_flask_cream", "034_thermos_flask_muji",
              "035_thermos_flask_sliver", "036_tragata_olive_oil", "037_tulip_luncheon_meat", "038_unicharm_cotton_pad",
              "039_vinda_tissue", "040_wrigley_doublemint_gum", "041_baseball_cap_black", "042_baseball_cap_pink",
              "043_bfe_facial_mask", "044_corgi_doll", "045_dinosaur_doll", "046_geo_mocha", "047_geo_roast_charcoal",
              "048_instant_noodle_black", "049_instant_noodle_red", "050_nabati_cheese_wafer", "051_truffettes",
              "052_acnes_cream", "053_aveda_conditioner", "054_banana_milk_drink", "055_candle_beast",
              "056_china_persimmon", "057_danisa_butter_cookies", "058_effaclar_duo", "059_evelom_cleanser",
              "060_glasses_box_blone", "061_handcream_iris", "062_handcream_lavender", "063_handcream_rosewater",
              "064_handcream_summer_hill", "065_hr_serum", "066_japanese_chocolate", "067_kerastase_hair_treatment",
              "068_kiehls_serum", "069_korean_beef_marinade", "070_korean_doenjang", "071_korean_gochujang",
              "072_korean_ssamjang", "073_loccitane_soap", "074_marvis_toothpaste_purple", "075_mouse_thinkpad",
              "076_oatly_chocolate", "077_oatly_original", "078_ousa_grated_cheese", "079_polaroid_film",
              "080_skinceuticals_be", "081_skinceuticals_cf", "082_skinceuticals_phyto", "083_stapler_black",
              "084_stapler_blue", "085_sunscreen_blue", "086_tempo_pocket_tissue", "087_thermos_flask_purple",
              "088_uha_matcha", "089_urban_decay_spray", "090_vitaboost_multivitamin", "091_watercolor_penbox",
              "092_youthlt_bilberry_complex", "093_daiso_mod_remover", "094_kaneyo_kitchen_bleach",
              "095_lays_chip_bag_blue", "096_lays_chip_bag_green", "097_lays_chip_tube_auburn",
              "098_lays_chip_tube_green", "099_mug_blue"]


def readXML(xml_file):
    data = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    info = {}
    info['dataname'] = []
    info['filename'] = []
    info['width'] = 1024
    info['height'] = 768
    info['depth'] = 1

    for eles in root:
        if eles.tag == 'folder':
            info['dataname'] = eles.text
        elif eles.tag == 'filename':
            info['filename'] = eles.text
        elif eles.tag == 'size':
            for elem in eles:
                if elem.tag == 'width':
                    info['width'] = elem.text
                elif elem.tag == 'height':
                    info['height'] = elem.text
                elif elem.tag == 'depth':
                    info['depth'] = elem.text
                else:
                    continue
        elif eles.tag == 'object':
            anno = dict()
            for elem in eles:
                if elem.tag == 'name':
                    anno['name'] = elem.text
                elif elem.tag == 'bndbox':
                    for subelem in elem:
                        if subelem.tag == 'xmin':
                            anno['xmin'] = float(subelem.text)
                        elif subelem.tag == 'xmax':
                            anno['xmax'] = float(subelem.text)
                        elif subelem.tag == 'ymin':
                            anno['ymin'] = float(subelem.text)
                        elif subelem.tag == 'ymax':
                            anno['ymax'] = float(subelem.text)
                        else:
                            continue
            data.append(anno)

    return info, data


def getCOCOjson(root_path, save_path, factor=1.0, flag=None):
    # parse all .xml files to a .json file
    dataset = dict()
    dataset['info'] = {}
    dataset['licenses'] = []
    dataset['images'] = []
    dataset['annotations'] = []
    dataset['categories'] = []

    dataset['info']['description'] = 'RealWorld Dataset'
    dataset['info']['url'] = ''
    dataset['info']['version'] = '1.0'
    dataset['info']['year'] = 2023
    dataset['info']['contributor'] = ''
    dataset['info']['date_created'] = ''

    licenses = {}
    licenses['url'] = ''
    licenses['id'] = 1
    licenses['name'] = ''
    dataset['licenses'].append(licenses)

    all_anno_count = 0
    img_list = sorted([p for p in glob.glob(os.path.join(root_path, 'images', '*'))
                       if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])
    for i_img, img_file in enumerate(img_list):
        file_name = os.path.basename(img_file)
        if flag == 'test':
            anno_path = os.path.join(root_path, 'annotations',
                                     file_name.split('.')[0] + '.xml')  # .xml files for RealScenes
        else:
            anno_path = os.path.join(root_path, 'annotations',
                                     file_name.split('_')[0] + '.xml')  # .xml files for cut-paste-learn

        info, objects = readXML(anno_path)

        # images
        images = {}
        images['license'] = 1
        images['file_name'] = file_name
        images['coco_url'] = ''
        images['height'] = int(float(info['height']) * factor)
        images['width'] = int(float(info['width']) * factor)
        images['date_captured'] = ''
        images['flickr_url'] = ''
        images['id'] = int(i_img)

        dataset['images'].append(images)

        # annotations
        for object in objects:
            if int(object['name'].split('_')[0]) > len(CATEGORIES) - 1:
                continue
            # bbox: [xmin,ymin,w,h]
            bbox = []
            bbox.append(object['xmin'])
            bbox.append(object['ymin'])
            bbox.append(object['xmax'] - object['xmin'])
            bbox.append(object['ymax'] - object['ymin'])

            if factor != 1:
                bbox = [x * factor for x in bbox]

            # when segmentation annotation not given, use [[x1,y1,x2,y1,x2,y2,x1,y2]] instead
            segmentation = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                             bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3]]]

            annotations = {}
            annotations['segmentation'] = segmentation
            annotations['area'] = bbox[-1] * bbox[-2]
            annotations['iscrowd'] = 0
            annotations['image_id'] = int(i_img)
            annotations['bbox'] = bbox
            annotations['category_id'] = int(object['name'].split('_')[0])
            annotations['id'] = all_anno_count

            dataset['annotations'].append(annotations)
            all_anno_count += 1

    # categories
    for i_cat, cat in enumerate(CATEGORIES):
        categories = {}
        categories['supercategory'] = cat
        categories['id'] = i_cat
        categories['name'] = cat
        dataset['categories'].append(categories)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
    print('ok')


if __name__ == '__main__':

    # root_path = "../syndata-generation/syndata_1"

    # image_paths = os.listdir(os.path.join(root_path, 'images'))
    # # train:val = 0.75:0.25
    # image_train, image_val = train_test_split(image_paths, test_size=0.25, random_state=77)

    # # copy image to train set --> create train_json
    # if not os.path.exists(os.path.join(root_path, 'train')):
    #     os.makedirs(os.path.join(root_path, 'train', 'images'))
    #     os.makedirs(os.path.join(root_path, 'train/annotations'))

    # for name in image_train:
    #     shutil.copy(os.path.join(root_path, 'images', name),
    #                 os.path.join(root_path, 'train/images', name))
    #     shutil.copy(os.path.join(root_path, 'annotations', name.split('_')[0] + '.xml'),
    #                 os.path.join(root_path, 'train/annotations', name.split('_')[0] + '.xml'))

    # getCOCOjson(os.path.join(root_path, 'train'), os.path.join(root_path, 'instances_train.json'))

    # # copy image to val set --> create val_json
    # if not os.path.exists(os.path.join(root_path, 'val')):
    #     os.makedirs(os.path.join(root_path, 'val/images'))
    #     os.makedirs(os.path.join(root_path, 'val/annotations'))

    # for name in image_val:
    #     shutil.copy(os.path.join(root_path, 'images', name),
    #                 os.path.join(root_path, 'val/images', name))
    #     shutil.copy(os.path.join(root_path, 'annotations', name.split('_')[0] + '.xml'),
    #                 os.path.join(root_path, 'val/annotations', name.split('_')[0] + '.xml'))

    # getCOCOjson(os.path.join(root_path, 'val'), os.path.join(root_path, 'instances_val.json'))

    # test data
    
    level = 'easy'  # 'all', 'hard', 'easy'
    factor = 4
    root_path = '../database/Scenes' #"../database/Scenes"
    test_path = "../database/Data/test_" + str(factor) + '_' + str(level)  # "../database/Data/test_" + str(factor) + '_' + str(level)
    if not os.path.exists(os.path.join(test_path, 'images')):
        os.makedirs(os.path.join(test_path, 'images'))
    if not os.path.exists(os.path.join(test_path, 'annotations')):
        os.makedirs(os.path.join(test_path, 'annotations'))
    
    if level == 'all':
        image_paths = sorted([p for p in glob.glob(os.path.join(root_path, '*/*/*'))
                              if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])
        anno_paths = sorted([p for p in glob.glob(os.path.join(root_path, '*/*/*'))
                             if re.search('/*\.xml', str(p))])
    else:
        image_paths = sorted([p for p in glob.glob(os.path.join(root_path, level, '*/*'))
                              if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])
        anno_paths = sorted([p for p in glob.glob(os.path.join(root_path, level, '*/*'))
                             if re.search('/*\.xml', str(p))])
    
    for i, file_path in enumerate(zip(image_paths, anno_paths)):
        file_name = 'test_' + '%03d' % i
        img_extend = os.path.splitext(file_path[0])[-1]  # extend for image file
        anno_extend = os.path.splitext(file_path[1])[-1]  # extend for image file
    
        shutil.copyfile(file_path[0], os.path.join(test_path, 'images', file_name + img_extend))
        shutil.copyfile(file_path[1], os.path.join(test_path, 'annotations', file_name + anno_extend))
    
    getCOCOjson(os.path.join(test_path),
                os.path.join(test_path, "instances_test_" + str(factor) + '_' + str(level) + ".json"),
                factor=1/factor, flag='test')
    # height = 6144
    # width = 8192
    # minify(os.path.join(test_path, 'images'), os.path.join(test_path, 'test'),
    #        factors=[], resolutions=[[int(height / factor), int(width / factor)]], extend='jpg')


