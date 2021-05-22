import os
import cv2
import json
import numpy as np

dir_path = '../imdb/m4c_textvqa'
imdb_file = 'imdb_val_ocr_en.npy'
textfont, textsize, boxthick = 2,1,3
imdb = np.load(os.path.join(dir_path, imdb_file),allow_pickle=True)
##sample 0 ['creation_time', 'version', 'dataset_type', 'has_answer']
for sample_ii in range(1, len(imdb)):
    """
    keys
    ['question', 'image_id', 'image_classes', 'flickr_original_url', 'flickr_300k_url', \
        'image_width', 'image_height', 'answers', 'question_tokens', 'question_id', 'set_name', \
        'image_name', 'image_path', 'feature_path', 'valid_answers', 'ocr_tokens', 'ocr_info', \
        'ocr_normalized_boxes', 'obj_normalized_boxes']
    """
    sample = imdb[sample_ii]
    qid = sample['question_id']
    if not os.path.isfile(os.path.join('ocrimagesII','%s.jpg'%qid)):
        continue
    ## FB OCR format
    img_path = os.path.join('ocrimagesII','%s.jpg'%qid)
    img = cv2.imread(img_path)
    # print(sample['ocr_tokens'])
    # print(sample['ocr_info'])
    # print(sample['ocr_normalized_boxes'])
    img_path = os.path.join('ocrimagesII','%s.jpg'%qid)
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    img_info = {}
    for ii in range(len(sample['ocr_tokens'])):
        token, bbox = sample['ocr_tokens'][ii], sample['ocr_normalized_boxes'][ii]
        bbox = [int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)]
        img_info[token] = [ii]
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,204,204), boxthick)
        cv2.putText(img,'%d'%ii,(bbox[0],bbox[1]-5),cv2.FONT_HERSHEY_COMPLEX,textsize,(0,144,144),textfont)
    cv2.imwrite(os.path.join('fb','%s.jpg'%qid),img)
    with open(os.path.join('fb','%s_info.json'%qid), 'w') as fp:
        json.dump(img_info, fp)

    ## MS OCR format
    ms_ocr = json.load(open(os.path.join('ocrimagesII','%s.jpg.ocr.json'%qid),'r'))
    lines = ms_ocr['analyzeResult']['readResults'][0]['lines']
    img_path = os.path.join('ocrimagesII','%s.jpg'%qid)
    img = cv2.imread(img_path)
    img_info = {}
    for line_ii in range(len(lines)):
        line = lines[line_ii]
        # print(line['boundingBox'])
        # print(line['text'])
        # print(line['words'])
        bbox = line['boundingBox']
        # img_info[line['text']] = line['words']
        img_info[line['text']] = [line_ii] + [x['text'] for x in line['words']]
        # # imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        # cv2.rectangle(img, (bbox[ii,0], bbox[ii,1]), (bbox[ii,2], bbox[ii,3]), (255,0,0), 8)
        # cv2.rectangle(img, (target_bbox[ii,0], target_bbox[ii,1]), (target_bbox[ii,2], target_bbox[ii,3]), (0,255,255), 8)
        poly = np.array([[bbox[:2],bbox[2:4],bbox[4:6],bbox[6:8]]], np.int32)
        cv2.polylines(img, [poly], True, (0,204,204), thickness=boxthick)
        cv2.putText(img,'%d'%line_ii,(bbox[0],bbox[1]-5),cv2.FONT_HERSHEY_COMPLEX,textsize,(0,144,144),textfont)
    cv2.imwrite(os.path.join('ms','%s.jpg'%qid),img)
    with open(os.path.join('ms','%s_info.json'%qid), 'w') as fp:
        json.dump(img_info, fp)