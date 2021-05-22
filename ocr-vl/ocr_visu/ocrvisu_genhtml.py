import os
import json
import numpy as np



html = """<!DOCTYPE html PUBLIC '-//W3C//DTD HTML 4.01 Transitional//EN'>
<!-- saved from url=(0041)https://people.eecs.berkeley.edu/~barron/ -->
<html><head><meta http-equiv='Content-Type' content='text/html; charset=windows-1252'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <meta name='generator' content='HTML Tidy for Linux/x86 (vers 11 February 2007), see www.w3.org'>
    <style type='text/css'>
        /* Color scheme stolen from Sergey Karayev */
        a {
        color: #1772d0;
        text-decoration:none;
        }
        a:focus, a:hover {
        color: #f09228;
        text-decoration:none;
        }
        body,td,th,tr,p,a {
        font-family: 'Lato', Verdana, Helvetica, sans-serif;
        font-size: 18px
        }
        strong {
        font-family: 'Lato', Verdana, Helvetica, sans-serif;
        font-size: 18px;
        }
        heading {
        font-family: 'Lato', Verdana, Helvetica, sans-serif;
        font-size: 26px;
        }
        papertitle {
        font-family: 'Lato', Verdana, Helvetica, sans-serif;
        font-size: 18px;
        font-weight: 700
        }
        name {
        font-family: 'Lato', Verdana, Helvetica, sans-serif;
        font-size: 32px;
        }
        .one
        {
        width: 480px;
        height: 480px;
        position: relative;
        }
        .two
        {
        width: 160px;
        height: 160px;
        position: absolute;
        transition: opacity .2s ease-in-out;
        -moz-transition: opacity .2s ease-in-out;
        -webkit-transition: opacity .2s ease-in-out;
        }
        .fade {
         transition: opacity .2s ease-in-out;
         -moz-transition: opacity .2s ease-in-out;
         -webkit-transition: opacity .2s ease-in-out;
        }
        span.highlight {
                background-color: #ffffd0;
        }
    </style>
    <title>-</title>
    
    <link href='./index_files/css' rel='stylesheet' type='text/css'>
    </head>
    <body>

        <table width='100%' align='center' border='0' cellspacing='0' cellpadding='20'>
            <tbody><tr>
                <td width='100%' valign='middle'>
                    <heading>---</heading>
                </td>
            </tr>
            </tbody></table>

        <table width='100%' align='center' border='0' cellspacing='0' cellpadding='20'>
            <tbody><tr>
                <td width='50%'>
                    <br /> <heading> --- </heading>
                </td>
            </tr>"""

dir_path = '../imdb/m4c_textvqa'
imdb_file = 'imdb_val_ocr_en.npy'
prediction_path = '../../tmp_results'
imdb = np.load(os.path.join(dir_path, imdb_file),allow_pickle=True)
##sample 0 ['creation_time', 'version', 'dataset_type', 'has_answer']
for sample_ii in range(1, len(imdb)):
    sample = imdb[sample_ii]
    qid = sample['question_id']
    if not os.path.isfile(os.path.join('ms','%s.jpg'%qid)):
        continue
    with open(os.path.join(prediction_path,'%s.json'%sample['question_id']), 'r') as f:
        prediction = json.load(f)
    ms_imgpath = os.path.join('ms','%s.jpg'%qid)
    fb_imgpath = os.path.join('fb','%s.jpg'%qid)
    ms_infopath = os.path.join('ms','%s_info.json'%qid)
    fb_infopath = os.path.join('fb','%s_info.json'%qid)
    ms_info = json.load(open(ms_infopath,'r'))
    fb_info = json.load(open(fb_infopath,'r'))



                # <td width='50%'>
                #     <p>
                #     ImId/ Qid: {0}, {1} <br />
                #     Question: {2}. <br />
                #     Prediction Answer: {3}. <br />
                #     GT Answer: {4}. <br />
                #     Acc: {5}. <br />
                #     OCR tokens: {6}. <br />
                #     </p>
                #     <img src='{7}' width='600px'>
                # </td>
    ms_ocr_print = ""
    fb_ocr_print = ""
    for key in ms_info:
        line_text, ind, words = key, ms_info[key][0], ms_info[key][1:]
        ms_ocr_print += "<br />Region %d: \"%s\"    Words: %s"%(ind, line_text, str(words))
    for key in fb_info:
        line_text, ind = key, fb_info[key][0]
        fb_ocr_print += "<br />Region %d: %s"%(ind, line_text)

    html = html + """
                <tr>
                <td width='50%'>
                <p>
                ImId/ Qid: {0}, {1} <br />
                Question: {2}. <br />
                Prediction Answer: {3}. <br />
                GT Answer: {4}. <br />
                Acc: {5}. <br /> <br />
                </p>
                </td>
                </tr>
                <tr>
                <td width='50%'>
                    <p>
                    FB: OCR Info: {6}. <br />
                    </p>
                    <img src='{7}' width='500px'>
                </td>
                <td width='50%'>
                    <p>
                    MS: OCR Info: {8}. <br />
                    </p>
                    <img src='{9}' width='500px'>
                </td>
            </tr>""".format(sample['image_id'],sample['question_id'],sample['question'],str(prediction['pred_answer']), str(prediction['gt_answers']), prediction['acc'],str(fb_ocr_print),fb_imgpath,str(ms_ocr_print),ms_imgpath)

html = html+"""
    </tbody></table>
    

</body></html>"""

f = open('ocr_visu.html','w')
# f = open('textvqa_visu_error.html','w')
f.write(html)
f.close()

