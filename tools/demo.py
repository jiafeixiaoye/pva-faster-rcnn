#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name, classes, result_path):
    """Detect object classes in an image using pre-computed object proposals."""
 
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
 
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    scores, boxes = im_detect(net, im, _t)
    #np.savetxt("E:\\Cathycode\\deeplearning\\pva-faster-rcnn-master\\data\\scores.txt",scores,fmt='%s',newline='\n')
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
 
    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.3
    flag = 0
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        plt.clf()
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
        pic_name = os.path.join(result_path, cls+image_name)
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds):
            plt.savefig(pic_name)
    #for cls in classes:
    #    cls_ind = CLASSES.index(cls)
    #    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    #    cls_scores = scores[:, cls_ind]
    #    if flag == 0:
    #        flag = flag + 1
    #    keep = np.where(cls_scores >= CONF_THRESH)[0]
    #    cls_boxes = cls_boxes[keep, :]
     #    cls_scores = cls_scores[keep]
     #   dets = np.hstack((cls_boxes,
     #                     cls_scores[:, np.newaxis])).astype(np.float32)
     #   order = cls_scores.argsort()[::-1]
     #   #print order
     #   keep = nms(dets, NMS_THRESH)
     #   #print keep
     #   dets = dets[keep, :]
     #   print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
     #                                                               CONF_THRESH)
     #   vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALE_MULTIPLE_OF=32
    cfg.TEST.MAX_SIZE =2000
    cfg.TEST.SCALE = (600,)
    cfg.TEST.BBOX_VOTE = True
    cfg.TEST.NMS = 0.4
    cfg.TEST.RPN_PRE_NMS_TOP_N = 12000
    cfg.TEST.RPN_POST_NMS_TOP_N = 200

    args = parse_args()

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
     #                       'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
     #                         NETS[args.demo_net][1])
    prototxt = "/home/work-station/pva-faster-rcnn/models/pvanet/pva9.1/faster_rcnn_train_test_21cls.pt"
    caffemodel = "/home/work-station/pva-faster-rcnn/models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712.caffemodel"

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    test_path = os.path.join(cfg.ROOT_DIR, 'data', 'demo')
    result_path = os.path.join(cfg.ROOT_DIR, 'data','result')
 
    index = 1
    for parent,dirnames,filenames in os.walk(test_path):
        for filename in filenames:
            name = filename.split('.')
            if len(name)== 3 or len(name) == 2:
                print filename
                sample_name = filename.split('.')
                plt.clf()
                demo(net, filename, ('person',), result_path)
                pic_name = os.path.join(result_path,filename)
                plt.savefig(pic_name)
                print "%d done"%(index)
                index = index+1
 
#    im_names = ['whole19691231T190149.444000.jpg', '000542.jpg', '001150.jpg',
#                '001763.jpg', '004545.jpg']
#    for im_name in im_names:
#        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
#        print 'Demo for data/demo/{}'.format(im_name)
 #       demo(net, im_name)
 
    #plt.show()
