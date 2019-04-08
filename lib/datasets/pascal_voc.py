# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval, age_eval
from fast_rcnn.config import cfg

class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__', # always index 0
                         #'aeroplane', 'bicycle', 'bird', 'boat',
                         #'bottle', 'bus', 'automobile', 'cat', 'chair',
                         #'cow', 'diningtable', 'dog', 'horse',
                         #'motorbike', 'person', 'pottedplant',
                         #'sheep', 'sofa', 'train', 'tvmonitor',#)
                         #'head','cart','face','shadow')
                         'male', 'female')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        fp = open('./data/record.txt','w')
        gt_roidb = [self._load_pascal_annotation(index, fp)
                    for index in self.image_index]
        fp.close()
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()
        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index, fp):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        #day_flag = int(tree.find('day_flag').text)
        size_o = tree.find('size')
        width = int(size_o.find('width').text)
        height = int(size_o.find('height').text)
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
#cathy age
        age = np.zeros((num_objs), dtype = np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
           
            cls_name = obj.find('name').text.lower().strip()
            
            if cls_name == 'dinningtable':
                cls_name = 'diningtable'
            if cls_name == 'oneperson' or cls_name == '3': #or cls_name == 'shadow': #or cls_name == 'aeroplane' or cls_name == 'bottle' or cls_name == 'bird' or cls_name == 'boat' or cls_name == 'cat' or cls_name == 'cow' or cls_name == 'diningtable' or cls_name == 'dog' or cls_name == 'horse' or cls_name == 'pottedplant' or cls_name == 'sheep' or cls_name == 'sofa' or cls_name == 'train' or cls_name == 'tvmonitor' or cls_name == 'bicycle' or cls_name == 'motorbike' or cls_name == 'chair' or cls_name == 'bus':
                continue
            if cls_name == 'car':
                cls_name = 'automobile'
#            if cls_name == 'person' or cls_name == 'head':
#                if day_flag == 0:
#                    print "get one"
                    #continue
#cathy age
            age_num = int(obj.find('truncated').text)
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)-1
            y1 = float(bbox.find('ymin').text)-1
            x2 = float(bbox.find('xmax').text)-1
            y2 = float(bbox.find('ymax').text)-1
            if x2 < x1:
                print index
            if y2 < y1:
                print index
            if x1 <0.0:
                x1 = 0.0
            if x2<0.0:
                x2 = 0.0
            if y1<0.0:
                y1 = 0.0
            if y2<0.0:
                y2=0.0
            if int(x2) >= width:
                x2 = width-1
            #    print index
            #    print x1,x2,y1,y2,width,height
            if y2 >= height:
                y2 = height-1
            #cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            #print cls_name 
            #print x1,y1,x2,y2
      
            #if width != r_width:
            #    print index
            cls = self._class_to_ind[cls_name]   
            boxes[ix, :] = [x1, y1, x2, y2]
            #if  ((width - boxes[:, 2]-1) >  (width - boxes[:, 0]-1)).all():
            #    print index, boxes, width
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            age[ix] = age_num
            forwrite = cls_name + ' : '+str(x1)+', '+str(y1)+', '+str(x2)+', '+str(y2)
            fp.writelines(forwrite)
            fp.write('\n')
            #ix = ix+1
        overlaps = scipy.sparse.csr_matrix(overlaps)
   
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'age':age}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -2], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        folder_results = os.path.join(self._data_path, 'folder_results.txt')
        folder_fp = open(folder_results, 'w')
        aps = []
        folder_aps = {}
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        folder_fp.write('all result :\n')
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            print cls
            rec, prec, ap, folder_rec, folder_prec, folder_ap= voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            if len(folder_aps) == 0:
                for key in folder_ap:
                    folder_aps[key] = []
            for key in folder_ap:
                folder_aps[key] +=[folder_ap[key]]
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            folder_fp.write('AP for {} = {:.4f}\n'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            
        for key in folder_aps:
            folder_fp.write('result of '+key + ' :\n')
            it = iter(folder_aps[key])
            temp_aps = []
            for i, cls in enumerate(self._classes):
                if cls == '__background__':
                    continue
                score = it.next()
                folder_fp.write('  AP for {} = {:.4f}\n'.format(cls, score))
                if score != 0:
                    temp_aps.append(score)
            folder_fp.write('Mean AP = {:.4f}'.format(np.mean(temp_aps)))
            folder_fp.write('\n')
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        folder_fp.write('Mean AP = {:.4f}'.format(np.mean(aps)))
        folder_fp.close()

        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def _do_age_eval(self, all_boxes):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        filename = self._get_voc_results_file_template().format('age')
        
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                #dets=[]
                dets_array_list = []
                
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    d = all_boxes[cls_ind][im_ind]
                    dets_array_list.append(d)
                    #dets.append(all_boxes[cls_ind][im_ind])
                    
                dets_array = np.vstack((dets_array_list[0],dets_array_list[1]))
                if dets_array == []:
                    continue
                scores = dets_array[:,-2]
                scores = scores.tolist()
                #scores = []
                #print dets
                #for item in dets:
                #    if item.shape[0] == 0:
                #        continue
                    #print item[:,-2]
                    #scores.append(item[:,-2])
                print scores
                k = scores.index(max(scores))
                # the VOCdevkit expects 1-based indices
                print k
                f.write('{:s} {:.3f} {} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                        format(index, dets_array[k, -2], dets_array[k, -1],
                               dets_array[k, 0] + 1, dets_array[k, 1] + 1,
                               dets_array[k, 2] + 1, dets_array[k, 3] + 1))
        acc, threshold= age_eval(filename, annopath, imagesetfile)
        for i in range(len(acc)):
            print('age acc of threshold {} is {:.4f}'.format(threshold[i], acc[i]))

    def evaluate_detections(self, all_boxes, output_dir):
        self._do_age_eval(all_boxes)
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)

        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
