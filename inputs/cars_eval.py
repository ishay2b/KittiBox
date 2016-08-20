#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the MediSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

import scipy as scp
import scipy.misc

import numpy as np

import tensorflow as tf

import utils.train_utils

from utils.annolist import AnnotationLib as AnnLib


def make_val_dir(hypes):
    val_dir = os.path.join(hypes['dirs']['output_dir'], 'val_out')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    return val_dir


def make_img_dir(hypes):
    val_dir = os.path.join(hypes['dirs']['output_dir'], 'val_images')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    return val_dir


def write_rects(rects, filename):
    with open(filename, 'w') as f:
        for rect in rects:
            string = "Car 0 1 0 %f %f %f %f 0 0 0 0 0 0 0 %f" % \
                (rect.x1, rect.y1, rect.x2, rect.y2, rect.score)
            print(string, file=f)


def evaluate(hypes, sess, image_pl, softmax):
    pred_annolist, true_annolist, image_list = get_results(hypes, sess,
                                                           image_pl, softmax)

    val_path = make_val_dir(hypes)
    subprocess.check_call(["evaluate_object", val_path])
    res_file = os.path.join(val_path, "stats_car_detection.txt")

    eval_list = []
    with open(res_file) as f:
        for mode in ['easy', 'medium', 'hard']:
            line = f.readline()
            result = np.array(line.rstrip().split(" ")).astype(float)
            mean = np.mean(result)
            eval_list.append((mode, mean))

    return eval_list, image_list


def get_results(hypes, sess, image_pl, softmax):
    pred_boxes, pred_confidences = softmax

    # Build Placeholder
    shape = [hypes['image_height'], hypes['image_width'], 3]

    pred_annolist = AnnLib.AnnoList()
    test_idl = os.path.join(hypes['dirs']['data_dir'],
                            hypes['data']['val_idl'])
    true_annolist = AnnLib.parse(test_idl)

    data_dir = os.path.dirname(test_idl)
    val_dir = make_val_dir(hypes)
    img_dir = make_img_dir(hypes)

    image_list = []

    for i in range(len(true_annolist)):
        true_anno = true_annolist[i]
        orig_img = scp.misc.imread('%s/%s' % (data_dir,
                                              true_anno.imageName))[:, :, :3]
        img = scp.misc.imresize(orig_img, (hypes["image_height"],
                                           hypes["image_width"]),
                                interp='cubic')
        feed = {image_pl: img}
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
                                                         pred_confidences],
                                                        feed_dict=feed)
        pred_anno = AnnLib.Annotation()
        pred_anno.imageName = true_anno.imageName
        new_img, rects = utils.train_utils.add_rectangles(
            hypes, [img], np_pred_confidences,
            np_pred_boxes, show_removed=False,
            use_stitching=True, rnn_len=hypes['rnn_len'],
            min_conf=0.001, tau=hypes['tau'])

        if i % 15 == 0:
            image_name = os.path.basename(pred_anno.imageName)
            image_name = os.path.join(img_dir, image_name)
            scp.misc.imsave(image_name, new_img)
        # get name of file to write to
        image_name = os.path.basename(true_anno.imageName)
        val_file_name = image_name.split('.')[0] + '.txt'
        val_file = os.path.join(val_dir, val_file_name)

        # write rects to file

        pred_anno.rects = rects
        pred_anno.imagePath = os.path.abspath(data_dir)
        pred_anno = utils.train_utils.rescale_boxes((
            hypes["image_height"],
            hypes["image_width"]),
            pred_anno, orig_img.shape[0],
            orig_img.shape[1])

        write_rects(rects, val_file)

        pred_annolist.append(pred_anno)

        image_list = []

    return pred_annolist, true_annolist, image_list