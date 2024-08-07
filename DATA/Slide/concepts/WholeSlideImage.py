"""
This file is use to define WSI i/o related class for the project

(1) WSI IO with svs format

"""
##############################################################################
#             related import
###############################################################################
import sys 
from PIL import Image
import math
import os
import time
# import xml.etree.ElementTree as ET
from xml.dom import minidom
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
# import pdb
# import h5py
import math
# from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, coord_generator, save_hdf5, sample_indices, screen_coords, isBlackPatch, isWhitePatch, to_percentiles
# import itertools
# from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
# from utils.file_utils import load_pkl, save_pkl
from scipy.stats import rankdata


from pathlib import Path

Image.MAX_IMAGE_PIXELS = 933120000
sys.path.append("..") 
sys.path.append(".") 
##############################################################################
#             import from other files
###############################################################################
from HistoMIL import logger
from HistoMIL.DATA.FileIO.wsi_backends import select_backends
from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.DATA.Slide.collector.items  import Items

#############################################################################
#              Whole Slide Image Source
#############################################################################
class WholeSlideImage(Items):
    def __init__(self,db_loc:Locations,wsi_loc:str,paras=None) -> None:
        logger.debug("Init instance for Whole Slide Image.")
        self.db_loc = db_loc
        self.paras = paras
        self.wsi_loc = Path(str(self.db_loc.abs_loc("slide"))+"/" + str(wsi_loc))
        self.worker = select_backends(self.wsi_loc)
        #self.get_meta()

    def set_backends(self,loc:str=None):
        actural_loc = Path(loc) if loc is not None else Path(self.wsi_loc)
        self.worker = select_backends(actural_loc)

    def pre_requests(self):
        return []

    def get(self):
        self.read()
    
    def loc(self):
        return self.wsi_loc

    def read(self):
        #logger.debug(f"WSI::Initialise file handler for {self.loc()}.")
        self.worker._open_handler_()
        self.get_meta()

    def len(self):
        return 1

    def close(self):
        #logger.debug(f"WSI::Remove file handler for {self.loc()}.")
        self.worker._close_handler_()

    def get_region(self, coords, patch_size, patch_level,**kwargs):
        #logger.debug(f"WSI::get region at {coords} with patch size {patch_size} at level {patch_level}.")
        img = self.worker.get_region(coords, patch_size, patch_level,**kwargs)
        return img

    def get_thumbnail(self, size, **kwargs):
        logger.debug(f"WSI::get thumbnail with size {size}.")
        return self.worker.get_thumbnail(size,**kwargs)

    def get_meta(self,):
        self.meta = self.worker.get_meta()


class WholeSlideImageHeatmap(object):
    def __init__(self, path):

        """
        Args:
            path (str): fullpath to WSI file
        """

#         self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.wsi = openslide.open_slide(path)
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions
    
        self.contours_tissue = None
        self.contours_tumor = None
        self.hdf5_file = None

    def getOpenSlide(self):
        return self.wsi
    
    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]
        
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
        
        return level_downsamples
    
    def screen_coords(self, scores, coords, top_left, bot_right):
        bot_right = np.array(bot_right)
        top_left = np.array(top_left)
        mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
        scores = scores[mask]
        coords = coords[mask]
        return scores, coords
    
    def to_percentiles(self, scores):
        from scipy.stats import rankdata
        scores = rankdata(scores, 'average')/len(scores) * 100   
        return scores
    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]


    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0,0)):
        print('\ncomputing foreground tissue mask')
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        contours_tissue = self.scaleContourDim(self.contours_tissue, scale)
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        contours_holes = self.scaleHolesDim(self.holes_tissue, scale)
        contours_tissue, contours_holes = zip(*sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
        for idx in range(len(contours_tissue)):
            cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1), offset=offset, thickness=-1)

            if use_holes:
                cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1)
            # contours_holes = self._scaleContourDim(self.holes_tissue, scale, holes=True, area_thresh=area_thresh)
                
        tissue_mask = tissue_mask.astype(bool)
        print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
        return tissue_mask
    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        print('\ncomputing blend')
        downsample = self.level_downsamples[vis_level]
        w = img.shape[1]
        h = img.shape[0]
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print('using block size: {} x {}'.format(block_size_x, block_size_y))

        shift = top_left # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                #print(x_start, y_start)

                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))
                
                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img+block_size_y)
                x_end_img = min(w, x_start_img+block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue
                #print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))
                
                # 3. fetch blend block and size
                blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img] 
                blend_block_size = (x_end_img-x_start_img, y_end_img-y_start_img)
                
                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)
                    canvas = np.array(self.wsi.read_region(pt, vis_level, blend_block_size).convert("RGB"))     
                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

                # 5. blend color block and canvas block
                img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
        return img
    def visHeatmap(self, scores, coords, vis_level=-1, 
                   top_left=None, bot_right=None,
                   patch_size=(224, 224), 
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4, 
                   blur=False, overlap=0.0, 
                   segment=True, use_holes=True,
                   convert_to_percentiles=False, 
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample = 1,
                   cmap='coolwarm'):

        """
        Args:
            scores (numpy array of float): Attention scores 
            coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
            vis_level (int): WSI pyramid level to visualize
            patch_size (tuple of int): Patch dimensions (relative to lvl 0)
            blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
            canvas_color (tuple of uint8): Canvas color
            alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
            blur (bool): apply gaussian blurring
            overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
            segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that 
                            self.contours_tissue and self.holes_tissue are not None
            use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
            convert_to_percentiles (bool): whether to convert attention scores to percentiles
            binarize (bool): only display patches > threshold
            threshold (float): binarization threshold
            max_size (int): Maximum canvas size (clip if goes over)
            custom_downsample (int): additionally downscale the heatmap by specified factor
            cmap (str): name of matplotlib colormap to use
        """

        if vis_level < 0:
            vis_level = self.wsi.get_best_level_for_downsample(32)

        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level
                
        if len(scores.shape) == 2:
            scores = scores.flatten()

        if binarize:
            if thresh < 0:
                threshold = 1.0/len(scores)
                
            else:
                threshold =  thresh
        
        else:
            threshold = 0.0

        ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
        if top_left is not None and bot_right is not None:
            scores, coords = self.screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)

        else:
            region_size = self.level_dim[vis_level]
            top_left = (0,0)
            bot_right = self.level_dim[0]
            w, h = region_size

        patch_size  = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)
        
        print('\ncreating heatmap for: ')
        print('top_left: ', top_left, 'bot_right: ', bot_right)
        print('w: {}, h: {}'.format(w, h))
        print('scaled patch size: ', patch_size)

        ###### normalize filtered scores ######
        if convert_to_percentiles:
            scores = self.to_percentiles(scores) 

        scores /= 100
        
        ######## calculate the heatmap of raw attention scores (before colormap) 
        # by accumulating scores over overlapped regions ######
        
        # heatmap overlay: tracks attention score over each pixel of heatmap
        # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)      
        count = 0
        for idx in range(len(coords)):
            # MAIN COMPUTATIONAL BOTTLENECK
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score=1.0
                    count+=1
            else:
                score=0.0
            # accumulate attention
            overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
            # accumulate counter
            counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))
        
        # fetch attended region and average accumulated attention
        zero_mask = counter == 0

        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter 
        if blur:
            overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if segment:
            tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
            # return Image.fromarray(tissue_mask) # tissue mask
        
        if not blank_canvas:
            # downsample original image and use as canvas
            img = np.array(self.wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
        else:
            # use blank canvas
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255))) 

        #return Image.fromarray(img) #raw image

        print('\ncomputing heatmap image')
        print('total of {} patches'.format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        
        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print('progress: {}/{}'.format(idx, len(coords)))
            
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:

                # attention block
                raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                
                # image block (either blank canvas or orig image)
                img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

                # color block (cmap applied to attention block)
                color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

                if segment:
                    # tissue mask block
                    mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] 
                    # copy over only tissue masked portion of color block
                    img_block[mask_block] = color_block[mask_block]
                else:
                    # copy over entire color block
                    img_block = color_block

                # rewrite image block
                img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()
        
        #return Image.fromarray(img) #overlay
        print('Done')
        del overlay

        if blur:
            img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if alpha < 1.0:
            img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024)
        
        img = Image.fromarray(img)
        w, h = img.size

        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img
    

#### UTILS for SAMPLING PATCHES FOR A PREDICTION. CODE COPIED FROM https://github.com/mahmoodlab/CLAM/blob/master/create_heatmaps.py

def to_percentiles(scores):
    
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores

def top_k(scores, k, invert=False):
    if invert:
        top_k_ids=scores.argsort()[:k]
    else:
        top_k_ids=scores.argsort()[::-1][:k]
    return top_k_ids

def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    np.random.seed(seed)
    if convert_to_percentile:
        end_value = np.quantile(scores, end)
        start_value = np.quantile(scores, start)
    else:
        end_value = end
        start_value = start
    score_window = np.logical_and(scores >= start_value, scores <= end_value)
    indices = np.where(score_window)[0]
    if len(indices) < 1:
        return -1 
    else:
        return np.random.choice(indices, min(k, len(indices)), replace=False)

def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):

    if len(scores.shape) == 2:
        scores = scores.flatten()

    scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    else:
        raise NotImplementedError
    # pdb.set_trace()
    coords = coords[:][sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset
