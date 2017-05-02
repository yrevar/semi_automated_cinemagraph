'''
Implements video texture routines.
Reference:
    Schodl, Arno, et al. "Video textures." Proceedings of the 27th annual conference on Computer graphics and interactive techniques. ACM Press/Addison-Wesley Publishing Co., 2000.
'''
import os
import sys
import errno
import shutil
from glob import glob
import cv2
import scipy.signal
import scipy as sp
import numpy as np

def create_vtexture(project_dir, video_dir, alpha):

    video_dir = os.path.basename(video_dir)
    image_dir = os.path.join(project_dir, video_dir)
    out_dir = os.path.join(project_dir, "videotexture_" + video_dir)
    out_dir_imgs = os.path.join(out_dir, "images")
    try:
        _out_dir = out_dir_imgs
        not_empty = not all([os.path.isdir(x) for x in
                             glob(os.path.join(_out_dir, "*.*"))])
        if not_empty:
            shutil.rmtree(_out_dir, ignore_errors=True)
         #   raise RuntimeError("Output directory is not empty - aborting.")
        else:
            os.makedirs(_out_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print "Reading images."
    images = readImages(image_dir)

    print "Computing video texture with alpha = {}".format(alpha)
    diff1, diff2, diff3, out_list = runTexture(images, alpha)

    cv2.imwrite(os.path.join(out_dir, '{}_diff1.png'.format(video_dir)), diff1)
    cv2.imwrite(os.path.join(out_dir, '{}_diff2.png'.format(video_dir)), diff2)
    cv2.imwrite(os.path.join(out_dir, '{}_diff3.png'.format(video_dir)), diff3)

    for idx, image in enumerate(out_list):
        cv2.imwrite(os.path.join(out_dir_imgs,
                    'frame{0:04d}.png'.format(idx)), image)

def visualize_difference(diff):
    """This function normalizes the difference matrices so that they can be
    shown as images.
    """
    return (((diff - diff.min()) /
             (diff.max() - diff.min())) * 255).astype(np.uint8)

def create_video_volume(images):
    video_volume = np.zeros((len(images),) + images[0].shape, dtype=np.uint8)
    for i, image in enumerate(images):
        video_volume[i,...] = images[i].copy()
    return video_volume


def compute_similarity_metric(video_volume):
    N = len(video_volume)
    flattened_imgs = video_volume.reshape(N, -1).astype(np.float)

    similarity_matrix = np.zeros((N,N), dtype=np.float)

    for i in range(N):
        print i,
        for j in range(i, N):
            tmpval = np.sum((flattened_imgs[i] - flattened_imgs[j])**2)**0.5
            similarity_matrix[i,j] = tmpval
            similarity_matrix[j,i] = tmpval

    #p = np.exp(-similarity_matrix/similarity_matrix.mean())
    #return p / p.sum(axis=1)
    return similarity_matrix/similarity_matrix.mean()

def compute_transition_diff(similarity):
    return cv2.filter2D(similarity, -1, np.eye(5)*binomial_filter5(), borderType=cv2.BORDER_REFLECT)[2:-2,2:-2]

def find_biggest_loop(transition_diff, alpha, return_score=False):
    # row indices
    row_idxs = np.tile(np.arange(0,transition_diff.shape[0]), (transition_diff.shape[1],1)).T
    # col indices
    col_idxs = np.tile(np.arange(0,transition_diff.shape[1]), (transition_diff.shape[0],1))

    score = alpha*(col_idxs-row_idxs) - (transition_diff)
    point_pair = np.array(np.unravel_index(np.argmax(score), score.shape)) + [2,2]

    if return_score:
        return score, point_pair
    else:
        return point_pair


def synthesize_videoclip(video_volume, start, end):
    clip_volume = []
    for i in range(start, end+1):
        clip_volume.append(video_volume[i,...])
    return clip_volume


def binomial_filter5():
    return np.array([1 / 16., 1 / 4., 3 / 8., 1 / 4., 1 / 16.], dtype=float)
