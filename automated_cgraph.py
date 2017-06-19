'''
Implements cinemagraph class and a simple GUI.
'''
import cv2
import numpy as np
import scipy as sp
import os
import glob
import shutil
import imageio

import matplotlib.pyplot as plt
from video_texture import *
from blending_utils import *


from enum import Enum, unique
@unique
class CGRAPHState(Enum):
    IDLE = 0
    START = 1
    WAITING = 2
    WRITING = 3
    READY = 4
    LOOP_CREATE = 5
    LOOP_READY = 6


class ImageHelper(object):

    def __init__(self, set_dir, img_encoding, file_prefix=None, pad_len=4):

        self.set_dir = set_dir
        self.img_encoding = img_encoding
        self.pad_len = pad_len #len(str(max_imgs))
        self.file_prefix = file_prefix if file_prefix else os.path.basename(set_dir)

    def path_by_no(self, img_no):
        """
        Parse file name using format: <./path_to_set/image_set>/<image_set>_<0001>.<img_encoding>
        """
        file_name = self.file_prefix +                         "_" + "{:0{pad_len}d}".format(img_no, pad_len=4) + "." + self.img_encoding

        return os.path.join(self.set_dir, file_name)

    def img_by_no(self, img_no):

        img = cv2.imread(self.path_by_no(img_no), cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
        if img is not None:
            return img
        else:
            raise ValueError('failed to read image {}'.format(self.path_by_no(img_no)))

    def frames(self, start=1, end=None): # TODO: speedup by caching

        if end is None:
            end = len(glob(self.set_dir + "/*." + self.img_encoding))-start
        return [self.img_by_no(i) for i in range(start, end)]

class VideoHelper(object):

    def __init__(self, video_file, frame_start=0, frame_end=np.float('inf'), do_loop=True, preload_frames=500):

        self.preloaded = False
        self.preload_frames = preload_frames
        self._init_capture(video_file, frame_start, frame_end)
        self.do_loop = do_loop
        self.paused = False

    def restart_capture(self):
        self.curr_frame_idx = self.frame_start

    def toggle_play(self):
        self.paused = not self.paused

    def is_paused(self):
        return self.paused

    def curr_frame_no(self):
        return self.curr_frame_idx-self.frame_start

    def _init_capture(self, video_file, frame_start, frame_end):

        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to initialize capture for file {}".format(video_file))

        self.n_orig_frames = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self.frame_start = max(frame_start, 0)

        if frame_end < 0:
            self.frame_end = max(self.n_orig_frames+frame_end, self.frame_start)
        else:
            self.frame_end = min(self.n_orig_frames-1, frame_end)

        self.curr_frame_idx = self.frame_start

        self.n_frames = (self.frame_end - self.frame_start + 1)

        if self.n_frames < self.preload_frames:
            self._preload()

    def _preload(self):

        self.frames = []
        self.preloaded = False

        for frame_no in range(self.frame_start, self.frame_end+1):
            self.frames.append(self.get_frame_by_idx(frame_no))

        self.preloaded = True

    def is_finished(self):
        return self.curr_frame_idx > self.frame_end and not self.do_loop

    def get_frame_by_idx(self, idx):

        if self.preloaded:
            idx -= self.frame_start
            return self.frames[idx]
        else:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read(1)
            if ret == False:
                raise RuntimeError("Failed to read frame no. {} from file {}".format(idx, self.video_file))
            return frame

    def seek_next(self, skip=1):

        self.curr_frame_idx += skip

        if self.curr_frame_idx > self.frame_end and self.do_loop:
            self.curr_frame_idx = self.frame_start

    def seek_prev(self, skip=1):

        idx = max(self.frame_start, self.curr_frame_idx-skip)
        self.curr_frame_idx = idx

    def curr_frame(self):

        idx = self.curr_frame_idx

        if idx > self.frame_end:
            raise RuntimeError('No more frames! You should have known earlier by calling is_finished()')

        frame = self.get_frame_by_idx(idx)
        return frame

    def next_frame(self, skip=1):

        if self.paused:
            skip = 0

        frame = self.curr_frame()

        self.seek_next(skip)

        return frame

    def prev_frame(self, skip=1):

        idx = max(self.frame_start, self.curr_frame_idx-skip)
        frame = self.get_frame_by_idx(idx)
        self.curr_frame_idx = idx
        return frame

    def __del__(self):
        self.cap.release()

class Cinemagraph(object):

    def __init__(self, project_dir, video_file, results_path="output", img_format="png",
                        blend_levels=3, alpha=None, frame_start=0, frame_end=np.float('inf')):

        self.project_dir = project_dir
        self.img_format = img_format
        self.set_input_video(video_file, results_path)
        self.blend_levels = blend_levels
        self.video = VideoHelper(video_file, frame_start, frame_end)
        self.init_params()

        if alpha:
            self.override_alpha = True
            self.vtexture_alpha = alpha
        else:
            self.override_alpha = False
            self.vtexture_alpha = 0.1

        cv2.namedWindow('Cinemagraph')
        cv2.setMouseCallback('Cinemagraph', self.draw_circle)
        self.cgraph_state_str = {
                    CGRAPHState.IDLE: "Idle",
                    CGRAPHState.START: "Starting",
                    CGRAPHState.WAITING: "Waiting..",
                    CGRAPHState.WRITING: "Writing ",
                    CGRAPHState.READY: "Ready",
                    CGRAPHState.LOOP_CREATE: "Looping",
                    CGRAPHState.LOOP_READY: "Finished" }

    def set_input_video(self, video_file, results_path):

        self.video_file = video_file
        self.indir = os.path.dirname(video_file)
        self.video_dir = os.path.basename(self.indir)
        self.outdir = os.path.join(self.project_dir, results_path, self.video_dir)
        self.outdir_cgraph = os.path.join(self.outdir, "cgraph")
        self.outdir_vtexture = os.path.join(self.outdir, "create_vtexture")

        if not os.path.exists(results_path):
            os.mkdir(results_path)

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        if not os.path.exists(self.outdir_cgraph):
            os.mkdir(self.outdir_cgraph)

        if not os.path.exists(self.outdir_vtexture):
            os.mkdir(self.outdir_vtexture)

    def init_params(self):
        self.drawing = False # true if mouse is pressed
        self.draw_alpha = 0.3
        self.poly = None
        self.poly_list = {}
        self.poly_idx = 0
        self.static_img, self.static_img_idx = None, None
        self.mask_ready = False
        self.fs, self.fe = None, None
        self.first_frame = self.video.curr_frame().copy()
        self.show_mask_always = False
        self.create_cinemagraph = False
        self.cgraph_ready = False
        self.cgraph_fame_no = 0
        self.cgraph_fames_cnt = 0
        self.loop_process_state = 0
        self.vtexture_alpha = 0.1
        self.vtexture_pair_idxs = None
        self.cgraph_state = CGRAPHState.IDLE
        self.static_img_mask = np.zeros_like(self.first_frame)
        self.cinemagraph_gif_result = None

    # mouse callback function
    def draw_circle(self, event,x,y,flags,param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            ix,iy = x,y
            self.poly = np.array([[ix, iy]], dtype=np.int32)
            self.poly_list[self.poly_idx] = self.poly

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.poly = np.vstack((self.poly, [x, y])).astype(np.int32)
                self.poly_list[self.poly_idx] = self.poly

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.mask_ready = True
            self.poly_idx += 1

    def create_vtexture_help_menu(self, img, pair_idxs, saved_gif=False):

        pad_img = np.zeros((img.shape[0], 200, img.shape[2]), dtype=img.dtype)
        img = np.hstack((img, pad_img))

        loc = (img.shape[1]-190, 15)
        text = "\n -- Alpha {}: Loop {}-{} -- \
                \n 'y':  save result \
                \n 'ESC': quit".format(self.vtexture_alpha, pair_idxs[0], pair_idxs[1])

        y0, dy = loc[1], 15
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(img, line, (loc[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)


        if saved_gif:
            msg = "Saved gif at: " + self.cinemagraph_gif_result
            cv2.putText(img, msg,(20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        return img

    def save_create_vtexture(self, img_list, diff1, diff2, diff3, create_gif=True):

        if os.path.exists(self.outdir_vtexture):
            shutil.rmtree(self.outdir_vtexture, ignore_errors=True)

        os.mkdir(self.outdir_vtexture)

        cv2.imwrite(os.path.join(self.outdir, '{}_diff1.png'.format(self.video_dir)), visualize_difference(diff1))
        cv2.imwrite(os.path.join(self.outdir, '{}_diff2.png'.format(self.video_dir)), visualize_difference(diff2))
        cv2.imwrite(os.path.join(self.outdir, '{}_diff3.png'.format(self.video_dir)), visualize_difference(diff3))

        for idx, image in enumerate(img_list):
            cv2.imwrite(os.path.join(self.outdir_vtexture, 'frame{0:04d}.png'.format(idx)), image)
            img_list[idx] = img_list[idx][...,::-1]

        # create gif
        if create_gif:
#             images = [Image.fromarray(x) for x in img_list]
#             writeGif(os.path.join(self.outdir, self.video_dir + "_cinemagraph.gif"), images, duration=0.2)
            self.cinemagraph_gif_result = os.path.join(self.outdir, self.video_dir + "_cinemagraph.gif")
            imageio.mimsave(self.cinemagraph_gif_result, img_list)

        for idx, image in enumerate(img_list):
            img_list[idx] = img_list[idx][...,::-1]

    def show_update(self, img, msg, window, location='auto'):

        if location == 'auto':
            H, W = img.shape[0], img.shape[1]
            location = (min(200,W-200), min(30, H))

        img = cv2.putText(img, msg, location, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow(window, img)
        k = cv2.waitKey(150) & 0xFF
        return img

    def help_menu(self, img, video):

        frame_no = video.curr_frame_no()
        pad_img = np.zeros((img.shape[0], 300, img.shape[2]), dtype=img.dtype)
        img = np.hstack((img, pad_img))

        cgraph_msg = self.cgraph_state_str[self.cgraph_state]
        if self.cgraph_state == CGRAPHState.WRITING:
            cgraph_msg += str(self.cgraph_fames_cnt)

        loc = (img.shape[1]-290, 15)
        alpha_msg = "Auto" if not self.override_alpha else "User"

        text = "Frame: {} \
                \n Still Frame: {} \
                \n Mask : {} \
                \n Motion Start: {} \
                \n Motion End: {} \
                \n CGraph: {}\
                \n {} Alpha: {}, Loop: {}".format(
                    frame_no,
                    self.static_img_idx,
                    "Yes" if self.mask_ready else "No",
                    self.fs, self.fe, cgraph_msg, alpha_msg, self.vtexture_alpha, self.vtexture_pair_idxs)

        y0, dy = loc[1], 15
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(img, line, (loc[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        helptext="---Help--- \
                \n 'space': pause \
                \n 's': still frame \
                \n 'q': dynamic start \
                \n 'e': dynamic end \
                \n 'c': create cinemagraph \
                \n 'r': reset cinemagraph \
                \n 'v': create video loop \
                \n 'm': toggle mask \
                \n '>': next frame \
                \n '<': previous frame \
                \n 'V': lower alpha \
                \n '^': increase alpha \
                \n 'j': skip 10 frames \
                \n 'k': rewind 10 frames \
                \n 'R': restart video \
                \n 'ESC': Exit"

        y0, dy = y + 12, 15
        for i, line in enumerate(helptext.split('\n')):
            y = y0 + i*dy
            cv2.putText(img, line, (loc[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        return img


    def get_best_videotexture_pair(self, tdiff, alpha_range=np.linspace(0,4,1000)):

        dd = {} # loop idxs: score
        for alpha in alpha_range:
            score, idxs = find_biggest_loop(tdiff, alpha, True)
            # tdiff score / std(tdiff row score)
            dd[idxs[0], idxs[1]] = (score[idxs[0]-2, idxs[1]-2]/score[idxs[0]-2,:].std(), alpha, score)

        best_score_idx = np.argmax([t[0] for t in dd.values()])
        best_pair = dd.keys()[best_score_idx]
        return best_pair, dd[best_pair][1], dd[best_pair][2]

    def launch_user_iterface(self):

        while not self.video.is_finished():

            frame_no = self.video.curr_frame_no()
            img = self.video.next_frame()
            render_img = img.copy()

            if self.drawing:
                if len(self.poly_list) != 0:
                    cv2.fillPoly(self.static_img_mask, self.poly_list.values(), (255, 255, 255))
                    self.poly_list = {}
                render_img[self.static_img_mask == 255] = self.draw_alpha*img[self.static_img_mask == 255] \
                                                            + (1-self.draw_alpha)*255

            if self.mask_ready and (self.video.is_paused() or self.show_mask_always):
                render_img[self.static_img_mask == 255] = self.draw_alpha*img[self.static_img_mask == 255] \
                                                            + (1-self.draw_alpha)*255

            if self.mask_ready and not self.drawing and self.static_img is not None:
                dst = self.static_img
                src = img
                render_img = blend_images(dst, src, self.static_img_mask, self.blend_levels).astype(np.uint8)

                if (self.cgraph_state == CGRAPHState.START \
                    or self.cgraph_state == CGRAPHState.WRITING \
                    or self.cgraph_state == CGRAPHState.WAITING) \
                    and not self.video.is_paused():

                    if not self.fs and not self.fe:

                        if self.cgraph_fames_cnt == self.video.n_frames-1:
                            self.cgraph_state = CGRAPHState.READY
                        else:
                            self.cgraph_state = CGRAPHState.WRITING
                            self.outpath = os.path.join(self.outdir_cgraph, "frame_{:04d}".format(frame_no) + "." + self.img_format)
                            cv2.imwrite(outpath, render_img)
                            self.cgraph_fames_cnt += 1

                    elif self.fs and self.fe:

                        self.cgraph_state = CGRAPHState.WAITING
                        self.cgraph_fame_no = -1

                        if self.fe > self.fs and (frame_no - self.fs) <= abs(self.fe-self.fs):
                            cgraph_fame_no = frame_no - self.fs

                            if cgraph_fame_no >= 0:
                                outpath = os.path.join(self.outdir_cgraph, "frame_{:04d}".format(cgraph_fame_no) + "." + self.img_format)
                                cv2.imwrite(outpath, render_img)
                                self.cgraph_state = CGRAPHState.WRITING
                                self.cgraph_fames_cnt += 1

                            if self.cgraph_fames_cnt > abs(self.fe-self.fs):
                                self.cgraph_state = CGRAPHState.READY

                        elif self.fs > self.fe:

                            if frame_no <= self.fe:
                                cgraph_fame_no = (self.video.n_frames) + (frame_no-self.fs)
                            else:
                                cgraph_fame_no = (frame_no-self.fs)

                            if cgraph_fame_no >= 0 and cgraph_fame_no <= abs(self.fe-self.fs):
                                outpath = os.path.join(self.outdir_cgraph, "frame_{:04d}".format(cgraph_fame_no) + "." + self.img_format)
                                cv2.imwrite(outpath, render_img)
                                self.cgraph_state = CGRAPHState.WRITING
                                self.cgraph_fames_cnt += 1

                            if self.cgraph_fames_cnt > self.video.n_frames - self.fs + self.fe:
                                self.cgraph_state = CGRAPHState.READY


            render_img = self.help_menu(render_img, self.video)
            cv2.imshow('Cinemagraph', render_img)
            k = cv2.waitKey(1) & 0xFF

            if self.cgraph_state == CGRAPHState.LOOP_CREATE:

                CGraphImgs = ImageHelper(self.outdir_cgraph, self.img_format, file_prefix="frame")
                frames = CGraphImgs.frames(0, self.cgraph_fames_cnt)
                video_volume = create_video_volume(frames)
                ssd_diff = compute_similarity_metric(video_volume)
                transition_diff = compute_transition_diff(ssd_diff)

                if not self.override_alpha:
                    # automatically derive alpha and best pair
                    self.vtexture_pair_idxs, self.vtexture_alpha, best_score = self.get_best_videotexture_pair(transition_diff)
                else:
                    best_score, self.vtexture_pair_idxs = find_biggest_loop(transition_diff, self.vtexture_alpha, True)

                self.cgraph_state = CGRAPHState.LOOP_READY

                start, end = self.vtexture_pair_idxs
                loop = frames[start: end+1]
                i = 0
                saved_gif = False
                while len(loop) != 0:

                    render_img2 = self.create_vtexture_help_menu(loop[i], self.vtexture_pair_idxs, saved_gif)
                    cv2.imshow('Result', render_img2)
                    kk = cv2.waitKey(18) & 0xFF

                    if kk == ord('y'):
                        self.save_create_vtexture(loop, ssd_diff, transition_diff, best_score)
                        saved_gif = True
                    elif kk == 27:
                        cv2.destroyWindow("Result")
                        break

                    i += 1
                    if i == len(loop):
                        i = 0

                self.cgraph_state = CGRAPHState.READY

            if k == 32: # space: play/pause
                self.video.toggle_play()
                state = "pause" if self.video.is_paused() else "play"
                render_img = self.show_update(render_img, state, 'Cinemagraph')
                print state, "video"
            elif k == 3: # > arrow key
                self.video.seek_next()
                self.show_update(render_img, "Seek Next", 'Cinemagraph')
            elif k == 2: # < arrow key
                self.video.seek_prev()
                self.show_update(render_img, "Seek Prev", 'Cinemagraph')
            elif k == ord('k'): # seek 10 frames forward
                self.video.seek_next(10)
                self.show_update(render_img, "Fwd 10", 'Cinemagraph')
            elif k == ord('j'): # seek 10 frames backward
                self.video.seek_prev(10)
                self.show_update(render_img, "Rwd 10", 'Cinemagraph')
            elif k == ord('s'): # capture static frame
                self.static_img = img.copy()
                self.static_img_idx = frame_no
                self.show_update(render_img, "Static Frame Captured", 'Cinemagraph')
            elif k == ord('q'): # dynamic frame start
                self.fs = frame_no
                print "set start anim frame", self.fs
                self.show_update(render_img, "Set Dynamic Frame Start: {}".format(frame_no) , 'Cinemagraph')
            elif k == ord('e'): # dynamic frame end
                self.fe = frame_no
                print "end start anim frame", self.fe
                self.show_update(render_img, "Set Dynamic Frame End: {}".format(frame_no) , 'Cinemagraph')
            elif k == ord('r'): # reset
                self.init_params()
                print "reset"
                self.show_update(render_img, "Reset Cinemagraph Config", 'Cinemagraph')
            elif k == ord('m'): # mask visibility toggle
                self.show_mask_always = not self.show_mask_always
                print "show mask always"
                self.show_update(render_img, "Show Mask Always" , 'Cinemagraph')
            elif k == ord('v'):
                print "creating video texture"
                if self.cgraph_state == CGRAPHState.READY:
                    self.cgraph_state = CGRAPHState.LOOP_CREATE
                self.show_update(render_img, "Create Video Loop" , 'Cinemagraph')
            elif k == 0:
                self.vtexture_alpha *= 3.0
                self.override_alpha = True
                print "alpha increase"
                self.show_update(render_img, "Alpha: {}".format(self.vtexture_alpha) , 'Cinemagraph')
            elif k == 1:
                self.vtexture_alpha /= 3.0
                self.override_alpha = True
                print "alpha decrease"
                self.show_update(render_img, "Alpha: {}".format(self.vtexture_alpha) , 'Cinemagraph')
            elif k == ord('c'): # create cinemagraph

                print "creating cinemagraph..."
                self.show_update(render_img, "Generating Cinemagraph" , 'Cinemagraph')
                self.cgraph_state = CGRAPHState.START

                if os.path.exists(self.outdir_cgraph):
                    shutil.rmtree(self.outdir_cgraph, ignore_errors=True)

                os.mkdir(self.outdir_cgraph)
                self.cgraph_fames_cnt = 0
            elif k == ord('R'): # restart
                print "restarting video..."
                self.video.restart_capture()
                self.show_update(render_img, "Restart Video" , 'Cinemagraph')
            elif k == 27:
                print "exiting..."
                self.show_update(render_img, "Exiting..." , 'Cinemagraph')
                cv2.destroyWindow("Cinemagraph")
                cv2.destroyAllWindows()
                break

# %cd $cinemagraph.outdir_vtexture
# !ffmpeg -framerate 24 -i ./frame%04d.png ../output.gif
