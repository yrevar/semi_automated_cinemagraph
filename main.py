from automated_cgraph import *
import argparse

PROJ_HOME=os.path.abspath(os.curdir)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help="cinemagraph input video", dest='video', nargs=1, required=True)
    parser.add_argument("-l", "--blend", type=int, help="iamge blending levels", dest='blend', nargs=1, required=False)
    parser.add_argument("-a", "--alpha", type=int, help="disable video texture auto alpha selection", dest='alpha', nargs=1, required=False)

    args = parser.parse_args()
    video_file = args.video[0]
    if args.blend:
        blend_levels = args.blend[0]
    else:
        blend_levels = 4

    alpha = None
    if args.alpha:
        alpha = args.alpha[0]

    cinemagraph = Cinemagraph(PROJ_HOME, video_file, blend_levels=blend_levels, alpha=alpha)
    cinemagraph.launch_user_iterface()
