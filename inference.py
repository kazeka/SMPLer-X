import os
import shutil

import argparse

def call_inference(args):

    vid = args.vid
    vid_name = vid[:-4]

    frame_path = os.path.join(args.output_path, vid_name, 'orig_img')
    # extract frames from video
    video_path = os.path.join(args.input_path, vid)

    os.makedirs(frame_path, exist_ok=True)
    os.system(f'ffmpeg -i {video_path} -f image2 ' 
                f'-vf fps={args.fps} {frame_path}/%06d.jpg')
    
    start_count = int(sorted(os.listdir(frame_path))[0].split('.')[0])
    end_count = len(os.listdir(frame_path)) + start_count - 1

    # prepare cmd for inference
    cmd_smplerx_inference = f'cd smplerx/main && python inference.py ' \
        f'--num_gpus 1 --pretrained_model {args.ckpt} ' \
        f'--agora_benchmark agora_model ' \
        f'--img_path {frame_path} --start {start_count} --end {end_count} ' \
        f'--output_folder {args.output_path}/{vid_name} '\
        f'--show_verts --show_bbox --save_mesh '
    
    os.system(cmd_smplerx_inference)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--vid', type=str, 
                        help='name of the video (with extension)')
    parser.add_argument('--input_path', type=str,
                        help='absolute path to input video folder')
    parser.add_argument('--output_path', type=str,
                        help='absolute path to output folder')

    args = parser.parse_args()

    # check format
    assert args.vid[-3:] == 'mp4', 'Only mp4 format is supported'
    
    # define
    args.format = 'mp4'
    args.fps = 30
    args.ckpt = 'smpler_x_h32'

    call_inference(args)