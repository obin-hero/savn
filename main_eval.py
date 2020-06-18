from __future__ import print_function, division

import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp

import time
import numpy as np
import random
import json
from tqdm import tqdm

from utils.net_util import ScalarMeanTracker
from runners import nonadaptivea3c_val

import imageio
import cv2

def main_eval(args, create_shared_model, init_agent):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = args.load_model

    processes = []

    res_queue = mp.Queue()
    args.learned_loss = False
    args.num_steps = 50
    target = nonadaptivea3c_val

    rank = 0
    episode_num = 50
    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                episode_num,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=episode_num * proc)
    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)
            with imageio.get_writer(outputs/'%03d_%s.mp4'%(count,train_result['target']), mode='I', fps=10) as writer:
                for t, image in enumerate(train_result['frames']):
                    img = image.astype(np.uint8)
                    target_name = train_result['target']
                    cv2.putText(img, 'step: %d target: %s'%(t, target_name),(20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                    if t == len(train_result['frames']) - 1:
                        if train_result['success']:
                            cv2.putText(img, 'SUCCESS! spl: %.2f'%(train_result['spl']), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
                        else:
                            cv2.putText(img, 'FAIL!', (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
                    writer.append_data(img)

        tracked_means = train_scalars.pop_and_reset()

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()

    with open(args.results_json, "w") as fp:
        json.dump(tracked_means, fp, sort_keys=True, indent=4)
