import datetime

import motmetrics as mm
import numpy as np


def mot_metrics_enhanced_calculator(gt_file_path, result_file_path, out_file_path):
    # load ground truth
    gt = np.loadtxt(gt_file_path, delimiter=',')

    # load tracking output
    t = np.loadtxt(result_file_path, delimiter=',')

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height. We already have this format.
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        iou = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5)  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:, 0].astype('int').tolist(), t_dets[:, 0].astype('int').tolist(), iou)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', 'recall', 'precision', 'num_objects',
                                       'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives',
                                       'num_misses', 'num_switches', 'num_fragmentations', 'mota', 'motp'], name='acc')

    str_summary = mm.io.render_summary(summary)
    print(str_summary)

    with open(out_file_path, 'w') as fp:
        fp.write(str_summary)


if __name__ == '__main__':
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mot_metrics_enhanced_calculator("./dataset/gt/gt.txt", "./dataset/res/res.txt", f"./dataset/res/out_{time}.txt")
