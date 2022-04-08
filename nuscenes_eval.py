import argparse
from nuscenes_eval_core import NuScenesEval


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_labels', type=str, required=True,
                        help='Prediction labels data path')
    parser.add_argument('--gt_labels', type=str, required=True,
                        help='Ground Truth labels data path')
    parser.add_argument('--save_loc', type=str, help='Save location')
    parser.add_argument('--format', type=str, default='class x y z l w h r score')
    parser.add_argument('--max_range', type=float, default=0.0, help='max evaluation range')
    parser.add_argument('--min_score', type=float, default=0.0, help='min detection score considered')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    NuScenesEval(args.pred_labels, args.gt_labels, args.format, args.save_loc,
                 max_range=args.max_range,
                 min_score=args.min_score)


if __name__ == '__main__':
    main()