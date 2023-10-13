import glob
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from label_parser import LabelParser

class NuScenesEval:
    def __init__(self, pred_label_path, gt_label_path, label_format, save_loc,
                 classes=['car', 'pedestrian', 'cyclist'],
                 distance_threshold=1.0,
                 min_score=0.0,
                 max_range=0.0):

        # Initialize
        self.save_loc = save_loc
        self.distance_threshold_sq = distance_threshold**2
        self.score_threshold = min_score
        self.max_range = max_range
        self.classes = classes
        self.total_N_pos = 0
        self.results_dict = {}
        for single_class in classes:
            class_dict = {}
            class_dict['class'] = single_class
            class_dict['T_p'] = np.empty((0, 8))
            class_dict['gt'] = np.empty((0, 7))
            class_dict['total_N_pos'] = 0
            class_dict['result'] = np.empty((0, 2))
            class_dict['precision'] = []
            class_dict['recall'] = []
            self.results_dict[single_class] = class_dict
        # Format
        if pred_label_path[-1] is not "/":
            pred_label_path += "/"
        if gt_label_path[-1] is not "/":
            gt_label_path += "/"
        # Run
        self.time = time.time()
        self.evaluate(pred_label_path, gt_label_path, label_format)

    def evaluate(self, pred_path, gt_path, label_format):
        pred_file_list = glob.glob(pred_path + "*")
        pred_file_list.sort()
        gt_file_list = glob.glob(gt_path + "*")
        gt_file_list.sort()
        num_examples = len(pred_file_list)
        print("Starting evaluation for {} file predictions".format(num_examples))
        print("--------------------------------------------")

        ## Check missing files
        print("Confirmation prediction ground truth file pairs.")
        for pred_fn in pred_file_list:
            if (gt_path + os.path.basename(pred_fn)) not in gt_file_list:
                print("Error loading labels: gt label for pred label {} was not found.".format(
                    os.path.basename(pred_fn)))
                sys.exit(1)

        ## Evaluate matches
        print("Evaluation examples")
        file_parsing = LabelParser(label_format)
        for i, pred_fn in enumerate(pred_file_list):
            # print("\r", i+1, "/", num_examples, end="")
            gt_fn = gt_path + os.path.basename(pred_fn)
            predictions = file_parsing.parse_label(pred_fn, prediction=True)
            ground_truth = file_parsing.parse_label(gt_fn, prediction=False)
        # Filter range
            if self.max_range > 0:
                predictions, ground_truth = self.filter_by_range(predictions, ground_truth, range=self.max_range)
            self.eval_pair(predictions.astype(np.float), ground_truth.astype(np.float))
        print("\nDone!")
        print("----------------------------------")
        ## Calculate
        for single_class in self.classes:
            class_dict = self.results_dict[single_class]
            print("Calculating metrics for {} class".format(single_class))
            print("----------------------------------")
            print("Number of ground truth labels: ", class_dict['total_N_pos'])
            print("Number of detections:  ", class_dict['result'].shape[0])
            print("Number of true positives:  ", np.sum(class_dict['result'][:, 0] == 1))
            print("Number of false positives:  ", np.sum(class_dict['result'][:, 0] == 0))
            if class_dict['total_N_pos'] == 0:
                print("No detections for this class!")
                print(" ")
                continue
            ## AP
            self.compute_ap_curve(class_dict)
            mean_ap = self.compute_mean_ap(class_dict['precision'], class_dict['recall'])
            print('Mean AP: %.3f ' % mean_ap)
            f1 = self.compute_f1_score(class_dict['precision'], class_dict['recall'])
            print('F1 Score: %.3f ' % f1)
            print(' ')
            ## Positive Thresholds
            # ATE 2D
            ate2d = self.compute_ate2d(class_dict['T_p'], class_dict['gt'])
            print('Average 2D Translation Error [m]:  %.4f ' % ate2d)
            # ATE 3D
            ate3d = self.compute_ate3d(class_dict['T_p'], class_dict['gt'])
            print('Average 3D Translation Error [m]:  %.4f ' % ate3d)
            # ASE
            ase = self.compute_ase(class_dict['T_p'], class_dict['gt'])
            print('Average Scale Error:  %.4f ' % ase)
            # AOE
            aoe = self.compute_aoe(class_dict['T_p'], class_dict['gt'])
            print('Average Orientation Error [rad]:  %.4f ' % aoe)
            print(" ")
        self.time = float(time.time() - self.time)
        print("Total evaluation time: %.5f " % self.time)

    def compute_ap_curve(self, class_dict):
        t_pos = 0
        class_dict['precision'] = np.ones(class_dict['result'].shape[0]+2)
        class_dict['recall'] = np.zeros(class_dict['result'].shape[0]+2)
        sorted_detections = class_dict['result'][(-class_dict['result'][:, 1]).argsort(), :]
        print(sorted_detections.shape)
        for i, (result_bool, result_score) in enumerate(sorted_detections):
            if result_bool == 1:
                t_pos += 1
            class_dict['precision'][i+1] = t_pos / (i + 1)
            class_dict['recall'][i+1] = t_pos / class_dict['total_N_pos']
        class_dict['precision'][i+2] = 0
        class_dict['recall'][i+2] = class_dict['recall'][i+1]

        ## Plot
        plt.figure()
        plt.plot(class_dict['recall'], class_dict['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve for {} Class'.format(class_dict['class']))
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.savefig(os.path.join(self.save_loc,class_dict['class'] + "_pr_curve.png"))

    def compute_f1_score(self, precision, recall):
        p, r = precision[(precision+recall) > 0], recall[(precision+recall) > 0]
        f1_scores = 2 * p * r / (p + r)
        return np.max(f1_scores)

    def compute_mean_ap(self, precision, recall, precision_threshold=0.0, recall_threshold=0.0):
        mean_ap = 0
        threshold_mask = np.logical_and(precision > precision_threshold,
                                        recall > recall_threshold)
        # calculate mean AP
        precision = precision[threshold_mask]
        recall = recall[threshold_mask]
        recall_diff = np.diff(recall)
        precision_diff = np.diff(precision)
        # Square area under curve based on i+1 precision, then linear difference in precision
        mean_ap = np.sum(precision[1:]*recall_diff + recall_diff*precision_diff/2)
        # We need to divide by (1-recall_threshold) to make the max possible mAP = 1. In practice threshold by the first
        # considered recall value (threshold = 0.1 -> first considered value may be = 0.1123)
        mean_ap = mean_ap/(1-recall[0])
        return mean_ap

    def compute_ate2d(self, predictions, ground_truth):
        # euclidean distance 3d
        mean_ate2d = np.mean(np.sqrt((predictions[:, 0] - ground_truth[:, 0])**2 +
                                     (predictions[:, 1] - ground_truth[:, 1])**2))
        return mean_ate2d

    def compute_ate3d(self, predictions, ground_truth):
        # euclidean distance 2d
        mean_ate3d = np.mean(np.sqrt((predictions[:, 0] - ground_truth[:, 0]) ** 2 +
                                     (predictions[:, 1] - ground_truth[:, 1]) ** 2 +
                                     (predictions[:, 2] - ground_truth[:, 2]) ** 2))
        return mean_ate3d

    def compute_ase(self, predictions, ground_truth):
        # simplified iou where boxes are centered and aligned with eachother
        pred_vol = predictions[:, 3]*predictions[:, 4]*predictions[:, 5]
        gt_vol = ground_truth[:, 3]*ground_truth[:, 4]*ground_truth[:, 5]
        iou3d = np.mean(1 - np.minimum(pred_vol, gt_vol)/np.maximum(pred_vol, gt_vol))
        return iou3d

    def compute_aoe(self, predictions, ground_truth):
        err = ground_truth[:,6] - predictions[:,6]
        aoe = np.mean(np.abs((err + np.pi) % (2*np.pi) - np.pi))
        return aoe

    def eval_pair(self, pred_label, gt_label):
        ## Check
        assert pred_label.shape[1] == 9
        assert gt_label.shape[1] == 8

        ## Threshold score
        if pred_label.shape[0] > 0:
            pred_label = pred_label[pred_label[:, 8].astype(np.float) > self.score_threshold, :]

        for class_idx,single_class in enumerate(self.classes):
            # get all pred labels, order by score
            class_pred_label = pred_label[pred_label[:, 0].astype(int) == class_idx, 1:]
            score = class_pred_label[:, 7].astype(np.float)
            class_pred_label = class_pred_label[(-score).argsort(), :].astype(np.float) # sort decreasing

            # add gt label length to total_N_pos
            class_gt_label = gt_label[gt_label[:, 0].astype(int) == class_idx, 1:].astype(np.float)
            self.results_dict[single_class]['total_N_pos'] += class_gt_label.shape[0]

            # match pairs
            pred_array, gt_array, result_score_pair = self.match_pairs(class_pred_label, class_gt_label)

            # add to existing results
            self.results_dict[single_class]['T_p'] = np.vstack((self.results_dict[single_class]['T_p'], pred_array))
            self.results_dict[single_class]['gt'] = np.vstack((self.results_dict[single_class]['gt'], gt_array))
            self.results_dict[single_class]['result'] = np.vstack((self.results_dict[single_class]['result'],
                                                                   result_score_pair))

    def match_pairs(self, pred_label, gt_label):
        true_preds = np.empty((0, 8))
        corresponding_gt = np.empty((0, 7))
        result_score = np.empty((0, 2))
        # Initialize matching loop
        match_incomplete = True
        while match_incomplete and gt_label.shape[0] > 0:
            match_incomplete = False
            for gt_idx, single_gt_label in enumerate(gt_label):
                # Check is any prediction is in range
                distance_sq_array = (single_gt_label[0] - pred_label[:, 0])**2 + (single_gt_label[1] - pred_label[:, 1])**2
                # If there is a prediction in range, pick closest
                if np.any(distance_sq_array < self.distance_threshold_sq):
                    min_idx = np.argmin(distance_sq_array)
                    # Store true prediction
                    true_preds = np.vstack((true_preds, pred_label[min_idx, :].reshape(-1, 1).T))
                    corresponding_gt = np.vstack((corresponding_gt, gt_label[gt_idx]))

                    # Store score for mAP
                    result_score = np.vstack((result_score, np.array([[1, pred_label[min_idx, 7]]])))

                    # Remove prediction and gt then reset loop
                    pred_label = np.delete(pred_label, obj=min_idx, axis=0)
                    gt_label = np.delete(gt_label, obj=gt_idx, axis=0)
                    match_incomplete = True
                    break

        # If there were any false detections, add them.
        if pred_label.shape[0] > 0:
            false_positives = np.zeros((pred_label.shape[0], 2))
            false_positives[:, 1] = pred_label[:, 7]
            result_score = np.vstack((result_score, false_positives))
        return true_preds, corresponding_gt, result_score

    def filter_by_range(self, pred_label, gt_label, range=0):
        pred_dist = np.linalg.norm(pred_label[:, 1:4].astype(np.float32), axis=1) < range
        gt_dist = np.linalg.norm(gt_label[:, 1:4].astype(np.float32), axis=1) < range
        return pred_label[pred_dist, :], gt_label[gt_dist, :]

