import glob
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from label_parser import LabelParser
import pandas as pd
def iou_3d_witout_RT(box1, box2):
    '''
        box [x1,y1,z1,x2,y2,z2]   分别是两对角定点的坐标
    '''
    area1 = abs((box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2]))
    area2 = abs((box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2]))
    area_sum = area1 + area2

    # 计算重叠部分 设重叠box坐标为 [x1,y1,z1,x2,y2,z2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    z1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    z2 = min(box1[5], box2[5])
    if x1 >= x2 or y1 >= y2 or z1 >= z2:
        return 0
    else:
        inter_area = abs((x2 - x1) * (y2 - y1) * (z2 - z1))

    return inter_area / (area_sum - inter_area)
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
        results = []
        for single_class in self.classes:
            res = [0]*12
            class_dict = self.results_dict[single_class]

            res[0] = single_class
            res[1] = class_dict['total_N_pos']#Number of ground truth labels
            res[2] = class_dict['result'].shape[0]#Number of detections
            res[3] = np.sum(class_dict['result'][:, 0] == 1)#Number of true positives
            res[4] = np.sum(class_dict['result'][:, 0] == 0)#Number of false positives
            
            if class_dict['total_N_pos'] == 0:
                print("No detections for this class!")
                print(" ")
                continue
            
            ## AP
            self.compute_ap_curve(class_dict)
            mean_ap = self.compute_mean_ap(class_dict['precision'], class_dict['recall'])
            res[5] = mean_ap

            f1 = self.compute_f1_score(class_dict['precision'], class_dict['recall'])
            res[6] = f1

            ## Positive Thresholds
            # ATE 2D
            ate2d = self.compute_ate2d(class_dict['T_p'], class_dict['gt'])
            res[7] = ate2d#Average 2D Translation Error [m]
            # ATE 3D
            ate3d = self.compute_ate3d(class_dict['T_p'], class_dict['gt'])
            res[8] = ate3d#Average 3D Translation Error [m]
            # ASE
            ase = self.compute_ase(class_dict['T_p'], class_dict['gt'])
            res[9] = ase#Average Scale Error
            # AOE
            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if single_class == 'barrier' else 2 * np.pi
            aoe = self.compute_aoe(class_dict['T_p'], class_dict['gt'], period)
            res[10] = aoe#Average Orientation Error [rad]
            res[11] = aoe*180/np.pi#Average Orientation Error [deg]
            results.append(res)
        self.time = float(time.time() - self.time)
        print("Total evaluation time: %.5f " % self.time)
        df = pd.DataFrame(results, columns = ['class', 'gt', 'dt', 'tp', 'fp', 'map', 'f1', 'ATE', 'ATE_3D', 'ASE', 'AOE_rad', 'AOE_deg'])
        print(df)
        df.to_csv(os.path.join(self.save_loc,'stat.csv'), index = None, header=True) 

    def compute_ap_curve(self, class_dict):
        t_pos = 0
        class_dict['precision'] = np.ones(class_dict['result'].shape[0]+2)
        class_dict['recall'] = np.zeros(class_dict['result'].shape[0]+2)
        sorted_detections = class_dict['result'][(-class_dict['result'][:, 1]).argsort(), :]
        print(sorted_detections.shape)
        result_scores = []
        for i, (result_bool, result_score) in enumerate(sorted_detections):
            if result_bool == 1:
                t_pos += 1
            class_dict['precision'][i+1] = t_pos / (i + 1)
            class_dict['recall'][i+1] = t_pos / class_dict['total_N_pos']
            if i == 0:
                result_scores.append(result_score)
            result_scores.append(result_score)
        
        class_dict['precision'][i+2] = 0
        class_dict['recall'][i+2] = class_dict['recall'][i+1]
        result_scores.append(result_score)
        results = np.hstack((np.array(result_scores).reshape(-1, 1),class_dict['recall'].reshape(-1, 1),class_dict['precision'].reshape(-1, 1)))
        df = pd.DataFrame(results, columns = ['score', 'recall', 'precision'])
        df.to_csv(os.path.join(self.save_loc,class_dict['class'] + "_pr_curve.csv"), index = None, header=True)
        ## Plot
        plt.figure()
        plt.plot(class_dict['recall'], class_dict['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve for {} Class'.format(class_dict['class']))
        plt.xticks(np.arange(0, 1, 0.1))# plt.xlim([0, 1])
        plt.yticks(np.arange(0, 1.05, 0.1))# plt.ylim([0, 1.05])
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
        # # simplified iou where boxes are centered and aligned with eachother
        # pred_vol = predictions[:, 3]*predictions[:, 4]*predictions[:, 5]
        # gt_vol = ground_truth[:, 3]*ground_truth[:, 4]*ground_truth[:, 5]
        # iou3d = np.mean(1 - np.minimum(pred_vol, gt_vol)/np.maximum(pred_vol, gt_vol))

        # return iou3d
        se_list = []
        for ii in range(len(predictions)):
            obj_size = ground_truth[ii][3:6]
            obp_size = predictions[ii][3:6]
            obj_box = [0 - obj_size[0] / 2,
                        0 - obj_size[1] / 2,
                        0 - obj_size[2] / 2,
                        0 + obj_size[0] / 2,
                        0 + obj_size[1] / 2,
                        0 + obj_size[2] / 2,
                        ]
            obp_box = [0 - obp_size[0]/ 2,
                        0 - obp_size[1] / 2,
                        0 - obp_size[2] / 2,
                        0 + obp_size[0]/ 2,
                        0 + obp_size[1] / 2,
                        0 + obp_size[2] / 2
                        ]
            iou = iou_3d_witout_RT(obj_box, obp_box)
            se_list.append(1-iou)
        return sum(se_list)/len(se_list)
    
    def angle_diff(self, x: float, y: float, period: float) -> float:
        """
        Get the smallest angle difference between 2 angles: the angle from y to x.
        :param x: To angle.
        :param y: From angle.
        :param period: Periodicity in radians for assessing angle difference.
        :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
        """

        # calculate angle difference, modulo to [0, 2*pi]
        diff = (x - y + period / 2) % period - period / 2
        if diff > np.pi:
            diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

        return diff
    def compute_aoe(self, predictions, ground_truth, period = 2*np.pi):
        aoe = []
        for ii in range(predictions.shape[0]):
            diff = self.angle_diff(ground_truth[ii,6],predictions[ii,6], period)
            aoe.append(abs(diff))
        return np.mean(aoe)

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

