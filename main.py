import time
import pickle
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


from ctrl import *
from utils import *
from dataset import *

from utils_FCOS import cal_iou, get_bboxes

from config import CONFIG
cfg = CONFIG()

from config_FCOS import Config

class FCOSLoss(nn.Module):

	def __init__(self,
				 num_classes,
				 strides=(8, 16, 32, 64, 128),
				 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
								 (512, np.inf)),
				 focal_alpha=0.25,
				 focal_gamma=2,
				 iou_eps=1e-6,
				 gt_bboxes_ignore=None):

		""" FCOS loss.
        Args:
            num_classes (int): num of classes
            focal_alpha: alpha in focal loss
            focal_gamma: gamma in focal loss
            iou_eps: eps in IoU loss
        Forward:
            cls_score (list): `(5, bs, 1, seq_i)`
            bbox_pred (list): `(5, bs, 2, seq_i)`
            centerness (list): `(5, bs, 1, seq_i)`
            targets (list): contain gt corner boxes and labels of each sample
                gt_bboxes (list, len BS): `(Object Num X 2)` Object Num=1

        Return:
            loss (dict)
                loss_cls (float)
                loss_bbox (float)
                loss_centerness (float)

        """

		super().__init__()
		self.strides = strides
		self.regress_ranges = regress_ranges
		self.num_classes = num_classes
		self.cls_out_channels = num_classes - 1
		self.alpha = focal_alpha
		self.gamma = focal_gamma
		self.eps = iou_eps

	def forward(self,
				cls_scores,
				bbox_preds,
				centernesses,
				gt_bboxes):

		assert len(cls_scores) == len(bbox_preds) == len(centernesses)
		featmap_sizes = [featmap.size()[-1] for featmap in cls_scores]
		# 映射各级pred点回原图
		all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
										   bbox_preds[0].device)
		# 将各级在范围内的点留下并根据boxes生成是否为相关片段的label
		labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes)

		num_imgs = cls_scores[0].size(0)
		# flatten cls_scores, bbox_preds and centerness

		flatten_cls_scores = [
			cls_score.permute(0, 2, 1).reshape(-1, self.cls_out_channels)
			for cls_score in cls_scores
		]
		flatten_bbox_preds = [
			bbox_pred.permute(0, 2, 1).reshape(-1, 2)
			for bbox_pred in bbox_preds
		]
		flatten_centerness = [
			centerness.permute(0, 2, 1).reshape(-1)
			for centerness in centernesses
		]
		flatten_cls_scores = torch.cat(flatten_cls_scores)
		flatten_bbox_preds = torch.cat(flatten_bbox_preds)
		flatten_centerness = torch.cat(flatten_centerness)
		flatten_labels = torch.cat(labels)
		flatten_bbox_targets = torch.cat(bbox_targets)
		# repeat points to align with bbox_preds
		flatten_points = torch.cat(
			[points.repeat(num_imgs, 1) for points in all_level_points])
		# find pos index
		pos_inds = flatten_labels.nonzero().reshape(-1)
		num_pos = len(pos_inds)

		# 分类损失 FocalLoss
		loss_cls = self.sigmoid_focal_loss(
			flatten_cls_scores, flatten_labels)
		pos_bbox_preds = flatten_bbox_preds[pos_inds]
		pos_centerness = flatten_centerness[pos_inds]

		if num_pos > 0:
			pos_bbox_targets = flatten_bbox_targets[pos_inds]
			pos_centerness_targets = self.centerness_target(pos_bbox_targets)
			pos_points = flatten_points[pos_inds]
			pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
			pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
			# bbox regression IoU Loss
			loss_bbox = self.iou_loss(
				pos_decoded_bbox_preds,
				pos_decoded_target_preds)
			# centerness CrossEntrophyLoss
			loss_centerness = self.sigmoid_crossentropy_loss(pos_centerness,
															 pos_centerness_targets)
		else:
			loss_bbox = pos_bbox_preds.sum()
			loss_centerness = pos_centerness.sum()

		return dict(
			loss_cls=loss_cls,
			loss_bbox=loss_bbox,
			loss_centerness=loss_centerness,
			num_pos=num_pos)

	def get_points_single(self, featmap_size, stride, dtype, device):
		w = featmap_size
		l_range = torch.arange(
			0, w * stride, stride, dtype=dtype, device=device)

		points_loc = l_range.reshape(-1)
		points = points_loc + stride // 2
		return points.unsqueeze(-1)

	def get_points(self, featmap_sizes, dtype, device):
		"""Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
		mlvl_points = []
		for i in range(len(featmap_sizes)):
			mlvl_points.append(
				self.get_points_single(featmap_sizes[i], self.strides[i],
									   dtype, device))
		return mlvl_points

	def fcos_target(self, points, gt_bboxes_list):
		assert len(points) == len(self.regress_ranges)
		num_levels = len(points)
		# expand regress ranges to align with points
		expanded_regress_ranges = [
			points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
				points[i].expand(points[i].shape[0], 2))
			for i in range(num_levels)
		]
		# concat all levels points and regress ranges
		concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
		concat_points = torch.cat(points, dim=0)
		# get labels and bbox_targets of each image
		labels_list, bbox_targets_list = multi_apply(
			self.fcos_target_single,
			gt_bboxes_list,
			points=concat_points,
			regress_ranges=concat_regress_ranges)

		# split to per img, per level
		num_points = [center.size(0) for center in points]
		labels_list = [labels.split(num_points, 0) for labels in labels_list]
		bbox_targets_list = [
			bbox_targets.split(num_points, 0)
			for bbox_targets in bbox_targets_list
		]

		# concat per level image
		concat_lvl_labels = []
		concat_lvl_bbox_targets = []
		for i in range(num_levels):
			concat_lvl_labels.append(
				torch.cat([labels[i] for labels in labels_list]))
			concat_lvl_bbox_targets.append(
				torch.cat(
					[bbox_targets[i] for bbox_targets in bbox_targets_list]))
		return concat_lvl_labels, concat_lvl_bbox_targets

	def fcos_target_single(self, gt_bboxes, points, regress_ranges):
		num_points = points.size(0)
		num_gts = gt_bboxes.size(0)
		gt_labels = torch.ones(num_gts).type_as(gt_bboxes)
		if num_gts == 0:
			return gt_bboxes.new_zeros((num_points, 2))

		# end - start
		areas = gt_bboxes[:, 1] - gt_bboxes[:, 0] + 1
		areas = areas[None].repeat(num_points, 1)
		regress_ranges = regress_ranges[:, None, :].expand(
			num_points, num_gts, 2)
		gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 2)

		loc = points[:, None].expand(num_points, num_gts, 1)

		front = loc[..., 0] - gt_bboxes[..., 0]
		back = gt_bboxes[..., 1] - loc[..., 0]

		bbox_targets = torch.stack((front, back), -1)

		# condition1: inside a gt bbox
		inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

		# condition2: limit the regression range for each location
		max_regress_distance = bbox_targets.max(-1)[0]
		inside_regress_range = (
									   max_regress_distance >= regress_ranges[..., 0]) & (
									   max_regress_distance <= regress_ranges[..., 1])

		# if there are still more than one objects for a location,
		# we choose the one with minimal area
		areas[inside_gt_bbox_mask == 0] = INF
		areas[inside_regress_range == 0] = INF
		min_area, min_area_inds = areas.min(dim=1)

		# relevant clip or not
		labels = gt_labels[min_area_inds]
		labels[min_area == INF] = 0
		bbox_targets = bbox_targets[range(num_points), min_area_inds]

		return labels, bbox_targets

	def centerness_target(self, pos_bbox_targets):
		# only calculate pos centerness targets, otherwise there may be nan 上下左右的偏移中心的值运算，反应距离center的归一化距离
		# min(front, back) / max(front, back)
		centerness_targets = pos_bbox_targets.min(dim=-1)[0] / pos_bbox_targets.max(dim=-1)[0]
		return torch.sqrt(centerness_targets)

	def sigmoid_focal_loss(self, confidence, label, reduction='sum'):
		"""Focal Loss, normalized by the number of positive anchor.
        Args:
            confidence (tensor): `(batch_size, num_priors, num_classes-1)`
            label (tensor): `(batch_size, num_priors)`
        """
		assert reduction in ['mean', 'sum']
		pred_sigmoid = confidence.sigmoid()
		# one_hot码左移一位

		label = F.one_hot(label.long(), num_classes=self.num_classes)[:, 1:].type_as(confidence)
		pt = (1 - pred_sigmoid) * label + pred_sigmoid * (1 - label)  # 1 - pt in paper
		focal_weight = (self.alpha * label + (1 - self.alpha) * (1 - label)) * \
					   pt.pow(self.gamma)
		loss = F.binary_cross_entropy_with_logits(
			confidence, label, reduction='none') * focal_weight
		if reduction == 'mean':
			return loss.mean()
		elif reduction == 'sum':
			return loss.sum()
		else:
			raise ValueError('reduction can only be `mean` or `sum`')

	def sigmoid_crossentropy_loss(self, confidence, label, reduction='sum'):
		"""CrossEntropy Loss.
        Args:
            confidence (tensor): `(batch_size, num_priors, num_classes)`
            label (tensor): `(batch_size, num_priors)`
        """
		loss = F.binary_cross_entropy_with_logits(
			confidence, label.float(), reduction=reduction)
		if reduction == 'mean':
			return loss.mean()
		elif reduction == 'sum':
			return loss.sum()
		else:
			raise ValueError('reduction can only be `mean` or `sum`')

	def iou_loss(self, confidence, label, reduction='sum', weight=1.0):
		"""IoU loss, Computing the IoU loss between a set of predicted bboxes and target bboxes.
        The loss is calculated as negative log of IoU.
        Args:
            pred (Tensor): Predicted bboxes of format (s, e),
                shape (n, 2).
            target (Tensor): Corresponding gt bboxes, shape (n, 2).
            eps (float): Eps to avoid log(0).
        """
		rows = confidence.size(0)
		cols = label.size(0)
		assert rows == cols
		if rows * cols == 0:
			return confidence.new(rows, 1)
		lt = torch.max(confidence[:, 0], label[:, 0])  # [rows, ]
		rb = torch.min(confidence[:, 1], label[:, 1])  # [rows, ]
		wh = (rb - lt + 1).clamp(min=0)  # [rows, ]
		overlap = wh
		area1 = confidence[:, 1] - confidence[:, 0] + 1
		area2 = label[:, 1] - label[:, 0] + 1
		ious = overlap / (area1 + area2 - overlap)
		safe_ious = ious.clamp(min=self.eps)
		loss = -safe_ious.log() * weight
		if reduction == 'mean':
			return loss.mean()
		elif reduction == 'sum':
			return loss.sum()
		else:
			raise ValueError('reduction can only be `mean` or `sum`')


class Processor():
	def __init__(self):
		self.load_data()
		self.load_model()
		self.load_optimizer()	

	def load_data(self):
		self.data_loader = dict()
		if cfg.phase == 'train':
			if cfg.dataset == 'Anet':
				current_dataset = AnetDataset(cfg.train_feature_dir,
								  cfg.train_csv_path,
								  cfg.visual_dim,
								  cfg.sentence_embed_dim,
								  cfg.IoU,
								  cfg.nIoU,
								  cfg.context_num,
                 				  cfg.context_size,
								)
			else:
				current_dataset = TrainDataset(cfg.train_feature_dir,
											   cfg.train_csv_path,
											   cfg.visual_dim,
											   cfg.sentence_embed_dim,
											   cfg.IoU,
											   cfg.nIoU,
											   cfg.context_num,
											   cfg.context_size,
											   )
			self.data_loader['train'] = torch.utils.data.DataLoader(
				dataset=current_dataset,
				batch_size=cfg.batch_size,
				shuffle=True,
				num_workers=cfg.num_worker)
		if cfg.dataset == 'Anet':
			self.testDataset = TestingAnetDataset(cfg.test_feature_dir, cfg.test_csv_path, cfg.test_batch_size)
		else:
			self.testDataset = TestingDataSet(cfg.test_feature_dir, cfg.test_csv_path, cfg.test_batch_size)

		# self.data_loader['test'] = torch.utils.data.DataLoader(
		# 	dataset=TestDataset(cfg.train_feature_dir, 
		# 						  cfg.train_csv_path,
		# 						  cfg.visual_dim,
		# 						  cfg.sentence_embed_dim,
		# 						  cfg.IoU,
		# 						  cfg.nIoU,
		# 						  cfg.context_num,
  #                				  cfg.context_size,
		# 						),
		# 	batch_size=cfg.test_batch_size,
		# 	shuffle=False,
		# 	num_workers=cfg.num_worker)

	def load_model(self):
		torch.manual_seed(cfg.seed)
		if torch.cuda.is_available():
			if type(cfg.device) is list and len(cfg.device) > 1:
				torch.cuda.manual_seed_all(cfg.seed)
			else:
				torch.cuda.manual_seed(cfg.seed)

			self.output_device = cfg.device[0] if type(cfg.device) is list else cfg.device

		self.model = CTRL(cfg.visual_dim, cfg.sentence_embed_dim, cfg.semantic_dim, cfg.middle_layer_dim)
		self.loss = CTRL_loss(cfg.lambda_reg)
		if torch.cuda.is_available():
			self.model.cuda(self.output_device)
			self.loss.cuda(self.output_device)

		if torch.cuda.is_available() and type(cfg.device) is list and len(cfg.device) > 1:				
			self.model = nn.DataParallel(self.model, device_ids=cfg.device, output_device=self.output_device)

	def load_optimizer(self):
		if cfg.optimizer == 'Adam':
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=cfg.vs_lr,
				weight_decay=cfg.weight_decay,
				)
		else:
			raise ValueError()


	def train(self):
		losses = []
		for epoch in range(cfg.max_epoch):
			for step, data_torch in enumerate(self.data_loader['train']):
				#self.evalAnet(step + 1, cfg.test_output_path)
				self.model.train()
				self.record_time()
				# forward
				output = self.model(data_torch['vis'].to("cuda"), data_torch['sent'].to("cuda"))
				loss = self.loss(output.to("cuda"), data_torch['offset'].to("cuda"))

				# backward
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				losses.append(loss.item())

				duration = self.split_time()

				if (step+1) % 5 == 0 or step == 0:
					self.print_log('Epoch %d, Step %d: loss = %.3f (%.3f sec)' % (epoch+1, step+1, losses[-1], duration))

				if (step+1) % 2000 == -1:
					self.print_log('Testing:')
					if cfg.dataset == 'Anet':
						self.evalAnet(step + 1, cfg.test_output_path)
					else:
						movie_length_info = pickle.load(open(cfg.movie_length_info_path, 'rb'), encoding='iso-8859-1')
						self.eval(movie_length_info, step + 1, cfg.test_output_path)

	def evalAnet(self, step, test_output_path):
		self.model.eval()
		with torch.no_grad():
			val_loader = self.testDataset
			length = len(val_loader.samples)
			results = {}
			video_start_id = 0
			for idx in range(length):
				sample = val_loader.samples[idx]
				val_data = sample[0]
				val_sent = sample[2]
				gt_box = sample[3]
				video_name = sample[4]
				sentence = sample[5]
				window_start = sample[6]

				val_data = torch.autograd.Variable(val_data.unsqueeze(0))
				val_sent = torch.autograd.Variable(torch.from_numpy(val_sent).unsqueeze(0))

				output = self.model(val_data, val_sent)
				output_np = output.detach().cpu().numpy()[0][0]

				# 开始预测
				#cls_score(list): `(5, bs, 1, seq_i)`
				#bbox_pred(list): `(5, bs, 2, seq_i)`
				#centerness(list): `(5, bs, 1, seq_i)`
				# start = float(visual_clip_name.split("_")[1])
				# end = float(visual_clip_name.split("_")[2].split("_")[0])

				reg_clip_length = (10 ** output_np[2])
				reg_mid_point = output_np[1]  # * movie_length

				reg_end = output_np[2]
				reg_start = output_np[1]

				c = torch.from_numpy(np.expand_dims(np.expand_dims(np.numarray(1, 2), axis=0), axis=2))
				cls_score = []
				cls_score.append(c)
				b = torch.from_numpy(np.expand_dims(np.expand_dims([reg_start, reg_end], axis=0), axis=2))
				bbox_pred = []
				bbox_pred.append((b))
				c = torch.from_numpy(np.expand_dims(np.expand_dims(np.numarray(1,2), axis=0), axis=2))
				centerness = []
				centerness.append(c)
				video_num = 1
				video_start_id = 0
				criterion = FCOSLoss(3)
				mfconfig = Config()
				result_list, video_start_id = get_bboxes(mfconfig,
														 criterion,
														 cls_score,
														 bbox_pred,
														 centerness,
														 video_num,
														 video_start_id)
				result_list = torch.cat(result_list)
				mask = (result_list[:, 0] == 1)
				a_scores = result_list[mask][:, 2].cpu().detach().numpy()
				a_min = result_list[mask][:, 3].cpu().detach().numpy()
				a_max = result_list[mask][:, 4].cpu().detach().numpy()

				corrected_min = np.maximum(a_min, 0.) + window_start
				corrected_max = np.minimum(a_max, config.window_size) + window_start

				if video_name not in results.keys():
					results[video_name] = {}
				if sentence not in results[video_name].keys():
					results[video_name][sentence] = [np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), gt_box]

				results[video_name][sentence] = [np.hstack([results[video_name][sentence][0], corrected_min]),
												 np.hstack([results[video_name][sentence][1], corrected_max]),
												 np.hstack([results[video_name][sentence][2], a_scores]),
												 results[video_name][sentence][3]]

		rank1_list = []
		rank5_list = []
		rank10_list = []
		rank1_5 = 0
		rank1_7 = 0
		rank5_5 = 0
		rank5_7 = 0
		num1 = 0
		num5 = 0
		for video_name in results.keys():
			for sentence in results[video_name].keys():
				corrected_min = results[video_name][sentence][0]
				corrected_max = results[video_name][sentence][1]
				a_scores = results[video_name][sentence][2]
				gt_box = results[video_name][sentence][3]
				# print(video_name + ': ' + sentence + '  start: ' + str(gt_box[0]) + ' --   end: ' + str(gt_box[1]))
				iou = cal_iou(config, corrected_min, corrected_max, a_scores, gt_box)
				if len(iou) == 0:
					continue
				rank1 = iou[0]
				rank1_5 += int(rank1 >= 0.5)
				rank1_7 += int(rank1 >= 0.7)
				rank1_list.append(rank1)
				num1 += 1
				rank5 = iou[:5]
				rank5_5 += int(max(rank5) >= 0.5)
				rank5_7 += int(max(rank5) >= 0.7)
				rank5_list.append(max(rank5))
				num5 += 1
				rank10 = iou[:10]
				rank10_list.append(max(rank10))

		print("rank@1-0.5: %.4f\trank@1-0.7: %.4f\trank@5-0.5: %.4f\trank@5-0.7: %.4f\n" % (
		100 * float(rank1_5) / num1, 100 * float(rank1_7) / num1, 100 * float(rank5_5) / num5,
		100 * float(rank5_7) / num5))
		print("rank1: %.4f\trank5: %.4f\trank10: %.4f\n" % (
		100 * np.mean(rank1_list), 100 * np.mean(rank5_list), 100 * np.mean(rank10_list)))

		return (100 * float(rank1_5) / num1 + 100 * float(rank1_7) / num1)


	def eval(self, movie_length_info, step, test_output_path):
		self.model.eval()		
		IoU_thresh = [0.1, 0.2, 0.3, 0.4, 0.5]
		all_correct_num_10 = [0.0] * 5
		all_correct_num_5  = [0.0] * 5
		all_correct_num_1  = [0.0] * 5
		all_retrievd = 0.0

		for movie_name in self.testDataset.movie_names:
			movie_length=movie_length_info[movie_name.split(".")[0]]
			self.print_log("Test movie: " + movie_name + "....loading movie data")
			movie_clip_featmaps, movie_clip_sentences = self.testDataset.load_movie_slidingclip(movie_name, 16)
			self.print_log("sentences: "+ str(len(movie_clip_sentences)))
			self.print_log("clips: "+ str(len(movie_clip_featmaps)))
			sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
			sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
			for k in range(len(movie_clip_sentences)):
				sent_vec = movie_clip_sentences[k][1]
				sent_vec = np.reshape(sent_vec,[1,sent_vec.shape[0]])
				for t in range(len(movie_clip_featmaps)):
					featmap = movie_clip_featmaps[t][1]
					visual_clip_name = movie_clip_featmaps[t][0]
					start = float(visual_clip_name.split("_")[1])
					end = float(visual_clip_name.split("_")[2].split("_")[0])
					featmap = np.reshape(featmap, [1, featmap.shape[0]])

					output = self.model(torch.from_numpy(featmap), torch.from_numpy(sent_vec))
					output_np = output.detach().numpy()[0][0]

					sentence_image_mat[k,t] = output_np[0]
					reg_clip_length = (end - start) * (10 ** output_np[2])
					reg_mid_point = (start + end) / 2.0 + movie_length * output_np[1]
					reg_end = end + output_np[2]
					reg_start = start + output_np[1]

					sentence_image_reg_mat[k, t, 0] = reg_start
					sentence_image_reg_mat[k, t, 1] = reg_end

			iclips = [b[0] for b in movie_clip_featmaps]
			sclips = [b[0] for b in movie_clip_sentences]

			# calculate Recall@m, IoU=n
			for k in range(len(IoU_thresh)):
				IoU=IoU_thresh[k]
				correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
				correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
				correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
				self.print_log(movie_name+" IoU="+str(IoU)+", R@10: "+str(correct_num_10/len(sclips))+"; IoU="+str(IoU)+", R@5: "+str(correct_num_5/len(sclips))+"; IoU="+str(IoU)+", R@1: "+str(correct_num_1/len(sclips)))
				all_correct_num_10[k]+=correct_num_10
				all_correct_num_5[k]+=correct_num_5
				all_correct_num_1[k]+=correct_num_1
			all_retrievd += len(sclips)
			
		for k in range(len(IoU_thresh)):
			self.print_log("IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd))
			with open(test_output_path, "w") as f:
				f.write("Step "+str(iter_step)+": IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd)+"\n")


	def record_time(self):
		self.cur_time = time.time()
		return self.cur_time

	def split_time(self):
		split_time = time.time() - self.cur_time
		self.record_time()
		return split_time

	def print_log(self, line, print_time=True):
		if print_time:
			localtime = time.asctime(time.localtime(time.time()))
			line = "[ " + localtime + ' ] ' + line
		print(line)
		if cfg.save_log:
			with open(cfg.log_dir, 'a') as f:
				print(line, file=f)

	def print_time(self):
		localtime = time.asctime(time.localtime(time.time()))
		self.print_log("Local current time :  " + localtime)


if __name__ == '__main__':
	processor = Processor()
	processor.train()	


