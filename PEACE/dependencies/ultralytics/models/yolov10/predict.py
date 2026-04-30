from ultralytics.models.yolo.detect import DetectionPredictor
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results
import pdb

from torchvision.ops import box_iou
from torchvision.ops import nms as torchvision_nms

def process_boxes(pred):
    # 将 pred 移动到 CPU
    pred = pred.cpu()

    # 获取所有类别的唯一值
    unique_classes = pred[:, -1].unique()

    # 创建一个列表用于存储保留的框
    kept_boxes = []

    for cls in unique_classes:
        if cls == 7 or cls == 4 or cls == 6 or cls == 5:
            # 类别4,6,7：不处理，直接跳过
            continue

        # 选取当前类别的框
        cls_mask = pred[:, -1] == cls
        cls_boxes = pred[cls_mask][:, :4]  # 当前类别的框

        # 如果没有当前类别的框，继续处理下一个类别
        if len(cls_boxes) == 0:
            continue
        
        # 检查是否有其他类别的框
        other_classes = unique_classes[unique_classes != cls]
        if len(other_classes) > 0:
            # 选取其他类别的框
            other_class_mask = torch.zeros(pred.shape[0], dtype=torch.bool)
            for oc in other_classes:
                other_class_mask |= pred[:, -1] == oc
            other_boxes = pred[other_class_mask][:, :4]

            # 计算当前类别框与其他类别框的IoU
            ious = box_iou(cls_boxes, other_boxes)

            # 找到每个当前类别框与其他类别框相交面积最小的IoU
            min_ious, _ = torch.min(ious, dim=1)

            # 找到相交面积最小的当前类别框
            min_iou_idx = torch.argmin(min_ious)
            best_box = cls_boxes[min_iou_idx]
        else:
            # 没有其他类别的框，选择当前类别中面积最大的框
            areas = (cls_boxes[:, 2] - cls_boxes[:, 0]) * (cls_boxes[:, 3] - cls_boxes[:, 1])
            max_area_idx = torch.argmax(areas)
            best_box = cls_boxes[max_area_idx]

        # 将最佳框添加到保留框列表中
        kept_boxes.append(torch.cat([best_box, torch.tensor([cls])]))

    # 将保留框列表转换为张量
    if kept_boxes:
        kept_boxes = torch.stack(kept_boxes)
    else:
        # 如果没有保留的框，返回原始的pred
        return pred

    # 生成一个布尔掩码以保留在 kept_boxes 中的框
    keep_mask = torch.zeros(pred.shape[0], dtype=torch.bool)
    for box in kept_boxes:
        cls = box[-1].item()
        box_coords = box[:4]
        mask = (pred[:, -1] == cls) & (torch.all(pred[:, :4] == box_coords, dim=1))
        keep_mask |= mask

    # 将类别7的框添加到 keep_mask 中
    cls7_mask = pred[:, -1] == 7
    keep_mask |= cls7_mask
    
    # 将类别4的框添加到 keep_mask 中
    cls7_mask = pred[:, -1] == 4
    keep_mask |= cls7_mask

    # 将类别6的框添加到 keep_mask 中
    cls7_mask = pred[:, -1] == 6
    keep_mask |= cls7_mask

    # 将类别5的框添加到 keep_mask 中
    cls7_mask = pred[:, -1] == 5
    keep_mask |= cls7_mask
    
    # 返回只包含保留框的 pred
    return pred[keep_mask]

class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.shape[-1] == 6:
            pass
        else:
            preds = preds.transpose(-1, -2)
            bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1]-4)
            bboxes = ops.xywh2xyxy(bboxes)
            preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        assert not len(preds) > 1
        # all_cls = torch.cat([pred[5] for pred in preds[0]], dim=0)
        # # Apply NMS only if all labels are 0
        # if torch.all(all_cls == 0):
        #     preds = torch.cat(preds, dim=0)
        #     keep = torchvision_nms(preds[:, :4], preds[:, 4], 0.8)
        #     preds = preds[keep]
        #     # split along dim=0, transformed into a list
        #     split_preds = torch.unbind(preds, dim=0)
        #     split_preds_list = list(split_preds)
        #     # top-down, then left-right
        #     preds = sorted(split_preds_list, key=lambda pred: (pred[1], pred[0]))
        # print('!!!!!!!!')
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            if torch.all(torch.abs(pred[:, -1]) > 1e-6):
                pred = process_boxes(pred)
                
            keep = torchvision_nms(pred[:, :4], pred[:, 4], 0.8)
            pred = pred[keep]
            
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
