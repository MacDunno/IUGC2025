import torch
import numpy as np
import pandas as pd

class PseudoLabelGenerator:
    def __init__(self, confidence_threshold=0.8, confidence_all_threshold=0.95):
        self.confidence_threshold = confidence_threshold
        self.confidence_all_threshold = confidence_all_threshold
    
    def calculate_heatmap_confidence(self, heatmaps):
        confidences = []
        confidences_all = []
        for heatmap in heatmaps:
            max_val = np.max(heatmap)
            if max_val == 0:
                confidences.append(0.0)
                continue
            peak_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y, x = peak_pos
            radius = 5
            y_min = max(0, y - radius)
            y_max = min(heatmap.shape[0], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(heatmap.shape[1], x + radius + 1)
            neighborhood = heatmap[y_min:y_max, x_min:x_max]
            neighborhood_mean = np.mean(neighborhood)
            all_mean = np.mean(heatmap)
            if neighborhood_mean == 0:
                sharpness = 0.0
            else:
                sharpness = max_val / neighborhood_mean
                confidence = 1 - 1 / sharpness
            if all_mean == 0:
                sharpness_all = 0.0
            else:
                sharpness_all = max_val / all_mean
                confidence_all = 1 - 1 / sharpness_all
            confidences.append(confidence)
            confidences_all.append(confidence_all)
        
        return np.array(confidences), np.array(confidences_all)
    
    def check_anatomical_constraints(self, ps1, ps2, fh1):
        ps_distance = np.sqrt((ps1[0] - ps2[0])**2 + (ps1[1] - ps2[1])**2)
        ps_valid = 20 <= ps_distance <= 150
        fh_valid = fh1[1] > ps1[1] and fh1 > ps2[1]
        aop = self.calculate_aop(ps1, ps2, fh1)
        aop_valid = 50 <= aop <= 150 
        ret = ps_valid and fh_valid and aop_valid        
        return ret
    
    def calculate_aop(self, ps1, ps2, fh1):
        vec1 = np.array([ps1[0] - ps2[0], ps1[1] - ps2[1]])
        vec2 = np.array([fh1[0] - ps1[0], fh1[1] - ps1[1]])
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        return angle
    
    def extract_keypoints_from_heatmaps(self, heatmaps):
        keypoints = []
        for heatmap in heatmaps:
            flat_idx = np.argmax(heatmap)
            y, x = np.unravel_index(flat_idx, heatmap.shape)
            x_coord = x * 512 / heatmap.shape[1]
            y_coord = y * 512 / heatmap.shape[0]
            keypoints.append((x_coord, y_coord))
        return keypoints
    
    def generate_pseudo_labels(self, model, unlabeled_dataloader, device='cuda'):
        model.eval()
        pseudo_labels = []
        with torch.no_grad():
            for batch in unlabeled_dataloader:
                images = batch['image'].to(device)
                filenames = batch['filename']
                predicted_heatmaps = model(images)  # Shape: [B, 3, H, W]
                for i in range(len(filenames)):
                    heatmaps = predicted_heatmaps[i].cpu().numpy()  # [3, H, W]
                    filename = filenames[i]
                    confidences, confidences_all = self.calculate_heatmap_confidence(heatmaps)
                    mean_confidence = np.mean(confidences)
                    mean_confidence_all = np.mean(confidences_all)
                    keypoints = self.extract_keypoints_from_heatmaps(heatmaps)
                    ps1, ps2, fh1 = keypoints
                    geometric = self.check_anatomical_constraints(ps1, ps2, fh1)
                    if (mean_confidence >= self.confidence_threshold and mean_confidence_all >= self.confidence_all_threshold and geometric):
                        aop = self.calculate_aop(ps1, ps2, fh1)
                        pseudo_labels.append({
                            'Filename': filename,
                            'PS1': ps1,
                            'PS2': ps2,
                            'FH1': fh1,
                            'AoP': aop,
                            'confidence': mean_confidence,
                            'geometric': geometric
                        })
        return pseudo_labels
    
    def save_pseudo_labels(self, pseudo_labels, save_path):
        df = pd.DataFrame(pseudo_labels)
        df.to_csv(save_path, index=False)
        print(f"Saved {len(pseudo_labels)} pseudo-labels to {save_path}")
        return df