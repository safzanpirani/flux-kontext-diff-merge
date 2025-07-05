import torch
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import binary_dilation, binary_erosion
import comfy.model_management as model_management


class FluxKontextDiffMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "edited_image": ("IMAGE",),
                "sensitivity": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "detection_method": (["adaptive", "color_diff", "ssim", "combined"], {
                    "default": "adaptive"
                }),
                "blend_method": (["poisson", "alpha", "multiband", "gaussian"], {
                    "default": "poisson"
                }),
                "mask_blur": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "mask_expand": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 50,
                    "step": 1
                }),
                "edge_feather": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 50,
                    "step": 1
                }),
                "min_change_area": ("INT", {
                    "default": 250,
                    "min": 0,
                    "max": 5000,
                    "step": 10
                }),
                "global_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
            "optional": {
                "manual_mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("merged_image", "difference_mask", "preview_diff")
    FUNCTION = "merge_diff"
    CATEGORY = "image/postprocessing"
    
    def tensor_to_numpy(self, tensor):
        """Convert ComfyUI tensor to numpy array"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        numpy_image = tensor.cpu().numpy()
        if numpy_image.dtype != np.uint8:
            numpy_image = (numpy_image * 255).astype(np.uint8)
        return numpy_image
    
    def numpy_to_tensor(self, numpy_array):
        """Convert numpy array back to ComfyUI tensor"""
        if numpy_array.dtype == np.uint8:
            numpy_array = numpy_array.astype(np.float32) / 255.0
        if len(numpy_array.shape) == 3:
            numpy_array = np.expand_dims(numpy_array, axis=0)
        return torch.from_numpy(numpy_array)
    
    def adaptive_detection(self, original, edited, sensitivity=0.02, global_threshold=0.15):
        """Adaptive detection that's robust to global changes"""
        # Convert to LAB color space for better perceptual differences
        orig_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
        edit_lab = cv2.cvtColor(edited, cv2.COLOR_RGB2LAB)
        
        # Calculate differences in LAB space
        diff_l = np.abs(orig_lab[:,:,0].astype(np.float32) - edit_lab[:,:,0].astype(np.float32))
        diff_a = np.abs(orig_lab[:,:,1].astype(np.float32) - edit_lab[:,:,1].astype(np.float32))
        diff_b = np.abs(orig_lab[:,:,2].astype(np.float32) - edit_lab[:,:,2].astype(np.float32))
        
        # Weighted combination (L channel is most important)
        combined_diff = (diff_l * 0.5 + diff_a * 0.25 + diff_b * 0.25)
        
        # Calculate global average difference
        global_avg = np.mean(combined_diff)
        
        # Adaptive thresholding based on global changes
        if global_avg > global_threshold * 255:
            # High global changes - use relative threshold
            threshold = global_avg + (sensitivity * 255)
        else:
            # Low global changes - use absolute threshold
            threshold = sensitivity * 255
        
        # Create binary mask
        mask = (combined_diff > threshold).astype(np.uint8) * 255
        
        return mask
    
    def edge_aware_detection(self, original, edited, sensitivity=0.02):
        """Detection that focuses on structural/edge changes"""
        # Convert to grayscale
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        edit_gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
        
        # Calculate edge maps
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        edit_edges = cv2.Canny(edit_gray, 50, 150)
        
        # Calculate edge differences
        edge_diff = cv2.absdiff(orig_edges, edit_edges)
        
        # Dilate edge differences to create regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_diff = cv2.dilate(edge_diff, kernel, iterations=2)
        
        # Combine with intensity differences
        intensity_diff = cv2.absdiff(orig_gray, edit_gray)
        threshold = sensitivity * 255
        intensity_mask = (intensity_diff > threshold).astype(np.uint8) * 255
        
        # Combine edge and intensity differences
        combined_mask = cv2.bitwise_or(edge_diff, intensity_mask)
        
        return combined_mask
    
    def detect_changes(self, original, edited, sensitivity=0.02, method="adaptive", global_threshold=0.15):
        """Main change detection with multiple methods"""
        if method == "adaptive":
            mask = self.adaptive_detection(original, edited, sensitivity, global_threshold)
        elif method == "color_diff":
            mask = self.detect_color_changes(original, edited, sensitivity)
        elif method == "ssim":
            mask = self.detect_ssim_changes(original, edited, sensitivity)
        elif method == "combined":
            # Combine multiple methods
            mask1 = self.adaptive_detection(original, edited, sensitivity, global_threshold)
            mask2 = self.edge_aware_detection(original, edited, sensitivity)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = self.adaptive_detection(original, edited, sensitivity, global_threshold)
        
        return mask
    
    def detect_color_changes(self, original, edited, sensitivity=0.02):
        """Detect changes in color channels"""
        diff_r = np.abs(original[:,:,0].astype(np.float32) - edited[:,:,0].astype(np.float32))
        diff_g = np.abs(original[:,:,1].astype(np.float32) - edited[:,:,1].astype(np.float32))
        diff_b = np.abs(original[:,:,2].astype(np.float32) - edited[:,:,2].astype(np.float32))
        
        combined_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
        threshold = sensitivity * 255
        mask = (combined_diff > threshold).astype(np.uint8) * 255
        
        return mask
    
    def detect_ssim_changes(self, original, edited, sensitivity=0.02):
        """Detect changes using structural similarity"""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        edit_gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
        
        try:
            score, diff = ssim(orig_gray, edit_gray, full=True, data_range=255)
        except:
            diff = np.abs(orig_gray.astype(np.float32) - edit_gray.astype(np.float32)) / 255.0
        
        diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        mask = (diff_normalized > sensitivity).astype(np.uint8) * 255
        
        return mask
    
    def filter_small_changes(self, mask, min_area=250):
        """Remove small change areas that are likely noise"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.fillPoly(filtered_mask, [contour], 255)
        
        return filtered_mask
    
    def refine_mask(self, mask, expand_pixels=8, blur_amount=15, feather_amount=15):
        """Refine the difference mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Expand the mask if requested
        if expand_pixels > 0:
            expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (expand_pixels*2+1, expand_pixels*2+1))
            mask = cv2.dilate(mask, expand_kernel, iterations=1)
        
        # Gaussian blur for smooth edges
        if blur_amount > 0:
            mask = cv2.GaussianBlur(mask, (blur_amount*2+1, blur_amount*2+1), 0)
        
        # Additional feathering
        if feather_amount > 0:
            mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 
                                   feather_amount/3)
        
        return mask
    
    def poisson_blend(self, source, target, mask):
        """Poisson blending for seamless integration"""
        try:
            binary_mask = (mask > 127).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    center = (source.shape[1] // 2, source.shape[0] // 2)
            else:
                center = (source.shape[1] // 2, source.shape[0] // 2)
            
            result = cv2.seamlessClone(source, target, binary_mask, center, cv2.NORMAL_CLONE)
            return result
            
        except Exception as e:
            print(f"Poisson blending failed: {e}, falling back to alpha blending")
            return self.alpha_blend(source, target, mask)
    
    def alpha_blend(self, source, target, mask):
        """Simple alpha blending"""
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        
        result = target.astype(np.float32) * (1 - mask_3ch) + source.astype(np.float32) * mask_3ch
        return result.astype(np.uint8)
    
    def multiband_blend(self, source, target, mask):
        """Multi-band blending for better transitions"""
        levels = 6
        mask_normalized = mask.astype(np.float32) / 255.0
        
        source_pyr = [source.astype(np.float32)]
        target_pyr = [target.astype(np.float32)]
        mask_pyr = [mask_normalized]
        
        for i in range(levels):
            source_pyr.append(cv2.pyrDown(source_pyr[i]))
            target_pyr.append(cv2.pyrDown(target_pyr[i]))
            mask_pyr.append(cv2.pyrDown(mask_pyr[i]))
        
        result_pyr = []
        for i in range(levels + 1):
            mask_3ch = np.stack([mask_pyr[i]] * 3, axis=-1)
            blended = target_pyr[i] * (1 - mask_3ch) + source_pyr[i] * mask_3ch
            result_pyr.append(blended)
        
        result = result_pyr[levels]
        for i in range(levels - 1, -1, -1):
            result = cv2.pyrUp(result, dstsize=(result_pyr[i].shape[1], result_pyr[i].shape[0]))
            result = result + result_pyr[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def gaussian_blend(self, source, target, mask):
        """Gaussian-weighted blending"""
        mask_normalized = mask.astype(np.float32) / 255.0
        
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_transform = dist_transform / (dist_transform.max() + 1e-8)
        
        gaussian_mask = cv2.GaussianBlur(dist_transform, (21, 21), 0)
        mask_3ch = np.stack([gaussian_mask] * 3, axis=-1)
        
        result = target.astype(np.float32) * (1 - mask_3ch) + source.astype(np.float32) * mask_3ch
        return result.astype(np.uint8)
    
    def create_preview_diff(self, original, edited, mask):
        """Create a preview showing the differences"""
        preview = original.copy()
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        
        tint_color = np.array([255, 100, 100])  # Red tint
        tinted = preview.astype(np.float32) * (1 - mask_3ch * 0.3) + tint_color * mask_3ch * 0.3
        
        return tinted.astype(np.uint8)
    
    def merge_diff(self, original_image, edited_image, sensitivity, detection_method,
                   blend_method, mask_blur, mask_expand, edge_feather, 
                   min_change_area, global_threshold, manual_mask=None):
        
        # Convert tensors to numpy
        original_np = self.tensor_to_numpy(original_image)
        edited_np = self.tensor_to_numpy(edited_image)
        
        # Ensure images are the same size
        if original_np.shape != edited_np.shape:
            edited_np = cv2.resize(edited_np, (original_np.shape[1], original_np.shape[0]))
        
        # Use manual mask if provided, otherwise detect changes
        if manual_mask is not None:
            mask = (manual_mask[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            # Detect changes using selected method
            mask = self.detect_changes(original_np, edited_np, sensitivity, detection_method, global_threshold)
            
            # Filter out small changes (likely noise)
            if min_change_area > 0:
                mask = self.filter_small_changes(mask, min_change_area)
        
        # Refine the mask
        refined_mask = self.refine_mask(mask, mask_expand, mask_blur, edge_feather)
        
        # Blend images based on method
        if blend_method == "poisson":
            result = self.poisson_blend(edited_np, original_np, refined_mask)
        elif blend_method == "alpha":
            result = self.alpha_blend(edited_np, original_np, refined_mask)
        elif blend_method == "multiband":
            result = self.multiband_blend(edited_np, original_np, refined_mask)
        elif blend_method == "gaussian":
            result = self.gaussian_blend(edited_np, original_np, refined_mask)
        else:
            result = self.alpha_blend(edited_np, original_np, refined_mask)
        
        # Create preview
        preview_diff = self.create_preview_diff(original_np, edited_np, refined_mask)
        
        # Convert back to tensors
        result_tensor = self.numpy_to_tensor(result)
        mask_tensor = torch.from_numpy(refined_mask.astype(np.float32) / 255.0).unsqueeze(0)
        preview_tensor = self.numpy_to_tensor(preview_diff)
        
        return (result_tensor, mask_tensor, preview_tensor)


NODE_CLASS_MAPPINGS = {
    "FluxKontextDiffMerge": FluxKontextDiffMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextDiffMerge": "Flux Kontext Diff Merge"
} 