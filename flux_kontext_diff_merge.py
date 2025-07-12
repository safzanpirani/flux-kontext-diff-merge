import torch
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import binary_dilation, binary_erosion
import comfy.model_management as model_management


class FluxKontextDiffMerge:
    """Flux Kontext Diff Merge

    This node compares an *original* image against an *edited* image, detects the
    regions that have changed, builds a refined mask, then blends the edited
    pixels back onto the original using one of several blending strategies.

    The most common workflow is:
      1. Render an image (original)
      2. Send it to an img-to-img pipeline where you repaint/erase parts
      3. Feed *both* images into this node so only the modified areas are
         re-inserted into the original with clean edges.

    Parameter Overview
    ------------------
    • Change Threshold –  Controls how sensitive the detector is.  Lower = pick up
      even small colour shifts; higher = only obvious changes.
    • Detection Method –  Algorithm used to build the initial difference mask.
        – adaptive  (LAB colour threshold + global-aware)
        – color_diff (RGB channel diff)
        – ssim       (structural similarity)
        – combined   (adaptive OR edge-aware)
    • Blend Method – How to merge the edited pixels back in.
        – poisson    Seamless-clone (best when mask is tidy)
        – alpha      Simple linear alpha composite (fast & safe fallback)
        – multiband  Laplacian pyramid blend for smooth transitions
        – gaussian   Distance-weighted alpha ramp
    • Mask Blur – Gaussian blur radius (pixels) applied to improve soft edges.
    • Mask Expand – Dilate mask by n pixels before blur (helps cover halos).
    • Edge Feather – Extra fine feather after blur for subtle fades.
    • Min Change Area – Ignore isolated specks below this size (pixel²).
    • Global Threshold – If the entire image shifted (eg colour grade) the
      adaptive detector automatically relaxes; this scalar lets you tune that.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "edited_image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "label": "Change Threshold",
                    "tooltip": "Sensitivity of change detection; lower values detect more subtle changes"
                }),
                "detection_method": (["adaptive", "color_diff", "ssim", "combined"], {
                    "default": "adaptive",
                    "tooltip": "Algorithm used to build the initial mask of differences"
                }),
                "blend_method": (["poisson", "alpha", "multiband", "gaussian"], {
                    "default": "poisson",
                    "tooltip": "How to merge the edited pixels back onto the original"
                }),
                "mask_blur": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Gaussian blur radius (in pixels) applied to the mask to soften edges"
                }),
                "mask_expand": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Dilate the detected mask by this many pixels before blurring"
                }),
                "edge_feather": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Additional feathering (fine Gaussian) after the main blur"
                }),
                "min_change_area": ("INT", {
                    "default": 250,
                    "min": 0,
                    "max": 5000,
                    "step": 10,
                    "tooltip": "Ignore change blobs smaller than this area (pixel²)"
                }),
                "global_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Controls how aggressively the adaptive detector compensates for global shifts"
                }),
            },
            "optional": {
                "manual_mask": ("MASK", {"tooltip": "User-supplied 1-channel mask (black/white) to override automatic detection"}),
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
    
    def tensor_to_numpy_list(self, tensor):
        """Convert a (batched) ComfyUI tensor to a list of HxWxC uint8 numpy arrays"""
        # Ensure we are working on CPU
        tensor_cpu = tensor.detach().cpu()
        # ComfyUI uses NHWC tensors (batch, height, width, channels)
        if len(tensor_cpu.shape) == 4:
            batch = tensor_cpu.shape[0]
            imgs = []
            for i in range(batch):
                img = tensor_cpu[i].numpy()
                if img.dtype != np.uint8:
                    img = (img * 255.0).astype(np.uint8)
                imgs.append(img)
            return imgs
        else:
            img = tensor_cpu.numpy()
            if img.dtype != np.uint8:
                img = (img * 255.0).astype(np.uint8)
            return [img]

    def numpy_list_to_tensor(self, np_list):
        """Convert list of HxWxC uint8/float numpy arrays to a batched ComfyUI tensor"""
        tensor_list = []
        for arr in np_list:
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            if len(arr.shape) == 3:
                arr = np.expand_dims(arr, axis=0)  # Add batch dim
            tensor_list.append(torch.from_numpy(arr))
        if len(tensor_list) == 0:
            raise ValueError("numpy_list_to_tensor received an empty list")
        return torch.cat(tensor_list, dim=0)
    
    def adaptive_detection(self, original, edited, threshold=0.02, global_threshold=0.15):
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
            thres = global_avg + (threshold * 255)
        else:
            # Low global changes - use absolute threshold
            thres = threshold * 255
        
        # Create binary mask
        mask = (combined_diff > thres).astype(np.uint8) * 255
        
        return mask
    
    def edge_aware_detection(self, original, edited, threshold=0.02):
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
        thres = threshold * 255
        intensity_mask = (intensity_diff > thres).astype(np.uint8) * 255
        
        # Combine edge and intensity differences
        combined_mask = cv2.bitwise_or(edge_diff, intensity_mask)
        
        return combined_mask
    
    def detect_changes(self, original, edited, threshold=0.02, method="adaptive", global_threshold=0.15):
        """Main change detection with multiple methods"""
        if method == "adaptive":
            mask = self.adaptive_detection(original, edited, threshold, global_threshold)
        elif method == "color_diff":
            mask = self.detect_color_changes(original, edited, threshold)
        elif method == "ssim":
            mask = self.detect_ssim_changes(original, edited, threshold)
        elif method == "combined":
            # Combine multiple methods
            mask1 = self.adaptive_detection(original, edited, threshold, global_threshold)
            mask2 = self.edge_aware_detection(original, edited, threshold)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = self.adaptive_detection(original, edited, threshold, global_threshold)
        
        return mask
    
    def detect_color_changes(self, original, edited, threshold=0.02):
        """Detect changes in color channels"""
        diff_r = np.abs(original[:,:,0].astype(np.float32) - edited[:,:,0].astype(np.float32))
        diff_g = np.abs(original[:,:,1].astype(np.float32) - edited[:,:,1].astype(np.float32))
        diff_b = np.abs(original[:,:,2].astype(np.float32) - edited[:,:,2].astype(np.float32))
        
        combined_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
        thres = threshold * 255
        mask = (combined_diff > thres).astype(np.uint8) * 255
        
        return mask
    
    def detect_ssim_changes(self, original, edited, threshold=0.02):
        """Detect changes using structural similarity"""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        edit_gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
        
        try:
            score, diff = ssim(orig_gray, edit_gray, full=True, data_range=255)
        except:
            diff = np.abs(orig_gray.astype(np.float32) - edit_gray.astype(np.float32)) / 255.0
        
        diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        mask = (diff_normalized > threshold).astype(np.uint8) * 255
        
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
            # Ensure inputs are valid
            if source.shape != target.shape:
                print(f"Poisson blending failed: shape mismatch {source.shape} vs {target.shape}, falling back to alpha blending")
                return self.alpha_blend(source, target, mask)
            
            binary_mask = (mask > 127).astype(np.uint8) * 255
            
            # Check if mask has any valid regions
            if np.sum(binary_mask) == 0:
                print("Poisson blending failed: empty mask, falling back to alpha blending")
                return self.alpha_blend(source, target, mask)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Check if contour is large enough
                if cv2.contourArea(largest_contour) < 10:
                    print("Poisson blending failed: contour too small, falling back to alpha blending")
                    return self.alpha_blend(source, target, mask)
                
                # Use bounding rectangle center instead of image moments to avoid occasional dislocation
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x = x + w // 2
                center_y = y + h // 2
                # Clip center to image bounds                         
                center_x = max(1, min(center_x, source.shape[1] - 2))
                center_y = max(1, min(center_y, source.shape[0] - 2))
                center = (center_x, center_y)
            else:
                center = (source.shape[1] // 2, source.shape[0] // 2)
            
            # Validate that the center point is within the mask region
            if binary_mask[center[1], center[0]] == 0:
                # Find a valid point within the mask
                mask_points = np.where(binary_mask > 0)
                if len(mask_points[0]) > 0:
                    center = (mask_points[1][0], mask_points[0][0])
                else:
                    print("Poisson blending failed: no valid mask points, falling back to alpha blending")
                    return self.alpha_blend(source, target, mask)
            
            # Ensure mask boundaries don't touch image edges to avoid ROI issues
            h, w = binary_mask.shape
            boundary_mask = binary_mask.copy()
            boundary_mask[0, :] = 0
            boundary_mask[-1, :] = 0
            boundary_mask[:, 0] = 0
            boundary_mask[:, -1] = 0
            
            # If mask touches edges, use alpha blending instead
            if np.sum(boundary_mask) != np.sum(binary_mask):
                print("Poisson blending failed: mask touches image boundaries, falling back to alpha blending")
                return self.alpha_blend(source, target, mask)
            
            result = cv2.seamlessClone(source, target, boundary_mask, center, cv2.NORMAL_CLONE)
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
        try:
            levels = 6
            mask_normalized = mask.astype(np.float32) / 255.0
            
            # Ensure inputs are valid
            if source.shape != target.shape:
                print(f"Multiband blending failed: shape mismatch, falling back to alpha blending")
                return self.alpha_blend(source, target, mask)
            
            source_pyr = [source.astype(np.float32)]
            target_pyr = [target.astype(np.float32)]
            mask_pyr = [mask_normalized]
            
            for i in range(levels):
                # Ensure minimum size for pyramid levels
                if source_pyr[i].shape[0] < 4 or source_pyr[i].shape[1] < 4:
                    levels = i
                    break
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
            
        except Exception as e:
            print(f"Multiband blending failed: {e}, falling back to alpha blending")
            return self.alpha_blend(source, target, mask)
    
    def gaussian_blend(self, source, target, mask):
        """Gaussian-weighted blending"""
        try:
            # Ensure inputs are valid
            if source.shape != target.shape:
                print(f"Gaussian blending failed: shape mismatch, falling back to alpha blending")
                return self.alpha_blend(source, target, mask)
            
            mask_normalized = mask.astype(np.float32) / 255.0
            
            # Check if mask has any content
            if np.sum(mask_normalized) == 0:
                return target  # No changes needed
            
            dist_transform = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            dist_transform = dist_transform / (dist_transform.max() + 1e-8)
            
            gaussian_mask = cv2.GaussianBlur(dist_transform, (21, 21), 0)
            mask_3ch = np.stack([gaussian_mask] * 3, axis=-1)
            
            result = target.astype(np.float32) * (1 - mask_3ch) + source.astype(np.float32) * mask_3ch
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Gaussian blending failed: {e}, falling back to alpha blending")
            return self.alpha_blend(source, target, mask)
    
    def create_preview_diff(self, original, edited, mask):
        """Create a preview showing the differences"""
        preview = original.copy()
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        
        tint_color = np.array([255, 100, 100])  # Red tint
        tinted = preview.astype(np.float32) * (1 - mask_3ch * 0.3) + tint_color * mask_3ch * 0.3
        
        return tinted.astype(np.uint8)
    
    def merge_diff(self, original_image, edited_image, threshold, detection_method,
                   blend_method, mask_blur, mask_expand, edge_feather, 
                   min_change_area, global_threshold, manual_mask=None):
        """Main entry point – now supports batched inputs"""
        # Convert input tensors (possibly batched) to lists of numpy images
        original_list = self.tensor_to_numpy_list(original_image)
        edited_list = self.tensor_to_numpy_list(edited_image)

        batch_size = len(original_list)
        if batch_size != len(edited_list):
            # Allow broadcast when original has 1 image and edited has multiple
            if batch_size == 1 and len(edited_list) > 1:
                print(f"Broadcasting single original image to match edited batch of size {len(edited_list)}")
                original_list = original_list * len(edited_list)
                batch_size = len(edited_list)
            else:
                raise ValueError(f"Original and edited image batch sizes differ: {batch_size} vs {len(edited_list)}")

        # Prepare manual mask list if provided
        manual_mask_list = None
        if manual_mask is not None:
            # manual_mask tensor shape is likely (B, H, W)
            manual_mask_np = manual_mask.detach().cpu().numpy()
            if len(manual_mask_np.shape) == 3:  # Batched masks
                manual_mask_list = [(manual_mask_np[i] * 255).astype(np.uint8) for i in range(manual_mask_np.shape[0])]
            else:  # Single mask, broadcast
                manual_mask_list = [(manual_mask_np * 255).astype(np.uint8)] * batch_size
            if len(manual_mask_list) != batch_size:
                # Resize/broadcast to match batch
                manual_mask_list = (manual_mask_list * batch_size)[:batch_size]

        # Containers for outputs
        result_np_list = []
        mask_np_list = []
        preview_np_list = []

        # Process each item in batch independently
        for idx in range(batch_size):
            original_np = original_list[idx]
            edited_np = edited_list[idx]

            # Ensure images are the same size per item
            if original_np.shape != edited_np.shape:
                print(f"Resizing edited image in batch index {idx} from {edited_np.shape} to {original_np.shape}")
                edited_np = cv2.resize(edited_np, (original_np.shape[1], original_np.shape[0]))

            # Handle manual mask if available
            if manual_mask_list is not None:
                mask = manual_mask_list[idx]
                # Resize mask if needed
                if mask.shape != original_np.shape[:2]:
                    mask = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]))
            else:
                # Detect changes
                mask = self.detect_changes(original_np, edited_np, threshold, detection_method, global_threshold)
                if min_change_area > 0:
                    mask = self.filter_small_changes(mask, min_change_area)

            # Refine mask
            refined_mask = self.refine_mask(mask, mask_expand, mask_blur, edge_feather)

            # Blend based on selected method
            if blend_method == "poisson":
                result_np = self.poisson_blend(edited_np, original_np, refined_mask)
            elif blend_method == "alpha":
                result_np = self.alpha_blend(edited_np, original_np, refined_mask)
            elif blend_method == "multiband":
                result_np = self.multiband_blend(edited_np, original_np, refined_mask)
            elif blend_method == "gaussian":
                result_np = self.gaussian_blend(edited_np, original_np, refined_mask)
            else:
                result_np = self.alpha_blend(edited_np, original_np, refined_mask)

            # Preview diff
            preview_np = self.create_preview_diff(original_np, edited_np, refined_mask)

            # Collect outputs
            result_np_list.append(result_np)
            mask_np_list.append(refined_mask)
            preview_np_list.append(preview_np)

        # Convert outputs back to batched tensors
        result_tensor = self.numpy_list_to_tensor(result_np_list)
        mask_tensor_list = []
        for m_np in mask_np_list:
            m_float = (m_np.astype(np.float32) / 255.0)
            mask_tensor_list.append(torch.from_numpy(np.expand_dims(m_float, axis=0)))  # add batch dim
        mask_tensor = torch.cat(mask_tensor_list, dim=0)
        preview_tensor = self.numpy_list_to_tensor(preview_np_list)

        return (result_tensor, mask_tensor, preview_tensor)


NODE_CLASS_MAPPINGS = {
    "FluxKontextDiffMerge": FluxKontextDiffMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextDiffMerge": "Flux Kontext Diff Merge"
} 