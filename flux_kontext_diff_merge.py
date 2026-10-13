import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


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
    • Global Threshold – If the entire image changed colour, the adaptive
      detector relaxes; this does not compensate for geometric shifts.
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
                    "tooltip": "Controls adaptive detection of global colour changes, not image position"
                }),
            },
            "optional": {
                "manual_mask": ("MASK", {"tooltip": "User-supplied 1-channel mask (black/white) to override automatic detection"}),
                "alignment": (["none", "translation"], {
                    "default": "none",
                    "tooltip": "Optionally align small global translations using agreement between unchanged image tiles"
                }),
                "max_alignment_shift": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Maximum accepted translation in pixels along either axis"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("merged_image", "difference_mask", "preview_diff")
    FUNCTION = "merge_diff"
    CATEGORY = "image/postprocessing"

    @staticmethod
    def _to_uint8_image(array):
        """Normalize an image array to uint8."""
        if array.dtype == np.uint8:
            return array
        return np.clip(array * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_uint8_mask(mask):
        """Normalize a mask array to uint8."""
        return np.clip(mask, 0, 255).astype(np.uint8)

    @staticmethod
    def _kernel_size(radius):
        """Convert a blur/expand radius to an OpenCV kernel size."""
        radius = max(0, int(radius))
        if radius == 0:
            return 0
        return radius * 2 + 1

    @staticmethod
    def _resize_mask(mask, target_shape):
        """Resize a mask to match the target image shape."""
        return cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    @staticmethod
    def align_translation(original, edited, max_shift=16):
        """Align only translations supported by spatially distributed tiles.

        Ambiguous estimates leave the edited image untouched. The returned
        validity mask marks pixels actually covered by the translated edit.
        """
        height, width = original.shape[:2]
        valid = np.ones((height, width), dtype=bool)
        if min(height, width) < 96:
            return edited, valid

        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
        edit_gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY).astype(np.float32)
        tiles = []
        for row in range(3):
            for col in range(3):
                y0, y1 = row * height // 3, (row + 1) * height // 3
                x0, x1 = col * width // 3, (col + 1) * width // 3
                orig_tile = orig_gray[y0:y1, x0:x1]
                edit_tile = edit_gray[y0:y1, x0:x1]
                if min(orig_tile.std(), edit_tile.std()) < 2.0:
                    continue
                window = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
                shift, response = cv2.phaseCorrelate(
                    orig_tile.copy(), edit_tile.copy(), window)
                if (np.isfinite(response) and response >= 0.3
                        and np.all(np.isfinite(shift))
                        and max(abs(shift[0]), abs(shift[1])) <= max_shift):
                    tiles.append((row, col, y0, y1, x0, x1, shift))

        if len(tiles) < 4:
            return edited, valid
        shifts = np.array([tile[6] for tile in tiles])
        # Select the largest agreeing group, not an average pulled by edits.
        distances = np.max(np.abs(shifts[:, None] - shifts[None, :]), axis=2)
        seed = np.argmax(np.sum(distances <= 1.0, axis=1))
        inliers = [tile for tile, distance in zip(tiles, distances[seed])
                   if distance <= 1.0]
        if (len(inliers) < 4 or len({tile[0] for tile in inliers}) < 2
                or len({tile[1] for tile in inliers}) < 2):
            return edited, valid
        shift = np.median([tile[6] for tile in inliers], axis=0)
        # Avoid resampling a near-integer shift or numerical identity noise.
        shift = np.where(np.abs(shift - np.round(shift)) <= 0.15,
                         np.round(shift), shift)
        if np.max(np.abs(shift)) < 0.25:
            return edited, valid

        transform = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]]],
                             dtype=np.float32)
        aligned = cv2.warpAffine(edited, transform, (width, height),
                                 flags=cv2.INTER_LINEAR)
        coverage = cv2.warpAffine(np.ones((height, width), dtype=np.float32),
                                  transform, (width, height),
                                  flags=cv2.INTER_LINEAR)
        valid = coverage >= 1.0
        aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY).astype(np.float32)
        improved = []
        for tile in inliers:
            row, col, y0, y1, x0, x1, _ = tile
            overlap = valid[y0:y1, x0:x1]
            if not np.any(overlap):
                continue
            reference = orig_gray[y0:y1, x0:x1][overlap]
            before = np.mean(np.abs(reference - edit_gray[y0:y1, x0:x1][overlap]))
            after = np.mean(np.abs(reference - aligned_gray[y0:y1, x0:x1][overlap]))
            if before > 1.0 and after <= before * 0.8:
                improved.append((row, col))
        if (len(improved) < 4 or len({row for row, _ in improved}) < 2
                or len({col for _, col in improved}) < 2):
            return edited, np.ones((height, width), dtype=bool)

        aligned[~valid] = original[~valid]
        return aligned, valid

    def _prepare_manual_mask_list(self, manual_mask, batch_size):
        """Normalize manual mask tensors into a per-item uint8 mask list."""
        if manual_mask is None:
            return None

        manual_mask_np = manual_mask.detach().cpu().numpy()
        if manual_mask_np.ndim == 2:
            masks = [self._to_uint8_mask(manual_mask_np * 255.0)]
        elif manual_mask_np.ndim == 3:
            masks = [
                self._to_uint8_mask(manual_mask_np[index] * 255.0)
                for index in range(manual_mask_np.shape[0])
            ]
        else:
            raise ValueError(
                f"Manual mask must have 2 or 3 dimensions, got {manual_mask_np.ndim}"
            )

        if len(masks) == 1:
            return masks * batch_size
        if len(masks) != batch_size:
            return (masks * batch_size)[:batch_size]
        return masks

    def _mask_list_to_tensor(self, mask_list):
        """Convert a list of uint8 masks into a batched ComfyUI tensor."""
        mask_tensor_list = []
        for mask in mask_list:
            mask_float = mask.astype(np.float32) / 255.0
            mask_tensor_list.append(
                torch.from_numpy(np.expand_dims(mask_float, axis=0))
            )
        return torch.cat(mask_tensor_list, dim=0)
    
    def tensor_to_numpy(self, tensor):
        """Convert ComfyUI tensor to numpy array"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        numpy_image = tensor.cpu().numpy()
        return self._to_uint8_image(numpy_image)
    
    def numpy_to_tensor(self, numpy_array):
        """Convert numpy array back to ComfyUI tensor"""
        if numpy_array.dtype == np.uint8:
            numpy_array = numpy_array.astype(np.float32) / 255.0
        if len(numpy_array.shape) == 3:
            numpy_array = np.expand_dims(numpy_array, axis=0)
        return torch.from_numpy(np.ascontiguousarray(numpy_array))
    
    def tensor_to_numpy_list(self, tensor):
        """Convert a (batched) ComfyUI tensor to a list of HxWxC uint8 numpy arrays"""
        tensor_cpu = tensor.detach().cpu()
        if len(tensor_cpu.shape) == 4:
            imgs = []
            for i in range(tensor_cpu.shape[0]):
                imgs.append(self._to_uint8_image(tensor_cpu[i].numpy()))
            return imgs
        return [self._to_uint8_image(tensor_cpu.numpy())]

    def numpy_list_to_tensor(self, np_list):
        """Convert list of HxWxC uint8/float numpy arrays to a batched ComfyUI tensor"""
        tensor_list = []
        for arr in np_list:
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            if len(arr.shape) == 3:
                arr = np.expand_dims(arr, axis=0)
            tensor_list.append(torch.from_numpy(np.ascontiguousarray(arr)))
        if not tensor_list:
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
            _, similarity_map = ssim(
                orig_gray,
                edit_gray,
                full=True,
                data_range=255,
            )
            diff_normalized = np.clip(
                (1.0 - similarity_map.astype(np.float32)) * 0.5,
                0.0,
                1.0,
            )
        except ValueError:
            diff_normalized = (
                np.abs(orig_gray.astype(np.float32) - edit_gray.astype(np.float32))
                / 255.0
            )

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
        
        if expand_pixels > 0:
            expand_kernel_size = self._kernel_size(expand_pixels)
            expand_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (expand_kernel_size, expand_kernel_size),
            )
            mask = cv2.dilate(mask, expand_kernel, iterations=1)
        
        if blur_amount > 0:
            blur_kernel_size = self._kernel_size(blur_amount)
            mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)
        
        if feather_amount > 0:
            feather_kernel_size = self._kernel_size(feather_amount)
            mask = cv2.GaussianBlur(
                mask,
                (feather_kernel_size, feather_kernel_size),
                feather_amount / 3,
            )
        
        return mask
    
    def poisson_blend(self, source, target, mask):
        """Poisson blending for seamless integration (robust to edge masks)."""
        try:
            # Ensure inputs are valid
            if source.shape != target.shape:
                print(f"Poisson blending failed: shape mismatch {source.shape} vs {target.shape}, falling back to alpha blending")
                return self.alpha_blend(source, target, mask)

            # Build binary mask
            binary_mask = (mask > 127).astype(np.uint8) * 255
            if np.sum(binary_mask) == 0:
                print("Poisson blending failed: empty mask, falling back to alpha blending")
                return self.alpha_blend(source, target, mask)

            h, w = binary_mask.shape

            # Use an eroded mask to find a stable center (avoid threshold-induced drift)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            inner_mask = cv2.erode(binary_mask, kernel, iterations=1)
            if np.sum(inner_mask) == 0:
                inner_mask = binary_mask

            # Select center as the pixel farthest from the boundary (distance transform)
            dist = cv2.distanceTransform((inner_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
            if dist[cy, cx] <= 0:
                # Fallback: bounding rect center
                x, y, ww, hh = cv2.boundingRect(binary_mask)
                cx = x + ww // 2
                cy = y + hh // 2

            # Clamp to safe interior for seamlessClone
            cx = int(max(1, min(cx, w - 2)))
            cy = int(max(1, min(cy, h - 2)))

            # If mask touches any image border, pad images instead of failing
            touches_edge = (
                np.any(binary_mask[0, :]) or np.any(binary_mask[-1, :]) or
                np.any(binary_mask[:, 0]) or np.any(binary_mask[:, -1])
            )

            if touches_edge:
                pad = max(16, (max(h, w) // 100) * 2 + 8)  # dynamic-ish padding, min 16
                src_p = cv2.copyMakeBorder(source, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
                dst_p = cv2.copyMakeBorder(target, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
                msk_p = cv2.copyMakeBorder(binary_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

                center = (cx + pad, cy + pad)
                result_p = cv2.seamlessClone(src_p, dst_p, msk_p, center, cv2.NORMAL_CLONE)
                result = result_p[pad:pad + h, pad:pad + w]
            else:
                center = (cx, cy)
                result = cv2.seamlessClone(source, target, binary_mask, center, cv2.NORMAL_CLONE)

            return self.alpha_blend(result, target, mask)

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
        
        return np.clip(tinted, 0, 255).astype(np.uint8)
    
    def merge_diff(self, original_image, edited_image, threshold, detection_method,
                   blend_method, mask_blur, mask_expand, edge_feather, 
                   min_change_area, global_threshold, manual_mask=None,
                   alignment="none", max_alignment_shift=16):
        """Main entry point – now supports batched inputs"""
        original_list = self.tensor_to_numpy_list(original_image)
        edited_list = self.tensor_to_numpy_list(edited_image)

        batch_size = len(original_list)
        if batch_size != len(edited_list):
            if batch_size == 1 and len(edited_list) > 1:
                print(f"Broadcasting single original image to match edited batch of size {len(edited_list)}")
                original_list = original_list * len(edited_list)
                batch_size = len(edited_list)
            else:
                raise ValueError(f"Original and edited image batch sizes differ: {batch_size} vs {len(edited_list)}")

        manual_mask_list = self._prepare_manual_mask_list(manual_mask, batch_size)
        if alignment not in ("none", "translation"):
            raise ValueError(f"Unknown alignment policy: {alignment}")

        result_np_list = []
        mask_np_list = []
        preview_np_list = []

        for idx in range(batch_size):
            original_np = original_list[idx]
            edited_np = edited_list[idx]

            if original_np.shape != edited_np.shape:
                print(f"Resizing edited image in batch index {idx} from {edited_np.shape} to {original_np.shape}")
                edited_np = cv2.resize(edited_np, (original_np.shape[1], original_np.shape[0]))

            valid = None
            if alignment == "translation":
                edited_np, valid = self.align_translation(
                    original_np, edited_np, max_alignment_shift)

            if manual_mask_list is not None:
                mask = manual_mask_list[idx]
                if mask.shape != original_np.shape[:2]:
                    mask = self._resize_mask(mask, original_np.shape)
            else:
                mask = self.detect_changes(
                    original_np,
                    edited_np,
                    threshold,
                    detection_method,
                    global_threshold,
                )
                if min_change_area > 0:
                    mask = self.filter_small_changes(mask, min_change_area)

            refined_mask = self.refine_mask(mask, mask_expand, mask_blur, edge_feather)
            if valid is not None:
                refined_mask[~valid] = 0

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

            if valid is not None:
                result_np[~valid] = original_np[~valid]

            preview_np = self.create_preview_diff(original_np, edited_np, refined_mask)

            result_np_list.append(result_np)
            mask_np_list.append(refined_mask)
            preview_np_list.append(preview_np)

        result_tensor = self.numpy_list_to_tensor(result_np_list)
        mask_tensor = self._mask_list_to_tensor(mask_np_list)
        preview_tensor = self.numpy_list_to_tensor(preview_np_list)

        return (result_tensor, mask_tensor, preview_tensor)


NODE_CLASS_MAPPINGS = {
    "FluxKontextDiffMerge": FluxKontextDiffMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextDiffMerge": "Flux Kontext Diff Merge"
} 