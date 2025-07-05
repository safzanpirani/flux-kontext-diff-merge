# Integration Guide: Adding Diff Merge to Your Flux Kontext Workflow

Here's how to integrate the Flux Kontext Diff Merge node to preserve image quality in your existing workflow.

## Understanding Your Current Workflow

Your workflow follows this pattern:
1. **LoadImageOutput** → **ImageStitch** → **FluxKontextImageScale** → **VAEEncode** → **KSampler** → **VAEDecode** → **SaveImage**

## The Problem
The entire image gets processed through the AI model, causing quality degradation even in areas that shouldn't change.

## The Solution
The Diff Merge node compares original and edited images, then only applies changes where they're actually needed.

## Step-by-Step Integration

### Step 1: Modify Your Workflow Structure

**Current Flow:**
```
LoadImageOutput → ImageStitch → FluxKontextImageScale → VAEEncode → [AI Processing] → VAEDecode → SaveImage
```

**New Flow:**
```
LoadImageOutput → ImageStitch → FluxKontextImageScale → ┬→ VAEEncode → [AI Processing] → VAEDecode → ┬→ FluxKontextDiffMerge → SaveImage
                                                       │                                            │
                                                       └→ [Keep Original] ────────────────────────────┘
```

### Step 2: Add the Diff Merge Node
1. **Add the node**: Search for "Flux Kontext Diff Merge" in your node menu
2. **Position it**: Place it after your VAEDecode node
3. **Connect inputs**:
   - `original_image`: Connect from your **FluxKontextImageScale** output (before VAEEncode)
   - `edited_image`: Connect from your **VAEDecode** output (after AI processing)

### Step 3: Specific Connections

Based on your workflow structure:

#### Original Image Path (Bypass AI):
```
Node 146 (ImageStitch) → Node 42 (FluxKontextImageScale) → FluxKontextDiffMerge (original_image)
```

#### Edited Image Path (Through AI):
```
Node 8 (VAEDecode) → FluxKontextDiffMerge (edited_image)
```

#### Output:
```
FluxKontextDiffMerge (merged_image) → Node 136 (SaveImage)
```

### Step 4: Configure Settings

Use these tested settings:

```
sensitivity: 0.02
detection_method: "adaptive"
global_threshold: 0.15
min_change_area: 250
blend_method: "poisson"
mask_blur: 15
mask_expand: 8
edge_feather: 15
```

## Quality Comparison

**Before (Standard Flux Kontext):**
- ❌ Entire image processed through AI
- ❌ Quality loss in unchanged areas
- ❌ Global color/lighting shifts

**After (With Diff Merge):**
- ✅ Only changed areas processed
- ✅ Original quality preserved
- ✅ Handles global AI processing artifacts
- ✅ No unwanted shifts in unchanged areas

## Troubleshooting

### Entire image shows red overlay
1. **Increase** `global_threshold` to 0.20+
2. **Decrease** `sensitivity` to 0.02
3. **Use** `detection_method: "adaptive"`
4. **Increase** `min_change_area` to 500+

### No changes detected
1. **Decrease** `global_threshold` to 0.08
2. **Increase** `sensitivity` to 0.06
3. **Decrease** `min_change_area` to 100
4. **Check** the `preview_diff` output for debugging

### Small noise areas detected
1. **Increase** `min_change_area` to 500+
2. **Increase** `global_threshold` slightly
3. **Use** `mask_blur` to smooth out small areas

### Blending looks unnatural
1. **Try** different `blend_method` - "poisson" usually works best
2. **Increase** `mask_blur` for smoother edges
3. **Adjust** `edge_feather` for more natural transitions

## Tips

- **Start with the recommended settings** above
- **Check the preview_diff output** to verify change detection
- **Use min_change_area** to filter out noise and small artifacts
- **Save your workflow** once you find optimal settings for your use case

## Final Workflow Structure

```
Original Images → ImageStitch → FluxKontextImageScale → ┬→ VAEEncode → KSampler → VAEDecode → ┬→ FluxKontextDiffMerge → SaveImage
                                                       │                                      │    ↓
                                                       └─────── Original Image ──────────────┘    preview_diff → PreviewImage
                                                                                                   ↓
                                                                                                   difference_mask → SaveImage (optional)
```

This preserves your original image quality while applying only the AI-generated changes where they're needed! 