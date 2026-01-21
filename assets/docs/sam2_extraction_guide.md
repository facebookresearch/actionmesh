# SAM2 Demo: Extracting Video with White Background

This guide explains how to use the [SAM2 demo](https://sam2.metademolab.com/demo) to extract a segmented video with a white background for use with ActionMesh.

## Why Use SAM2?

While ActionMesh includes [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) for automatic background removal, this is mainly designed for flat backgrounds. For more complex scenes, we recommend using SAM2 to generate a video of the object in motion with a white background.

## Step-by-Step Instructions

### 1. Open SAM2 Demo

Navigate to [https://sam2.metademolab.com/demo](https://sam2.metademolab.com/demo)

### 2. Upload Your Video

- Click **Change Video** → **Upload** and choose your video file

### 3. Select the Object

- Navigate to the first frame of your video
- Click on the object you want to segment (e.g., an animal, person, or object)
- SAM2 will automatically generate a mask for the selected object

### 4. Propagate the Mask

- Click **Track objects** → **Next** to propagate the mask across all frames

### 5. Export with White Background

- In the **Selected Objects** section, select **Original**
- In the **Background** section, select **Erase**
- Click **Next** to validate your choice
- Download the exported video

### 6. Use with ActionMesh

Run ActionMesh on the exported video:

```bash
python inference/video_to_animated_mesh.py --input path/to/your/video.mp4
```

## Tips for Best Results

- **Start with a clear first frame**: Choose a frame where the object is clearly visible and unoccluded
- **Use multiple positive clicks**: For complex objects, click on multiple parts to ensure full coverage
- **Check key frames**: Review the mask on frames with significant motion or occlusion
- **Re-segment if needed**: If tracking drifts, go back and re-initialize on the problematic frame

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Mask includes background | Add negative points (right-click) on unwanted areas |
| Mask misses parts of object | Add positive points on missed regions |
| Tracking drifts over time | Re-initialize the mask on the problematic frame |
