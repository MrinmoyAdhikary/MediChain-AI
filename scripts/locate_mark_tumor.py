import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# -------------------------------------------------------------------------
# 1. Configuration & Model Initialization
# -------------------------------------------------------------------------
MODEL_PATH = "brain_tumor_ct_model.keras"
# Test on your saturated or focal tumor scan
IMAGE_PATH = Path(sys.argv[1])

OUTPUT_PLOT_PATH = "tumor_marked_result.jpg"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at: {IMAGE_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Extract pipeline layers
rescaling_layer = model.get_layer("rescaling")
base_model = model.get_layer("mobilenetv2_1.00_224")
gap_layer = model.get_layer("global_average_pooling2d")
dropout_layer = model.get_layer("dropout")
output_layer = model.get_layer("tumor_probability")

# Connect feature extractor directly to the final conv activation (out_relu)
target_conv_layer = base_model.get_layer("out_relu")
backbone_feature_extractor = tf.keras.Model(
    inputs=base_model.inputs,
    outputs=target_conv_layer.output
)

# Extract dense weights/biases to calculate pre-activation logits directly
output_weights, output_bias = output_layer.get_weights()
output_weights = tf.constant(output_weights, dtype=tf.float32)
output_bias = tf.constant(output_bias, dtype=tf.float32)

print(f"Loaded {MODEL_PATH} successfully.")
print("Pre-sigmoid logit gradient engine ready.")

# -------------------------------------------------------------------------
# 2. Non-Saturating Logit Grad-CAM Engine
# -------------------------------------------------------------------------
def generate_gradcam_logits(img_tensor):
    # Pass raw [0, 255] pixels to the native MobileNet rescaling layer
    x_scaled = rescaling_layer(img_tensor)

    with tf.GradientTape() as tape:
        # Extract conv feature maps
        conv_outputs = backbone_feature_extractor(x_scaled)
        tape.watch(conv_outputs)

        # Forward pass through classification head
        pooled = gap_layer(conv_outputs)
        dropped = dropout_layer(pooled, training=False)

        # Compute pre-sigmoid logit: z = W*x + b
        logit = tf.matmul(dropped, output_weights) + output_bias
        
        # Calculate true probability for diagnosis reporting
        raw_prob = float(tf.sigmoid(logit)[0][0])
        is_tumor = raw_prob >= 0.5

        # Target the linear logit directly (derivative is constant, preventing zero collapse)
        target_score = logit[0][0] if is_tumor else -logit[0][0]

    # Compute gradients w.r.t. out_relu
    grads = tape.gradient(target_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Standard ReLU and normalization
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / (max_val + 1e-10)

    return heatmap.numpy(), raw_prob, is_tumor

# -------------------------------------------------------------------------
# 3. Dynamic Thresholding, Localization & Bounding Box
# -------------------------------------------------------------------------
def locate_and_mark_tumor(image_path, save_path=OUTPUT_PLOT_PATH):
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise ValueError(f"Could not load image: {image_path}")

    orig_h, orig_w = raw_img.shape[:2]
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_tensor = np.expand_dims(img_resized.astype(np.float32), axis=0)

    # Run inference with non-vanishing gradients
    heatmap, raw_prob, is_tumor = generate_gradcam_logits(img_tensor)

    confidence = raw_prob * 100 if is_tumor else (1.0 - raw_prob) * 100
    pred_label = "Tumor" if is_tumor else "Healthy"

    print("\n==========================================")
    print("        TUMOR LOCALIZATION REPORT         ")
    print("==========================================")
    print(f"Sigmoid Confidence : {confidence:.2f}%")
    print(f"Diagnosis          : {pred_label}")

    if not is_tumor:
        print("Result: Normal scan. No tumor bounding box needed.")
        return []

    # Step A: Scale heatmap back to original scan resolution
    heatmap_scaled = cv2.resize(heatmap, (orig_w, orig_h))
    heatmap_uint8 = np.uint8(255 * heatmap_scaled)

    # Step B: Adaptive peak thresholding
    # Targets the top 50% of the active hotspot to handle both compact and diffuse lesions
    peak_val = np.max(heatmap_uint8)
    cutoff = int(peak_val * 0.75) if peak_val > 50 else 50
    _, binary_mask = cv2.threshold(heatmap_uint8, cutoff, 255, cv2.THRESH_BINARY)

    # Step C: Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_DILATE, kernel)

    # Step D: Extract contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated_img = raw_img.copy()
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(raw_img, 0.65, color_map, 0.35, 0)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:  # Minimum pixel threshold
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            detections.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "center_x": int(cx),
                "center_y": int(cy),
                "area": int(area)
            })

            # Draw Red Bounding Box & Yellow Centroid
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.circle(annotated_img, (cx, cy), 6, (0, 255, 255), -1)

            # Label box
            label = f"Tumor ({confidence:.1f}%)"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_img, (x, max(y - 25, 0)), (x + text_size[0] + 8, max(y, 25)), (0, 0, 255), -1)
            cv2.putText(annotated_img, label, (x + 4, max(y - 7, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    print(f"Detected Regions   : {len(detections)}")
    for i, d in enumerate(detections, 1):
        print(f"  [Box {i}] X={d['x']}, Y={d['y']}, Width={d['width']}px, Height={d['height']}px | Centroid=({d['center_x']}, {d['center_y']})")
    print("==========================================\n")

    # Display & Save side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Input CT Scan")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("2. Grad-CAM Focus Heatmap")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"3. Localized Tumor ({confidence:.1f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Diagnostic grid saved to '{save_path}'")
    plt.show()

    return detections

# -------------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    locate_and_mark_tumor(IMAGE_PATH)