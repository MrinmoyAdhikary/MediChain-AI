import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "brain_tumor_ct_model.keras"  # or your .h5 file path
IMAGE_PATH = r"D:\Medichain Ai\Testing\glioma\Te-gl_1.jpg"

model = tf.keras.models.load_model(MODEL_PATH)


def run_correct_sigmoid_audit(image_path):
  img = cv2.imread(image_path)
  if img is None:
    print(f"Error: Unable to load image at {image_path}")
    return

  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_224 = cv2.resize(img_rgb, (224, 224))

  # 1. Original Unmodified Scan
  orig_input = np.expand_dims(img_224.astype(np.float32) / 255.0, axis=0)
  raw_orig = float(model.predict(orig_input, verbose=0)[0][0])
  # Inverse mapping: 0 = Tumor, 1 = Healthy
  tumor_prob_orig = (1.0 - raw_orig) * 100

  # 2. Masked Scan (Black out the center 45% of the brain)
  h, w = img_224.shape[:2]
  masked_img = img_224.copy()
  cv2.circle(
      masked_img, (w // 2, h // 2), int(min(h, w) * 0.45), (0, 0, 0), -1
  )

  masked_input = np.expand_dims(masked_img.astype(np.float32) / 255.0, axis=0)
  raw_masked = float(model.predict(masked_input, verbose=0)[0][0])
  tumor_prob_masked = (1.0 - raw_masked) * 100

  print("\n==========================================")
  print("    SIGMOID-CORRECTED OCCLUSION AUDIT     ")
  print("==========================================")
  print(f"Raw Sigmoid (Original) : {raw_orig:.6f}")
  print(f"Raw Sigmoid (Masked)   : {raw_masked:.6f}")
  print("------------------------------------------")
  print(f"Original Tumor Probability : {tumor_prob_orig:.2f}%")
  print(f"Masked Tumor Probability   : {tumor_prob_masked:.2f}%")
  print("==========================================")

  drop = tumor_prob_orig - tumor_prob_masked
  if drop > 30.0:
    print("VERDICT: PASS ✅")
    print(
        f"Tumor probability dropped by {drop:.2f}% when brain tissue was"
        " masked."
    )
    print("The model tracks internal tissue rather than background borders.")
  else:
    print("VERDICT: UNRESOLVED FOCUS ⚠️")
    print(
        f"Probability only shifted by {abs(drop):.2f}%. Check where the glioma"
        " is located."
    )


run_correct_sigmoid_audit(IMAGE_PATH)