import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import math
import os

# Set up page configuration
st.set_page_config(page_title="Magical Face Bloom AR", layout="centered")
st.title("🌸 Magical Face Bloom AR")
st.markdown("Control the blooming and growth of a magical face-flower using your hands over WebRTC!")

# ---------------------------------------------------------------------------
# GLOBAL/SESSION FACE IMAGE SETUP
# ---------------------------------------------------------------------------
FACE_IMAGE_PATH = "nishanth-removebg-preview.png"
MAX_FACE_SIZE = 140

def load_face_source():
    """Loads and returns the custom face image if it exists, or None."""
    if os.path.exists(FACE_IMAGE_PATH):
        try:
            # Load face image using OpenCV (BGR)
            img = cv2.imread(FACE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # If no alpha channel, add one
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                return img
        except Exception as e:
            print(f"Warning: could not load '{FACE_IMAGE_PATH}' ({e}).")
    return None

FACE_SOURCE = load_face_source()

# ---------------------------------------------------------------------------
# MATH & UTILITY FUNCTIONS (NumPy/OpenCV Equivalents)
# ---------------------------------------------------------------------------
def clamp(v, low, high):
    return max(low, min(high, v))

def lerp(a, b, t):
    return a + (b - a) * t

def get_face_surface(face_src, size):
    """Generates a circular cropped image using OpenCV masks."""
    size = max(4, int(size))
    
    if face_src is not None:
        # Crop to square center
        h, w = face_src.shape[:2]
        side = min(w, h)
        cx, cy = w // 2, h // 2
        square = face_src[cy - side//2 : cy + side//2, cx - side//2 : cx + side//2]
        scaled = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    else:
        # Fallback soft red circle
        scaled = np.zeros((size, size, 4), dtype=np.uint8)
        cv2.circle(scaled, (size // 2, size // 2), size // 2, (48, 12, 215, 230), -1)

    # Create circular mask
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2, 255, -1)
    
    # Apply mask split & merge channels
    b, g, r, a = cv2.split(scaled)
    a = cv2.bitwise_and(a, mask)
    result = cv2.merge([b, g, r, a])
    
    # Glowing outer rim
    cv2.circle(result, (size // 2, size // 2), size // 2 - 1, (110, 90, 255, 200), 2)
    return result

def overlay_image(background, overlay, x, y):
    """Overlays an RGBA image onto a BGR background at specific center coordinates."""
    h, w = overlay.shape[:2]
    x_start = int(x - w // 2)
    y_start = int(y - h // 2)
    
    # Bounds safety check
    bg_h, bg_w = background.shape[:2]
    if x_start < 0 or y_start < 0 or x_start + w > bg_w or y_start + h > bg_h:
        return background

    # Extract regions
    crop_bg = background[y_start:y_start+h, x_start:x_start+w]
    
    # Alpha blend profiles
    overlay_img = overlay[:, :, :3]
    overlay_mask = overlay[:, :, 3:] / 255.0
    
    background[y_start:y_start+h, x_start:x_start+w] = (
        (1.0 - overlay_mask) * crop_bg + overlay_mask * overlay_img
    ).astype(np.uint8)

def draw_stem_leaf(img, x, y, stem_angle, side, size):
    leaf_angle = math.radians(stem_angle + side * 36)
    tip_x = int(x + math.cos(leaf_angle) * size)
    tip_y = int(y - math.sin(leaf_angle) * size)
    
    # Streamlined line presentation for leaf structure
    cv2.line(img, (int(x), int(y)), (tip_x, tip_y), (82, 145, 24), 2)

def draw_face_flower(img, x, y, bloom, stem_angle):
    bloom = clamp(bloom, 0.0, 1.0)
    size = int(28 + bloom * (MAX_FACE_SIZE - 28))
    
    # Draw simple backing decoration
    cv2.circle(img, (int(x), int(y)), int((size + 10) // 2), (25, 0, 90), -1)
    
    # Create and blend cropped face graphic
    face = get_face_surface(FACE_SOURCE, size)
    overlay_image(img, face, x, y)

def draw_single_stem(img, base_x, base_y, angle, length, bloom, grow, frame_count):
    sway = math.sin(frame_count * 0.05 + angle) * 2.4
    final_angle = angle + sway
    real_length = length * (0.58 + grow * 0.76)
    
    end_x = base_x + math.cos(math.radians(final_angle)) * real_length
    end_y = base_y - math.sin(math.radians(final_angle)) * real_length
    
    # Draw stem lines
    cv2.line(img, (int(base_x), int(base_y)), (int(end_x), int(end_y)), (255, 115, 55), 4)
    cv2.line(img, (int(base_x), int(base_y)), (int(end_x), int(end_y)), (255, 235, 175), 1)
    
    draw_stem_leaf(img, lerp(base_x, end_x, 0.40), lerp(base_y, end_y, 0.40), final_angle, -1, 16)
    draw_stem_leaf(img, lerp(base_x, end_x, 0.68), lerp(base_y, end_y, 0.68), final_angle, 1, 13)
    
    draw_face_flower(img, end_x, end_y, bloom, final_angle)
# ---------------------------------------------------------------------------
# HEART PARTICLE SYSTEM
# ---------------------------------------------------------------------------

def draw_heart(img, x, y, size, color):
    """Draw a filled heart using circles + polygon."""
    r = size // 2

    # top circles
    cv2.circle(img, (x - r, y - r), r, color, -1)
    cv2.circle(img, (x + r, y - r), r, color, -1)

    # bottom triangle
    pts = np.array([
        [x - size, y - r],
        [x + size, y - r],
        [x, y + size]
    ], np.int32)

    cv2.fillPoly(img, [pts], color)
# ---------------------------------------------------------------------------
# WEBRTC VIDEO STREAMING PROCESSING INTERFACE
# ---------------------------------------------------------------------------
class ARVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.grow_factor = 0.75
        self.bloom_factor = 0.05
        self.smooth_grow = 0.75
        self.smooth_bloom = 0.05
        self.frame_count = 0
                # Romantic heart particles
        self.hearts = []

        for _ in range(80):
            self.hearts.append({
                "x": np.random.randint(0, 640),
                "y": np.random.randint(-720, 0),
                "speed": np.random.uniform(1.5, 4.0),
                "size": np.random.randint(6, 18),
                "drift": np.random.uniform(-0.6, 0.6)
            })

    def transform(self, frame):
    # Read incoming frame matrix (in BGR space layout)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        self.frame_count += 1
    
        # Draw falling heart particles
        self.update_hearts(img)
    
        # Color conversion for Mediapipe execution
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        flower_base = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                middle = hand_landmarks.landmark[9]
                
                p4 = (int(thumb.x * w), int(thumb.y * h))
                p8 = (int(index.x * w), int(index.y * h))
                p9 = (int(middle.x * w), int(middle.y * h))
                
                finger_dist = math.hypot(p8[0] - p4[0], p8[1] - p4[1])
                normalized = clamp((finger_dist - 18) / 125.0, 0.0, 1.0)
                
                # Check mapping context allocations
                if label == "Left" or len(results.multi_hand_landmarks) == 1:
                    flower_base = p9
                    self.bloom_factor = normalized
                    cv2.line(img, p4, p8, (255, 210, 0), 2)
                    cv2.circle(img, p4, 4, (255, 210, 0), -1)
                    cv2.circle(img, p8, 4, (255, 210, 0), -1)
                else:
                    self.grow_factor = normalized
                    cv2.line(img, p4, p8, (80, 55, 255), 2)
                    cv2.circle(img, p4, 4, (80, 55, 255), -1)
                    cv2.circle(img, p8, 4, (80, 55, 255), -1)

        # Smooth positional updates
        self.smooth_bloom = lerp(self.smooth_bloom, self.bloom_factor, 0.065)
        self.smooth_grow = lerp(self.smooth_grow, self.grow_factor, 0.075)

        if flower_base:
            bx, by = flower_base
            cv2.circle(img, (bx, by), 14, (255, 80, 35), -1)
            
            # Draw our magical composite structures
            draw_single_stem(img, bx, by, 34, 132, self.smooth_bloom, self.smooth_grow, self.frame_count)
            draw_single_stem(img, bx, by, 68, 170, self.smooth_bloom, self.smooth_grow, self.frame_count)
            draw_single_stem(img, bx, by, 104, 170, self.smooth_bloom, self.smooth_grow, self.frame_count)
            draw_single_stem(img, bx, by, 138, 132, self.smooth_bloom, self.smooth_grow, self.frame_count)

        # Draw HUD readout directly into the processing pipeline
        magic_intensity = clamp((self.smooth_bloom - 0.25) / 0.40, 0.0, 1.0)
        cv2.putText(img, f"Bloom: {self.smooth_bloom:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Grow: {self.smooth_grow:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Magic: {magic_intensity:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        return img
    def update_hearts(self, img):

        h, w = img.shape[:2]

        overlay = img.copy()

        for heart in self.hearts:

            heart["y"] += heart["speed"]
            heart["x"] += math.sin(
                self.frame_count * 0.05 + heart["y"] * 0.03
            ) * heart["drift"]

            draw_heart(
                overlay,
                int(heart["x"]),
                int(heart["y"]),
                heart["size"],
                (180, 105, 255)     # Pink (BGR)
            )

            # Respawn
            if heart["y"] > h + 30:
                heart["y"] = np.random.randint(-300, -20)
                heart["x"] = np.random.randint(0, w)
                heart["speed"] = np.random.uniform(1.5, 4.5)
                heart["size"] = np.random.randint(6, 18)

        # Blend for transparency
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

# Initialize Streamlit WebRTC component interface block
webrtc_streamer(
    key="face-bloom-ar",
    video_transformer_factory=ARVideoTransformer,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False}
)