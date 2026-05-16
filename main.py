import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.65,
)

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def pinch_ratio(lm, w, h):
    tx, ty = lm[4].x * w, lm[4].y * h
    ix, iy = lm[8].x * w, lm[8].y * h
    raw = math.hypot(tx - ix, ty - iy)
    return clamp((raw - 12) / 160.0, 0.0, 1.0)

def classify_hand(label):
    return "left" if label == "Right" else "right"

def smooth(current, target, speed=0.07):
    return current + (target - current) * speed

def build_plant(W, H, anchor_x):
    stem_bottom = (anchor_x, H + 50)
    junction    = (anchor_x, H)

    # Shorter branches for a tighter bouquet
    blen = int(H * 0.55)
    branch_tips = []
    sub_junctions = []
    
    num_flowers = 9
    for i in range(num_flowers):
        # Tighter spread: 60 degrees total instead of 130
        base_angle = -30 + i * (60 / (num_flowers - 1))
        
        # Tilt slightly towards center based on anchor position
        tilt = 15 if anchor_x < W / 2 else -15
        angle_deg = base_angle + tilt
        angle_rad = math.radians(angle_deg - 90)
        
        # Stagger lengths so flowers sit compactly but distinctly
        length_factor = 0.85 + 0.15 * math.cos(math.radians(base_angle * 2))
        if i % 2 == 1:
            length_factor *= 0.65  # lower inner layer
            
        length = blen * length_factor
        
        tx = int(junction[0] + math.cos(angle_rad) * length)
        ty = int(junction[1] + math.sin(angle_rad) * length)
        branch_tips.append((tx, ty))
        
        sx = int(junction[0] + math.cos(angle_rad) * length * 0.4)
        sy = int(junction[1] + math.sin(angle_rad) * length * 0.4)
        sub_junctions.append((sx, sy))

    return stem_bottom, junction, sub_junctions, branch_tips

def draw_stem_segment(img, p1, p2, progress, base_color=(90, 190, 90)):
    if progress <= 0:
        return
    ex = int(lerp(p1[0], p2[0], progress))
    ey = int(lerp(p1[1], p2[1], progress))
    end = (ex, ey)
    cv2.line(img, p1, end, base_color, 7, cv2.LINE_AA)

def draw_pink_rose(canvas, cx, cy, radius, bloom):
    if bloom < 0.02 or radius < 5:
        return

    pad = int(radius * 1.5) + 8
    x1 = max(0, cx - pad);  y1 = max(0, cy - pad)
    x2 = min(canvas.shape[1], cx + pad);  y2 = min(canvas.shape[0], cy + pad)
    if x2 <= x1 or y2 <= y1:
        return

    patch   = canvas[y1:y2, x1:x2].copy()
    lcx, lcy = cx - x1, cy - y1
    
    max_r = radius * bloom
    num_layers = 6
    
    for layer in range(num_layers):
        layer_ratio = 1.0 - (layer / num_layers)
        layer_r = max_r * layer_ratio
        
        num_petals = 5 + layer * 2
        
        # Pink color gradient (BGR format)
        blue = int(lerp(220, 130, layer / num_layers))
        green = int(lerp(150, 40, layer / num_layers))
        red = int(lerp(255, 180, layer / num_layers))
        
        color = (blue, green, red)
        outline_color = (max(0, blue - 30), max(0, green - 30), max(0, red - 40))
        
        angle_offset = layer * math.pi / 3.5
        
        for p in range(num_petals):
            angle = angle_offset + 2 * math.pi * p / num_petals
            
            dist = layer_r * 0.5
            px = int(lcx + math.cos(angle) * dist)
            py = int(lcy + math.sin(angle) * dist)
            
            axes = (max(2, int(layer_r * 0.7)), max(2, int(layer_r * 0.5)))
            angle_deg = int(math.degrees(angle))
            
            # Draw directly on the patch to make it solid and opaque
            cv2.ellipse(patch, (px, py), axes, angle_deg, 0, 360, color, -1)
            cv2.ellipse(patch, (px, py), axes, angle_deg, 0, 360, outline_color, 1, cv2.LINE_AA)
            
    center_r = max(2, int(radius * 0.1 * bloom))
    cv2.circle(patch, (lcx, lcy), center_r, (80, 30, 150), -1)

    canvas[y1:y2, x1:x2] = patch

class PlantState:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.anchor_x = int(W * 0.2)
        self.stem_prog  = 0.0
        self.bloom_prog = 0.0
        self._stem_t    = 0.0
        self._bloom_t   = 0.0
        self.geo = build_plant(W, H, self.anchor_x)

    def update(self, left_ratio, right_ratio):
        if left_ratio  is not None: self._stem_t  = left_ratio
        if right_ratio is not None: self._bloom_t = right_ratio
        self.stem_prog  = smooth(self.stem_prog,  self._stem_t,  0.06)
        self.bloom_prog = smooth(self.bloom_prog, self._bloom_t, 0.06)

    def draw(self, canvas):
        stem_bot, junction, sub_jncts, tips = self.geo
        sp = self.stem_prog
        bp = self.bloom_prog

        if sp < 0.01:
            return

        main_prog = clamp(sp / 0.50, 0.0, 1.0)
        draw_stem_segment(canvas, stem_bot, junction, main_prog)

        if main_prog < 0.05:
            return

        branch_prog = clamp((sp - 0.50) / 0.50, 0.0, 1.0)

        flower_r = int(min(self.W, self.H) * 0.10)

        for i, (sj, tip) in enumerate(zip(sub_jncts, tips)):
            stagger = (i % 3) * 0.15
            denom   = 1.0 - stagger + 0.001
            bp_local = clamp((branch_prog - stagger) / denom, 0.0, 1.0)

            bp_phase_a = clamp(bp_local / 0.50, 0.0, 1.0)
            draw_stem_segment(canvas, junction, sj, bp_phase_a)

            bp_phase_b = clamp((bp_local - 0.50) / 0.50, 0.0, 1.0)
            if bp_phase_a > 0.01:
                draw_stem_segment(canvas, sj, tip, bp_phase_b)

            if bp_local > 0.65:
                flower_gate = clamp((bp_local - 0.65) / 0.35, 0.0, 1.0)
                effective_bloom = self.bloom_prog * flower_gate
                draw_pink_rose(canvas, tip[0], tip[1], flower_r, effective_bloom)

def draw_hud(canvas, sp, bp, l_det, r_det):
    h, w = canvas.shape[:2]
    f = cv2.FONT_HERSHEY_SIMPLEX

    def bar(label, val, y, col):
        cv2.rectangle(canvas, (12, y), (152, y + 9), (40, 40, 40), -1)
        cv2.rectangle(canvas, (12, y), (12 + int(140 * val), y + 9), col, -1)
        cv2.rectangle(canvas, (12, y), (152, y + 9), (140, 140, 140), 1)
        cv2.putText(canvas, label, (12, y - 3), f, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

    bar("LEFT  -> stem growth",  sp, 28, (130, 200, 130))
    bar("RIGHT -> rose bloom", bp, 55, (200,  100, 255))

    dl = (0, 220, 80) if l_det else (50, 50, 50)
    dr = (0, 220, 80) if r_det else (50, 50, 50)
    cv2.circle(canvas, (w - 28, 18), 6, dl, -1)
    cv2.circle(canvas, (w - 13, 18), 6, dr, -1)
    cv2.putText(canvas, "L", (w - 31, 22), f, 0.28, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "R", (w - 16, 22), f, 0.28, (0, 0, 0), 1, cv2.LINE_AA)

    hint = "Left Pinch = Grow Stems  |  Right Pinch = Bloom Roses  |  Press 'Q' to Quit"
    cv2.putText(canvas, hint, (10, h - 10), f, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("No frame from webcam")
        return
    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]

    plant = PlantState(W, H)

    print("Running Pink Rose AR Hand Gesture - Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb)

        left_ratio  = None
        right_ratio = None
        l_det = r_det = False

        if result.multi_hand_landmarks and result.multi_handedness:
            for lm_set, handedness in zip(result.multi_hand_landmarks,
                                           result.multi_handedness):
                label = classify_hand(handedness.classification[0].label)
                lm    = lm_set.landmark
                ratio = pinch_ratio(lm, W, H)

                if label == "left":
                    left_ratio = ratio
                    l_det = True
                    tx = int(lm[4].x * W); ty = int(lm[4].y * H)
                    ix = int(lm[8].x * W); iy = int(lm[8].y * H)
                    cv2.line(frame, (tx, ty), (ix, iy), (130, 200, 130), 1, cv2.LINE_AA)
                    cv2.circle(frame, (tx, ty), 4, (130, 200, 130), -1)
                    cv2.circle(frame, (ix, iy), 4, (130, 200, 130), -1)

                elif label == "right":
                    right_ratio = ratio
                    r_det = True
                    tx = int(lm[4].x * W); ty = int(lm[4].y * H)
                    ix = int(lm[8].x * W); iy = int(lm[8].y * H)
                    cv2.line(frame, (tx, ty), (ix, iy), (200, 100, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (tx, ty), 4, (200, 100, 255), -1)
                    cv2.circle(frame, (ix, iy), 4, (200, 100, 255), -1)

        plant.update(left_ratio, right_ratio)
        plant.draw(frame)
        draw_hud(frame, plant.stem_prog, plant.bloom_prog, l_det, r_det)

        cv2.imshow("Pink Rose AR Bouquet", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()