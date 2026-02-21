import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Bilateral Motor Network",
    layout="wide"
)

st.title("🧠 Bilateral Motor Network Simulator")
st.markdown("Explore shared vs split neural control and simulate stroke lesions.")

# --------------------------------------------------
# ARM PARAMETERS
# --------------------------------------------------

L1 = 1.0
L2 = 1.0

def forward_kinematics(theta1, theta2, base_x=0):
    x1 = base_x + L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    return x1, y1, x2, y2

# --------------------------------------------------
# SHARED MODEL (MATCHES TRAINING)
# --------------------------------------------------

class SharedBilateralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.sensory_cortex = nn.Linear(2, 64)
        self.association_cortex = nn.Linear(64, 64)
        self.motor_cortex = nn.Linear(64, 4)

        self.relu = nn.ReLU()

    def forward(self, x, lesion_side=None, lesion_severity=0.0):

        s = self.relu(self.sensory_cortex(x))
        a = self.relu(self.association_cortex(s))

        # Lesion applied at association cortex
        if lesion_side == "Left":
            a[:, :32] *= (1 - lesion_severity)
        elif lesion_side == "Right":
            a[:, 32:] *= (1 - lesion_severity)

        out = self.motor_cortex(a)
        return out

# --------------------------------------------------
# SPLIT MODEL (MATCHES TRAINING)
# --------------------------------------------------

class SplitBilateralNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Left hemisphere
        self.left_sensory = nn.Linear(2, 32)
        self.left_association = nn.Linear(32, 32)
        self.left_motor = nn.Linear(32, 2)

        # Right hemisphere
        self.right_sensory = nn.Linear(2, 32)
        self.right_association = nn.Linear(32, 32)
        self.right_motor = nn.Linear(32, 2)

        self.relu = nn.ReLU()

    def forward(self, x, lesion_side=None, lesion_severity=0.0):

        # Left pathway
        ls = self.relu(self.left_sensory(x))
        la = self.relu(self.left_association(ls))

        # Right pathway
        rs = self.relu(self.right_sensory(x))
        ra = self.relu(self.right_association(rs))

        # Apply lesion
        if lesion_side == "Left":
            la *= (1 - lesion_severity)
        elif lesion_side == "Right":
            ra *= (1 - lesion_severity)

        left_out = self.left_motor(la)
        right_out = self.right_motor(ra)

        return torch.cat([left_out, right_out], dim=1)

# --------------------------------------------------
# SAFE MODEL LOADING
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(model, filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        st.error(f"Model file not found: {filename}")
        st.stop()

    try:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error("Model loading failed.")
        st.write("Error details:")
        st.write(e)
        st.stop()

    model.eval()
    return model

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("🧠 Brain Controls")

model_type = st.sidebar.radio(
    "Network Type",
    ["Shared Network", "Split Network"]
)

lesion_side = st.sidebar.selectbox(
    "Stroke Side",
    ["None", "Left", "Right"]
)

lesion_severity = st.sidebar.slider(
    "Stroke Severity",
    0.0, 1.0, 0.0, 0.01
)

target_x = st.sidebar.slider(
    "Target X",
    -1.5, 1.5, 0.5, 0.01
)

target_y = st.sidebar.slider(
    "Target Y",
    -1.5, 1.5, 0.5, 0.01
)

if lesion_side == "None":
    lesion_side = None

# --------------------------------------------------
# LOAD CORRECT MODEL
# --------------------------------------------------

if model_type == "Shared Network":
    model = SharedBilateralNet()
    model = load_model(model, "shared_model.pth")
else:
    model = SplitBilateralNet()
    model = load_model(model, "split_model.pth")

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

input_tensor = torch.tensor([[target_x, target_y]], dtype=torch.float32)

with torch.no_grad():
    angles = model(
        input_tensor,
        lesion_side=lesion_side,
        lesion_severity=lesion_severity
    )

angles = angles.numpy()[0]

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Left arm (base -1)
lx1, ly1, lx2, ly2 = forward_kinematics(
    angles[0], angles[1], base_x=-1
)

# Right arm (base +1)
rx1, ry1, rx2, ry2 = forward_kinematics(
    angles[2], angles[3], base_x=1
)

# Plot arms
ax.plot([-1, lx1, lx2], [0, ly1, ly2], linewidth=4)
ax.plot([1, rx1, rx2], [0, ry1, ry2], linewidth=4)

# Plot target
ax.scatter(target_x, target_y, s=120, marker="x")

ax.set_title("Bilateral Arm Movement Simulation")

st.pyplot(fig)

# --------------------------------------------------
# ERROR METRICS
# --------------------------------------------------

left_error = np.sqrt((lx2 - target_x)**2 + (ly2 - target_y)**2)
right_error = np.sqrt((rx2 - target_x)**2 + (ry2 - target_y)**2)

col1, col2 = st.columns(2)
col1.metric("Left Arm Error", f"{left_error:.3f}")
col2.metric("Right Arm Error", f"{right_error:.3f}")

# --------------------------------------------------
# EXPLANATION
# --------------------------------------------------

st.markdown("---")
st.markdown("""
### 🧠 What You're Observing

- **Shared Network:** Both arms rely on a common association cortex.
- **Split Network:** Each arm has its own independent cortical pathway.
- **Stroke Severity:** Weakens neurons in one hemisphere.
- Observe how architecture affects robustness and compensation.
""")
