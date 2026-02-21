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
# MODEL DEFINITIONS
# --------------------------------------------------

class SharedBilateralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 4)

    def forward(self, x, lesion_side=None, lesion_severity=0.0):
        h = self.hidden(x)

        # Simulate hemispheric lesion
        if lesion_side == "Left":
            h[:, :32] *= (1 - lesion_severity)
        elif lesion_side == "Right":
            h[:, 32:] *= (1 - lesion_severity)

        h = self.relu(h)
        return self.output(h)


class SplitBilateralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.left_hidden = nn.Linear(2, 32)
        self.right_hidden = nn.Linear(2, 32)

        self.left_out = nn.Linear(32, 2)
        self.right_out = nn.Linear(32, 2)

        self.relu = nn.ReLU()

    def forward(self, x, lesion_side=None, lesion_severity=0.0):

        left_h = self.left_hidden(x)
        right_h = self.right_hidden(x)

        if lesion_side == "Left":
            left_h *= (1 - lesion_severity)
        elif lesion_side == "Right":
            right_h *= (1 - lesion_severity)

        left_h = self.relu(left_h)
        right_h = self.relu(right_h)

        left_out = self.left_out(left_h)
        right_out = self.right_out(right_h)

        return torch.cat([left_out, right_out], dim=1)

# --------------------------------------------------
# MODEL LOADING FUNCTION (SAFE)
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
        st.error("Model loading failed. Likely architecture mismatch.")
        st.write("Error details:")
        st.write(e)
        st.stop()

    model.eval()
    return model

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("Controls")

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
# INITIALIZE MODEL
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

# Left Arm (base -1)
lx1, ly1, lx2, ly2 = forward_kinematics(
    angles[0], angles[1], base_x=-1
)

# Right Arm (base +1)
rx1, ry1, rx2, ry2 = forward_kinematics(
    angles[2], angles[3], base_x=1
)

# Plot arms
ax.plot([-1, lx1, lx2], [0, ly1, ly2], linewidth=4)
ax.plot([1, rx1, rx2], [0, ry1, ry2], linewidth=4)

# Plot target
ax.scatter(target_x, target_y, s=120, marker="x")

ax.set_title("Bilateral Arm Movement")

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
# EXPLANATION PANEL
# --------------------------------------------------

st.markdown("---")

st.markdown("""
### 🧠 What You're Seeing

- **Shared Network:** Both arms depend on a common hidden representation.
- **Split Network:** Each arm is controlled by a separate neural pathway.
- **Stroke Severity:** Simulates damage by weakening hidden neurons.
- Observe how different architectures respond to damage.
""")
