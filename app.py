import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ==============================
# Arm Geometry
# ==============================

L1 = 1.0
L2 = 1.0

def forward_kinematics(theta1, theta2, base_x=0):
    x1 = base_x + L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    return x1, y1, x2, y2


# ==============================
# Model Architectures
# ==============================

class SharedBilateralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 4)

    def forward(self, x, lesion_severity=0.0, lesion_side=None):
        h = self.hidden(x)

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

    def forward(self, x, lesion_severity=0.0, lesion_side=None):

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


# ==============================
# Sidebar Controls
# ==============================

st.sidebar.title("🧠 Brain Controls")

model_type = st.sidebar.radio("Network Type",
                              ["Shared Network", "Split Network"])

lesion_side = st.sidebar.selectbox("Stroke Side",
                                   ["None", "Left", "Right"])

lesion_severity = st.sidebar.slider("Stroke Severity",
                                    0.0, 1.0, 0.0)

target_x = st.sidebar.slider("Target X", -1.5, 1.5, 0.5)
target_y = st.sidebar.slider("Target Y", -1.5, 1.5, 0.5)

# ==============================
# Load Model
# ==============================

if model_type == "Shared Network":
    model = SharedBilateralNet()
    model.load_state_dict(torch.load("shared_model.pth"))
else:
    model = SplitBilateralNet()
    model.load_state_dict(torch.load("split_model.pth"))

model.eval()

# ==============================
# Inference
# ==============================

x_input = torch.tensor([[target_x, target_y]], dtype=torch.float32)

if lesion_side == "None":
    lesion_side = None

angles = model(x_input,
               lesion_severity=lesion_severity,
               lesion_side=lesion_side)

angles = angles.detach().numpy()[0]

# ==============================
# Visualization
# ==============================

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")

# Left Arm (base -1)
lx1, ly1, lx2, ly2 = forward_kinematics(
    angles[0], angles[1], base_x=-1)

# Right Arm (base +1)
rx1, ry1, rx2, ry2 = forward_kinematics(
    angles[2], angles[3], base_x=1)

# Plot Arms
ax.plot([-1, lx1, lx2], [0, ly1, ly2], linewidth=4)
ax.plot([1, rx1, rx2], [0, ry1, ry2], linewidth=4)

# Plot Target
ax.scatter(target_x, target_y, s=100)
ax.set_title("Bilateral Motor Control Simulation")

st.pyplot(fig)

# ==============================
# Error Display
# ==============================

left_error = np.sqrt((lx2 - target_x)**2 + (ly2 - target_y)**2)
right_error = np.sqrt((rx2 - target_x)**2 + (ry2 - target_y)**2)

col1, col2 = st.columns(2)
col1.metric("Left Arm Error", f"{left_error:.3f}")
col2.metric("Right Arm Error", f"{right_error:.3f}")
