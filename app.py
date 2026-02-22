import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")
st.title("🧠 Bilateral Motor Network Simulator")

# =====================================================
# MODEL DEFINITIONS (EXACT MATCH TO TRAINING)
# =====================================================

class SimpleIntegratedBrain(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=12):
        super().__init__()

        self.sensory_cortex = nn.Linear(input_dim, hidden_dim)
        self.association_cortex = nn.Linear(hidden_dim, hidden_dim)
        self.motor_cortex = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, lesion_side=None, lesion_severity=0.0):

        h1 = self.act(self.sensory_cortex(x))
        h2 = self.act(self.association_cortex(h1))

        # Lesion at association cortex
        if lesion_side == "Left":
            h2[:, :32] *= (1 - lesion_severity)
        elif lesion_side == "Right":
            h2[:, 32:] *= (1 - lesion_severity)

        return self.tanh(self.motor_cortex(h2))


class BilateralSplitBrain(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        half_dim = int(hidden_dim / 2)

        self.encoder_left = nn.Linear(4, half_dim)
        self.encoder_right = nn.Linear(4, half_dim)

        self.shared_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.shared_layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.motor_left = nn.Linear(hidden_dim, 6)
        self.motor_right = nn.Linear(hidden_dim, 6)

        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lesion_side=None, lesion_severity=0.0):

        input_L = x[:, 0:4]
        input_R = x[:, 4:8]

        h_L = self.act(self.encoder_left(input_L))
        h_R = self.act(self.encoder_right(input_R))

        h_combined = torch.cat([h_L, h_R], dim=1)

        h_shared = self.act(self.shared_layer1(h_combined))
        h_shared = self.act(self.shared_layer2(h_shared))

        # Lesion at shared integration area
        if lesion_side == "Left":
            h_shared[:, :32] *= (1 - lesion_severity)
        elif lesion_side == "Right":
            h_shared[:, 32:] *= (1 - lesion_severity)

        out_L = self.sigmoid(self.motor_left(h_shared))
        out_R = self.sigmoid(self.motor_right(h_shared))

        return torch.cat([out_L, out_R], dim=1)

# =====================================================
# SAFE LOADING
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(model, filename):
    path = os.path.join(BASE_DIR, filename)

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🧠 Brain Controls")

model_type = st.sidebar.radio(
    "Network Type",
    ["Shared (Integrated Brain)", "Split (Bilateral Brain)"]
)

lesion_side = st.sidebar.selectbox(
    "Stroke Side",
    ["None", "Left", "Right"]
)

severity = st.sidebar.slider(
    "Stroke Severity",
    0.0, 1.0, 0.0, 0.01
)

if lesion_side == "None":
    lesion_side = None

# =====================================================
# LOAD MODEL
# =====================================================

if model_type.startswith("Shared"):
    model = load_model(SimpleIntegratedBrain(), "shared_model.pth")
else:
    model = load_model(BilateralSplitBrain(), "split_model.pth")

# =====================================================
# FAKE DEMO INPUT (since real dataset not loaded)
# =====================================================

st.sidebar.markdown("### Target Inputs")

left_x = st.sidebar.slider("Left Target X", -1.0, 1.0, 0.5)
left_y = st.sidebar.slider("Left Target Y", -1.0, 1.0, 0.5)
left_z = st.sidebar.slider("Left Target Z", -1.0, 1.0, 0.5)
left_reach = st.sidebar.slider("Left Reach", 0.0, 1.0, 0.5)

right_x = st.sidebar.slider("Right Target X", -1.0, 1.0, 0.5)
right_y = st.sidebar.slider("Right Target Y", -1.0, 1.0, 0.5)
right_z = st.sidebar.slider("Right Target Z", -1.0, 1.0, 0.5)
right_reach = st.sidebar.slider("Right Reach", 0.0, 1.0, 0.5)

input_tensor = torch.tensor([[
    left_x, left_y, left_z, left_reach,
    right_x, right_y, right_z, right_reach
]], dtype=torch.float32)

# =====================================================
# INFERENCE
# =====================================================

with torch.no_grad():
    output = model(input_tensor, lesion_side, severity)

output = output.numpy()[0]

# =====================================================
# DISPLAY OUTPUT
# =====================================================

st.subheader("💪 Muscle Activations (12 Outputs)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Left Arm Muscles")
    for i in range(6):
        st.metric(f"Muscle L{i+1}", f"{output[i]:.3f}")

with col2:
    st.markdown("### Right Arm Muscles")
    for i in range(6):
        st.metric(f"Muscle R{i+1}", f"{output[i+6]:.3f}")

st.markdown("---")
st.markdown("""
### 🧠 What You Are Seeing

- **Shared Brain:** One integrated sensory → association → motor pathway.
- **Split Brain:** Separate encoders + shared integration + separate motor heads.
- Stroke weakens half of the shared integration space.
- Watch how muscle activation patterns degrade differently.
""")
