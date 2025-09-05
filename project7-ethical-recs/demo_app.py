import streamlit as st
import numpy as np
from src.models.mf import MLP if False else None  # placeholder; wire your MF
from src.train import main as train_stub  # or load pre-trained factors

st.title("ConsentfulRec — steer your recommendations")
diversity = st.slider("Diversity weight (λ)", 0.0, 1.0, 0.3, 0.05)
no_horror = st.checkbox("Exclude horror content", value=True)
long_tail = st.slider("Min. long-tail exposure (%)", 0, 30, 10)

# demo slate (mocked for now)
np.random.seed(42)
items = [f"Movie #{i}" for i in np.random.choice(10000, 10, replace=False)]
if no_horror:
  items = [it for it in items if hash(it) % 7 != 0]
st.write("**Your slate:** ", ", ".join(items[:10]))
st.caption(f"λ={diversity}  •  long-tail≥{long_tail}%  •  filters: {'no-horror' if no_horror else 'none'}")
