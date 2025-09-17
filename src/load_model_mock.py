from tlob_model import TLOB, TLOBConfig
import torch, json

cfg = json.load(open("../logs/.../model_config.json"))
model = TLOB(TLOBConfig(**cfg))
state = torch.load("../logs/.../model_best.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
