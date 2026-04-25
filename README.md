# dog-breed-detection
EfficientNetB0 + Yolo Detection + Cat Detection<br>

Can work with multiple Dogs<br>

Can detect Zebra, etc.<br>
breeds_info is having lesser info than breed_database.py<br>
<br>
<br>
Mini Proj IV Repo <br><br>
train_model.ipynb file contains training a MobileNetV2 model<br>
Total layers in the full model: 162 layers<br>
Total layers in the base MobileNetV2 model: 154 layers<br>
Regarding the freezing strategy:<br>

Phase 1 (Training Top Layers): The entire base_model (MobileNetV2, which has 154 layers) was frozen (base_model.trainable = False). Only the newly added classification head layers were trained.<br>
154 frozen + 8 trained<br>
Phase 2 (Fine-tuning Entire Model): The base_model was partially unfrozen. Specifically, the last 30 layers of the base_model were made trainable (base_model.trainable = True, then layers[:-30] set to False), while the initial 124 layers of MobileNetV2 remained frozen.<br>
124 frozen + last 30 base model layers
