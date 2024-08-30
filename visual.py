import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ptbxl.zqy.network import CapsuleNetwork

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        return cam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型
model = CapsuleNetwork().to(device)
model.load_state_dict(torch.load('capsule_network.pth'))
model.eval()

# 选择模型中的目标层
target_layer = model.conv1

# 创建Grad-CAM对象
grad_cam = GradCAM(model, target_layer)

# 加载数据
test_data = np.load('E:/search/daima/ptbxl/zqy/origin_test_data.npz')
X_test = [torch.tensor(test_data[f'X_test_{i}'], dtype=torch.float32) for i in range(12)]
y_test = torch.tensor(test_data['y_test'], dtype=torch.float32)

# 将12导联数据堆叠为一个三维张量，形状为 (batch_size, channels, length)
input_image = torch.stack(X_test, dim=1).unsqueeze(1).to(device)  # (1513, 1, 12, 5000)
target_class = y_test.argmax(dim=1)

# 选择一个样本进行可视化
sample_index = 3
sample_image = input_image[sample_index].unsqueeze(0)
sample_target = target_class[sample_index]

# 生成CAM
cam = grad_cam.generate_cam(sample_image, sample_target.item())

# 可视化所有导联
plt.figure(figsize=(15, 10))
num_leads = len(X_test)

for i in range(num_leads):
    plt.subplot(num_leads, 1, i + 1)
    lead_data = X_test[i][sample_index].cpu().numpy()
    plt.plot(lead_data, label=f'Lead {i + 1}', color='blue')

    # 使用CAM
    cam_resized = cv2.resize(cam, (lead_data.shape[0], 1)).squeeze()
    plt.imshow(cam_resized[np.newaxis, :], cmap='Purples', alpha=0.5, aspect='auto',
               extent=(0, lead_data.shape[0], np.min(lead_data), np.max(lead_data)))

    plt.legend(loc='upper right')

plt.suptitle("ECG Leads with Grad-CAM Overlay")
plt.savefig('ecg_grad_cam_overlay.png')
plt.show()
