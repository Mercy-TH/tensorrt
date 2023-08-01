import torch
import onnx
from onnxsim import simplify

model_path = '../model/resnet50.pt'
export_onnx_path = '../model/resnet.onnx'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
example_input = torch.rand(1, 3, 224, 224).to(device)


Model = torch.load(model_path)
Model.eval()
Model.to(device)


def export_to_onnx(model, export_onnx_path=export_onnx_path):
    torch.onnx.export(
        model,
        example_input,
        export_onnx_path,
        input_names=['input'],
        export_params=True,
        verbose=True,
        opset_version=11
    )
    onnx_model = onnx.load(export_onnx_path)
    model_simple, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simple, export_onnx_path)
    print('ok.')


export_to_onnx(Model)
