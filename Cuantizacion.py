import json

from torch.quantization.quantize_fx import prepare_fx, convert_fx
import torch
import platform
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx


with open("config.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

def cuantizar_estatica(model, device, calibration_loader, num_batches=10):

    # Seleccionar backend
    machine = platform.machine().lower()
    supported = torch.backends.quantized.supported_engines

    if machine in ['arm64', 'aarch64', 'arm']:
        preferred = ['qnnpack']
    elif machine in ['x86_64', 'amd64', 'i386', 'i686']:
        preferred = ['fbgemm', 'onednn', 'x86']
    else:
        raise SystemError(f"Arquitectura no soportada: {machine}")

    backend = next((b for b in preferred if b in supported), None)
    if backend is None:
        raise SystemError(f"Ningún backend disponible. Encontrados: {supported}")

    print(f"Usando backend {backend} para {machine}")
    torch.backends.quantized.engine = backend

    # Cuantización estática solo está soportada en CPU
    model = model.eval().cpu()

    qconfig_mapping = (
        QConfigMapping()
        .set_global(get_default_qconfig(backend))
        .set_object_type(torch.nn.ConvTranspose2d, None)
        #   ConvTranspose2d no se cuantiza porque no lo soporta torchao, también podría cuantizarse per-tensor,
        #   pero esto produce resultados bastante peores (perdida de 4.48% vs 0.15% en student)
    )

    example_inputs = (torch.randn(1, 2, 128, 128),)

    model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)

    print(f"Calibrando con {num_batches} batches...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_loader):
            model_prepared(inputs.cpu())
            if i + 1 >= num_batches:
                break
    print("Calibración completada.")

    model_quantized = convert_fx(model_prepared)

    return model_quantized.to(device)

def entrenar_qat(model, device, calibration_loader, num_batches=10):
    # Seleccionar backend
    machine = platform.machine().lower()
    supported = torch.backends.quantized.supported_engines

    if machine in ['arm64', 'aarch64', 'arm']:
        preferred = ['qnnpack']
    elif machine in ['x86_64', 'amd64', 'i386', 'i686']:
        preferred = ['fbgemm', 'onednn', 'x86']
    else:
        raise SystemError(f"Arquitectura no soportada: {machine}")

    backend = next((b for b in preferred if b in supported), None)
    if backend is None:
        raise SystemError(f"Ningún backend disponible. Encontrados: {supported}")

    print(f"Usando backend {backend} para {machine}")

    qconfig_mapping = (
        get_default_qat_qconfig_mapping("onednn")
        .set_object_type(torch.nn.ConvTranspose2d, None)
    )
    example_input = next(iter(calibration_loader))[0][:1]  # un batch de ejemplo

    student_qat = prepare_qat_fx(model, qconfig_mapping, example_input)
    student_qat.train()
    return student_qat

def cuantizar_qat(model):
    # Convertir a INT8 real
    model.eval()
    student_int8 = convert_fx(model)
    return student_int8

