import json

from torch.quantization.quantize_fx import prepare_fx, convert_fx
import torch
import platform
from torch.ao.quantization import QConfigMapping, QConfig, MinMaxObserver, PerChannelMinMaxObserver

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

    # Cuantización estatica solo funciona en CPU
    model = model.eval().cpu()

    if (conf["Cuant"]["mode"] == 'int8'):
        qconfig_mapping = QConfigMapping().set_global(
            torch.quantization.get_default_qconfig(backend) # Default config es INT8
        )

        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)

        print(f"Calibrando con {num_batches} batches...")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_loader):
                model_prepared(inputs.cpu())
                if i + 1 >= num_batches:
                    break
        print("Calibración completada.")

        model_quantized = convert_fx(model_prepared)
    else:
        model_quantized = model.half() # TODO: Ahora mismo esto crea problemas porque cambia el tipo de pesos

    return model_quantized.to(device)