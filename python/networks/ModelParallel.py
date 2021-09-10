import torch
import torch.nn as nn
import threading


def distribute_module(module, device):
    return module.cuda(device)


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(kwargs_tup) == len(modules)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        raise VauleError('devices is None')

    lock = threading.Lock()
    results = {}
    #grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        # torch.set_grad_enabled(grad_enabled)
        try:
            with torch.cuda.device(device):
                output = module(input)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class ModelParallel(nn.Module):

    def __init__(self, model, device_ids=None, output_device=None):
        super(ModelParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = model
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
            if not hasattr(model, 'module'):
                raise ValueError("model does not has module attribute")
            if len(device_ids) < len(model.module):
                print('warning: number of devices is not enough for module parallel')
            else:
                device_ids = device_ids[:len(model.module)]

        if output_device is None:
            output_device = device_ids[0]
        self.output_device = output_device
        self.device_ids = device_ids
        self.module = model.module  # module is a list
        self.distribute(self.module, device_ids)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        outputs = self.parallel_apply(self.module, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def distribute(self, module, device_ids):
        return [distribute_module(m, id) for m, id in zip(module, device_ids)]

    def scatter(self, inputs, kwargs, device_ids):
        if len(inputs) == 1:
            inputs = [inputs[0].cuda(id) for id in device_ids]
        else:
            inputs = [input.cuda(id) for input, id in zip(inputs, device_ids)]
        kwargs = None
        inputs = tuple(inputs)
        return inputs, kwargs

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids)

    def gather(self, outputs, output_device):
        outputs = [output.cuda(output_device) for output in outputs]
        return outputs
