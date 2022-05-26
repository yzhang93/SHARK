# Copyright 2020 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import iree.runtime as ireert
import iree.runtime.scripts.iree_benchmark_module as benchmark_module
import iree.compiler as ireec
import subprocess
import numpy as np
import os
from shark.torch_mlir_utils import get_module_name_for_asm_dump
import re

IREE_DEVICE_MAP = {
    "cpu": "dylib",
    "gpu": "cuda",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "vulkan"
}

UNIT_TO_SECOND_MAP = {
    "ms": 0.001,
    "s": 1
}

def check_device_drivers(device):
    """Checks necessary drivers present for gpu and vulkan devices"""
    if (device in ["gpu", "cuda"]):
        try:
            subprocess.check_output('nvidia-smi')
        except Exception:
            return True
    elif (device in ["metal", "vulkan"]):
        try:
            subprocess.check_output('vulkaninfo')
        except Exception:
            return True
    elif (device == "cpu"):
        return False
    # Unknown device.
    else:
        return True

    return False


def get_iree_compiled_module(module, device: str):
    """Given an mlir module returns the compiled .vmfb"""
    args = ["--iree-llvm-target-cpu-features=host"]
    if device == "cpu":
        find_triple_cmd = "uname -s -m"
        os_name, proc_name = subprocess.run(
            find_triple_cmd, shell=True, stdout=subprocess.PIPE,
            check=True).stdout.decode('utf-8').split()
        if os_name == "Darwin":
            find_kernel_version_cmd = "uname -r"
            kernel_version = subprocess.run(find_kernel_version_cmd,
                                            shell=True,
                                            stdout=subprocess.PIPE,
                                            check=True).stdout.decode('utf-8')
            target_triple = f"{proc_name}-apple-darwin{kernel_version}"
        elif os_name == "Linux":
            target_triple = f"{proc_name}-linux-gnu"
        else:
            error_message = f"OS Type f{os_name} not supported and triple can't be determined, open issue to dSHARK team please :)"
            raise Exception(error_message)
        print(f"Target triple found:{target_triple}")
        args.append(f"-iree-llvm-target-triple={target_triple}")

    if device in ["gpu", "cuda"]:
        args += ["--iree-hal-cuda-disable-loop-nounroll-wa"]
        ireert.flags.FUNCTION_INPUT_VALIDATION = False
        ireert.flags.parse_flags("--cuda_allow_inline_execution")

    if device in ["vulkan", "metal"]:
        args += ["--iree-flow-demote-i64-to-i32=false", "--iree-flow-demote-f64-to-f32=true"]

    flatbuffer_blob = ireec.compile_str(
        str(module), target_backends=[IREE_DEVICE_MAP[device]], extra_args=args)
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    config = ireert.Config(IREE_DEVICE_MAP[device])
    ctx = ireert.SystemContext(config=config)
    # TODO add optimisation args.
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module["forward"]
    return ModuleCompiled, config


def export_iree_module_to_vmfb(module, device: str, directory: str):
    module_name = get_module_name_for_asm_dump(module)
    flatbuffer_blob = ireec.compile_str(
        str(module), target_backends=[IREE_DEVICE_MAP[device]])
    filename = os.path.join(directory, module_name + ".vmfb")
    with open(filename, 'wb') as f:
        f.write(flatbuffer_blob)
    return filename


def get_results(compiled_vm, input, config):
    """Runs a .vmfb file given inputs and config and returns output."""
    device_inputs = [ireert.asdevicearray(config.device, a) for a in input]
    result = compiled_vm(*device_inputs)
    result_tensors = []
    if (isinstance(result, tuple)):
        for val in result:
            result_tensors.append(np.copy(np.asarray(val, val.dtype)))
        return result_tensors
    else:
        return np.copy(np.asarray(result, dtype=result.dtype))

######### Benchmark Related Tools ###########
def tensor_to_type_str(input_tensors : tuple):
    # TODO: Support more than floats, and ints
    list_of_type = []
    for input_tensor in input_tensors:
        type_string = "x".join([str(dim) for dim in input_tensor.shape])
        dtype_string = str(input_tensor.dtype).replace("torch.","")
        regex_split = re.compile("([a-zA-Z]+)([0-9]+)")
        match = regex_split.match(dtype_string)
        mlir_type_string = str(match.group(1)[0])+str(match.group(2))
        type_string += f"x{mlir_type_string}"
        list_of_type.append(type_string)
    return list_of_type

def build_benchmark_args(input_file : str, device : str, input_tensors : tuple, training=False):
    path = benchmark_module.__path__[0]
    benchmark_cl = [os.path.join(path, "..", "..",
                        "iree-benchmark-module"), f"--module_file={input_file}"]
    fn_name = "forward"
    if training == True:
        # TODO: Replace name of train with actual train fn name.
        fn_name = "train"
    benchmark_cl.append(f"--entry_function={fn_name}")
    benchmark_cl.append(f"--driver={IREE_DEVICE_MAP[device]}")
    mlir_input_types = tensor_to_type_str(input_tensors)
    for mlir_input in mlir_input_types:
        benchmark_cl.append(f"--function_input={mlir_input}")
    time_extractor = "| awk \'END{{print $2 $3}}\'"
    benchmark_cl.append(time_extractor)
    return benchmark_cl

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        result_str = result.stdout.decode()
        return result_str
    except Exception:
        sys.exit("Exiting program due to error running:", cmd)

def run_benchmark(benchmark_cl):
    """ Outputs iteration per second"""
    bench_result = run_cmd(' '.join(benchmark_cl))
    regex_split = re.compile("([0-9]+[.]*[0-9]*)([a-zA-Z]+)")
    match = regex_split.match(bench_result)
    time = float(match.group(1))
    unit = match.group(2)
    return 1.0/(time*UNIT_TO_SECOND_MAP[unit])
