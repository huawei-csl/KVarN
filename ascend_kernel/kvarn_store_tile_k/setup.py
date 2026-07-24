import os 
import platform 
import shutil 
from setuptools import setup, Extension
from torch.utils import cpp_extension


# # some global variables 
ascend_libraries = ['runtime', 'ascendcl', 'stdc++', 'hccl', 'nnopbase', 'opapi', 'msprofiler', 'cust_opapi']
ascend_toolkit_install_path = os.environ['ASCEND_HOME_PATH']
custom_op_path = os.path.dirname(os.path.abspath(__file__))
machine = platform.machine()
if machine!='x86_64':
    machine = 'aarch64'

print('Ascend Home:', ascend_toolkit_install_path)
print('Custom Op:', custom_op_path)
print('CPU arch:', machine)

pytorch_extension_name = 'custop_torchapi'
pytorch_interface_cpp_files = ['custop_torchapi.cpp']


# build pytorch extension 
import torch 
import torch_npu
torch_npu_path = torch_npu.__file__
torch_npu_path = '/'.join(torch_npu_path.split('/')[:-1])
custop_lib = custom_op_path + '/vendors/customize/op_api/lib/'
extra_link_args = [
    f'-Wl,-rpath={torch_npu_path}/lib',
    f'-Wl,-rpath={custop_lib}'
]
setup(name=pytorch_extension_name,
      ext_modules=[cpp_extension.CppExtension(pytorch_extension_name, pytorch_interface_cpp_files,
                                              include_dirs=[
                                                  torch_npu_path + '/include',
                                                  custom_op_path + '/vendors/customize/op_api/include/',
                                                  ascend_toolkit_install_path + f'/{machine}-linux/include',
                                                  ascend_toolkit_install_path + '/acllib/include',
                                                  ],
                                              library_dirs=[
                                                  torch_npu_path + '/lib',
                                                  ascend_toolkit_install_path + '/runtime/lib64',
                                                  ascend_toolkit_install_path + f'/{machine}-linux/lib64',
                                                  custom_op_path + '/vendors/customize/op_api/lib/',
                                                  ],
                                              libraries= ascend_libraries + ['torch_npu'],
                                              extra_link_args=extra_link_args,
                                              )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
