import os
import torch
from options.test_options import TestOptions
from models import create_model

if __name__ == '__main__':
    opt = TestOptions().parse()

    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    input_data = torch.randn(1, 3, 256, 256)
    print(input_data.shape)

    model = create_model(opt)
    model.setup(opt)
    model.eval()
    model.convert_to_onnx('G_A', input_data, '/home/cwy/Codes/dl/pytorch-CycleGAN-and-pix2pix/'
                                             'onnx/oxford_resnet/oxford_G_A.onnx')
