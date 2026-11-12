import logging
import torch
import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from os import path as osp
# from torchsummary import summary
# from fvcore.nn import FlopCountAnalysis
from thop import profile, clever_format
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)

    logger.info("Calculating model parameters and FLOPs using thop...")
    dummy_input = torch.randn(1, 3, 128, 128).to(model.device)

    flops, params = profile(model.net_g, inputs=(dummy_input,), verbose=False)

    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

    logger.info(f"Total Parameters: {params_formatted}")
    logger.info(f"Total FLOPs: {flops_formatted}")
    logger.info(f"Parameters (raw): {params / 1e6:.2f} M")
    logger.info(f"FLOPs (raw): {flops / 1e9:.2f} G")

#     logger.info("Calculating model parameter count...")
#     try:
#         summary(model.net_g, input_size=(3, 128, 128))
#     except Exception as e:
#         logger.error(f"Error in calculating model summary: {e}")
#     logger.info("Calculating FLOPs...")
#     try:
#         dummy_input = torch.randn(1, 3, 128, 128).to(model.device)
#         flops = FlopCountAnalysis(model.net_g, dummy_input)
#         logger.info(f"FLOPs: {flops.total()/1e9} GFLOPs")
#     except Exception as e:
#         logger.error(f"Error in calculating FLOPs: {e}")
#     start_time = time.time()

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # wheather use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr, use_image=use_image)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000
        logger.info(f"Time for testing {test_set_name}: {elapsed_time:.4f} seconds")

    # total_time = time.time() - start_time
    # logger.info(f'Total time for testing: {total_time:.4f} seconds')

if __name__ == '__main__':
    main()
