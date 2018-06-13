class FLAGS(object):

    """
    General settings
    """
    input_size = 512
    heatmap_size = 64
    cpm_stages = 6
    num_of_joints = 24
    color_channel = 'RGB'
    joint_gaussian_variance = 1.0
    normalize_img = True
    use_gpu = True
    gpu_id = '1'
    if_show = False
    img_show_iters = 100

    """
       Training settings
    """
    network_def = 'RefineNet'

    pretrained_model = './logs_and_weights_refine2/models/weights/RefineNet/input_512_output_64/joints_24/stages_6/init_0_rate_0_step_0'
    datagenerator_config_file = './preprocess/config.cfg'

    batch_size = 16

    init_lr = 0
    lr_decay_rate = 0
    lr_decay_step = 0

    total_epoch = 45
    total_num = 40000
    training_iters = int(total_num * total_epoch / batch_size)
    validation_iters = 100
    model_save_iters = 2500
