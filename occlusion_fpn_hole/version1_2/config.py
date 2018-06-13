class FLAGS(object):
    
    """
    General settings
    """
    input_size = 512
    heatmap_size = 128
    cpm_stages = 6
    num_of_joints = 24
    color_channel = 'RGB'
    joint_gaussian_variance = 1.0
    normalize_img = False
    use_gpu = True
    if_show = True
    img_show_iters = 100

    """
       Training settings
    """
    network_def = 'cpm_hole'

    pretrained_model = './logs_and_weights_hole/models/weights/cpm_hole/input_512_output_64/joints_24/stages_6/init_0_rate_0_step_0'
    datagenerator_config_file = './preprocess/config.cfg'

    batch_size = 8

    init_lr = 0
    lr_decay_rate = 0
    lr_decay_step = 0

    total_epoch = 40
    total_num = 40000
    training_iters = int(total_num * total_epoch / batch_size)
    validation_iters = 100
    model_save_iters = 10000
